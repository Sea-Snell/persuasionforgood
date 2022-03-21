import csv
from collections import defaultdict
from transformers import GPT2Tokenizer, GPT2Model
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
import json
import random
from torch.distributions.normal import Normal
from torch.distributions import Categorical
import wandb
import numpy as np
import spacy
import math

def listtobig5(x):
    return {'extrovert': x[0], 'agreeable': x[1], 'conscentious': x[2], 'neurotic': x[3], 'open': x[4]}

def big5tolist(x):
    return [x['extrovert'], x['agreeable'], x['conscentious'], x['neurotic'], x['open']]

def uniform_descretize_big5(list_big5, min_big5, max_big5, n_discrete):
    return [int(math.ceil(((item-min_big5)*n_discrete)/(max_big5-min_big5)))-1 for item in list_big5]

def tokenize_glove(spacy_tokenizer, tok2idx, text):
    text = text.lower()
    toks = [item.text for item in spacy_tokenizer(text)]
    return [tok2idx[tok] if tok in tok2idx else tok2idx['<unk>'] for tok in toks]

def glove_tokenizer(spacy_tokenizer, tok2idx):
    def _tokenizer(text):
        return tokenize_glove(spacy_tokenizer, tok2idx, text)
    return _tokenizer

def glove_decoder(idx2tok):
    def _detokenizer(tokens):
        return ' '.join([idx2tok[item] for item in tokens])
    return _detokenizer

def initalize_glove(glove_path):
    spacy_tokenizer = spacy.load("en_core_web_sm")
    glove_idx2tok = []
    glove_embs = []
    with open(glove_path, 'r') as f:
        for line in f:
            word, *emb = line.split()
            emb = np.array(emb)
            glove_embs.append(list(map(float, emb)))
            glove_idx2tok.append(word)
    glove_embs.append(np.zeros(50))
    glove_idx2tok.append('<unk>')
    glove_tok2idx = {word: i for i, word in enumerate(glove_idx2tok)}
    my_glove_tokenizer = glove_tokenizer(spacy_tokenizer, glove_tok2idx)
    my_glove_decoder = glove_decoder(glove_idx2tok)
    glove_embs = np.stack(glove_embs, axis=0)
    emb_layer = nn.Embedding(glove_embs.shape[0], glove_embs.shape[1])
    emb_layer.weight = nn.Parameter(torch.tensor(glove_embs.tolist()))
    for param in emb_layer.parameters():
        param.requires_grad = False
    return my_glove_tokenizer, my_glove_decoder, emb_layer, len(glove_idx2tok), glove_tok2idx['<unk>']

def initalize_gpt2():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    gpt2_embs = GPT2Model.from_pretrained('gpt2').resize_token_embeddings(len(tokenizer))
    for param in gpt2_embs.parameters():
        param.requires_grad = False
    return lambda x: tokenizer(x)['input_ids'], lambda x: tokenizer.decode(x), gpt2_embs, len(tokenizer), tokenizer.pad_token_id

class PersonalityClassifier(nn.Module):
    def __init__(self, emb_dim, h_dim, n_tokens, pad_token, init_embs=None):
        super().__init__()
        self.emb_dim = emb_dim
        self.h_dim = h_dim
        self.n_tokens = n_tokens
        self.pad_token = pad_token
        if init_embs is None:
            self.emb = nn.Embedding(self.n_tokens, self.emb_dim)
        else:
            self.emb = init_embs
        self.mean_mlp = nn.Sequential(
            nn.Linear(self.emb_dim, self.h_dim), 
            nn.ReLU(), 
            nn.Linear(self.h_dim, 5), 
        )
        self.log_uncertainty_mlp = nn.Sequential(
            nn.Linear(self.emb_dim, self.h_dim), 
            nn.ReLU(), 
            nn.Linear(self.h_dim, 5), 
        )

    def forward(self, tokens):
        mask = tokens != self.pad_token
        avg_emb = (self.emb(tokens) * mask.unsqueeze(2)).sum(dim=1) / mask.sum(dim=1).unsqueeze(1)
        # print(avg_emb.min(), avg_emb.max(), avg_emb.mean(), avg_emb.std())
        # print(self.log_uncertainty_mlp(avg_emb))
        # print(self.mean_mlp(avg_emb))
        return Normal(self.mean_mlp(avg_emb), torch.exp(self.log_uncertainty_mlp(avg_emb)))

    def eval(self, data_loader):
        loss = 0.0
        entropy = 0.0
        entropy_std = 0.0
        total = 0
        mean = np.zeros((5,))
        mean_std = np.zeros((5,))
        std = np.zeros((5,))
        std_std = np.zeros((5,))
        for tokens, classes in data_loader:
            tokens, classes = tokens.to(device), classes.to(device)
            predictions = self(tokens)
            l = -predictions.log_prob(classes).mean()
            loss += l.item() * tokens.shape[0]
            entropy += predictions.entropy().mean().item() * tokens.shape[0]
            entropy_std += predictions.entropy().std().item() * tokens.shape[0]
            mean += predictions.loc.mean(dim=0).detach().cpu().numpy() * tokens.shape[0]
            mean_std += predictions.loc.std(dim=0).detach().cpu().numpy() * tokens.shape[0]
            std += predictions.scale.mean(dim=0).detach().cpu().numpy() * tokens.shape[0]
            std_std += predictions.scale.std(dim=0).detach().cpu().numpy() * tokens.shape[0]
            total += tokens.shape[0]
        return {'loss': loss / total, 
                'entropy': entropy / total, 
                'entropy_std': entropy_std / total, 
                'mean': listtobig5((mean / total).tolist()), 
                'mean_std': listtobig5((mean_std / total).tolist()),
                'std': listtobig5((std / total).tolist()), 
                'std_std': listtobig5((std_std / total).tolist())}, None

class DiscretePersonalityClassifier(nn.Module):
    def __init__(self, n_discrete, emb_dim, h_dim, n_tokens, pad_token, detokenizer, k, init_embs=None):
        super().__init__()
        self.n_discrete = n_discrete
        self.emb_dim = emb_dim
        self.h_dim = h_dim
        self.n_tokens = n_tokens
        self.pad_token = pad_token
        self.detokenizer = detokenizer
        self.k = k
        if init_embs is None:
            self.emb = nn.Embedding(self.n_tokens, self.emb_dim)
        else:
            self.emb = init_embs
        self.logit_mlp = nn.Sequential(
            nn.Linear(self.emb_dim, self.h_dim), 
            nn.ReLU(), 
            nn.Linear(self.h_dim, 5*self.n_discrete), 
        )

    def forward(self, tokens):
        mask = tokens != self.pad_token
        avg_emb = (self.emb(tokens) * mask.unsqueeze(2)).sum(dim=1) / mask.sum(dim=1).unsqueeze(1)
        return Categorical(logits=self.logit_mlp(avg_emb).reshape(tokens.shape[0], -1, self.n_discrete))
    
    def eval(self, data_loader):
        loss = 0.0
        entropy = 0.0
        entropy_std = 0.0
        total = 0
        entropies = []
        for tokens, classes in data_loader:
            tokens, classes = tokens.to(device), classes.to(device)
            predictions = self(tokens)
            l = -predictions.log_prob(classes).mean()
            loss += l.item() * tokens.shape[0]
            e = predictions.entropy().sum(dim=1)
            entropy += e.mean().item() * tokens.shape[0]
            p = predictions.probs
            entropies.extend([(e[i].item(), tokens[i], p[i].detach().cpu().tolist(),) for i in range(tokens.shape[0])])
            entropy_std += e.std().item() * tokens.shape[0]
            total += tokens.shape[0]
        logs = {'loss': loss / total, 
                'entropy': entropy / total, 
                'entropy_std': entropy_std / total}
        entropies = sorted(entropies, key=lambda x: x[0])
        top_k_ents = entropies[-self.k:]
        top_k_ents = [(x[0], self.detokenizer(x[1][:(x[1] != self.pad_token).long().sum().item()].detach().cpu().tolist()), x[2]) for x in top_k_ents]
        bottom_k_ents = entropies[:self.k]
        bottom_k_ents = [(x[0], self.detokenizer(x[1][:(x[1] != self.pad_token).long().sum().item()].detach().cpu().tolist()), x[2]) for x in bottom_k_ents]
        str_ = ''
        str_ += 'top k:\n'
        str_ += '\n'.join(map(lambda x: str(x[0])+'\n'+('='*25)+'\n'+x[1]+'\n'+str(listtobig5(x[2]))+'\n'+('='*25), top_k_ents))
        str_ += '\nbottom k:\n'
        str_ += '\n'.join(map(lambda x: str(x[0])+'\n'+('='*25)+'\n'+x[1]+'\n'+str(listtobig5(x[2]))+'\n'+('='*25), bottom_k_ents))
        return logs, str_
        

class PersuasionData(Dataset):
    def __init__(self, info_path, dialogue_path, tokenizer, pad_token, split, frac_train, discrete, n_discrete) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_token = pad_token
        self.discrete = discrete
        self.n_discrete = n_discrete

        data = defaultdict(lambda: dict())
        with open(info_path, 'r') as f:
            info = csv.reader(f)
            for row in info:
                if row[0] == 'B2' or row[5] == '':
                    continue
                data[row[0]]['big5_%s'%(row[2])] = {'donation': float(row[3]), 'extrovert': float(row[5]), 
                                                    'agreeable': float(row[6]), 'conscentious': float(row[7]), 
                                                    'neurotic': float(row[8]), 'open': float(row[9])}
                data[row[0]]['dialogue'] = []
        with open(dialogue_path, 'r') as f:
            dialogue = csv.reader(f)
            for row in dialogue:
                if row[4] == 'B2':
                    continue
                if row[4] in data:
                    data[row[4]]['dialogue'].append({'role': int(row[3]), 'text': row[1]})
        
        self.datapoints = []
        for k in sorted(data.keys()):
            self.datapoints += self.prepare_data(data[k])
        if split == 'train':
            self.datapoints = self.datapoints[:int(frac_train*len(self.datapoints))]
        elif split == 'eval':
            self.datapoints = self.datapoints[int(frac_train*len(self.datapoints)):]
        else:
            raise NotImplementedError

    def prepare_data(self, data_item):
        examples = []
        if 'big5_0' in data_item:
            classes = big5tolist(data_item['big5_0'])
            # print(classes)
            if self.discrete:
                classes = uniform_descretize_big5(classes, 0, 5, self.n_discrete)
            # print(classes)
            text = '\n'.join([item['text'] for item in data_item['dialogue'] if item['role'] == 0])
            tokens = self.tokenizer(text)
            examples.append((text, tokens, classes,))
        if 'big5_1' in data_item:
            classes = big5tolist(data_item['big5_1'])
            # print(classes)
            if self.discrete:
                classes = uniform_descretize_big5(classes, 0, 5, self.n_discrete)
            # print(classes)
            text = '\n'.join([item['text'] for item in data_item['dialogue'] if item['role'] == 1])
            tokens = self.tokenizer(text)
            examples.append((text, tokens, classes,))
        return examples
    
    def __getitem__(self, i):
        return self.datapoints[i]
    
    def __len__(self):
        return len(self.datapoints)
    
    def collate(self, items):
        _, tokens, classes = list(zip(*items))
        tokens = nn.utils.rnn.pad_sequence(list(map(lambda x: torch.tensor(x), tokens)), 
                                           batch_first=True, 
                                           padding_value=self.pad_token)
        classes = torch.tensor(classes)
        return tokens, classes

class PersonaChatData(Dataset):
    def __init__(self, path, tokenizer, pad_token):
        self.tokenizer = tokenizer
        self.pad_token = pad_token
        with open(path, 'r') as f:
            data = json.load(f)
        self.datapoints = [self.prepare_item(item) for item in data]
    
    def prepare_item(self, item):
        persona = item['personality']
        text = '\n'.join([utterance['candidates'][-1] for utterance in item['utterances']])
        tokens = self.tokenizer(text)
        persona_tokens = self.tokenizer('\n'.join(item['personality']))
        return persona, persona_tokens, text, tokens
    
    def __getitem__(self, i):
        return self.datapoints[i]
    
    def __len__(self):
        return len(self.datapoints)
    
    def collate(self, items):
        _, tokens, classes = list(zip(*items))
        tokens = nn.utils.rnn.pad_sequence(list(map(lambda x: torch.tensor(x), tokens)), 
                                           batch_first=True, 
                                           padding_value=self.pad_token)
        classes = torch.tensor(classes)
        return tokens, classes

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    info_path = '../data/FullData/full_info.csv'
    dialogue_path = '../data/FullData/full_dialog.csv'
    persona_path = '../personachat_truecased/personachat_truecased_full_train.json'
    bsize = 32
    emb_dim = 768
    h_dim = 128
    lr = 1e-4
    weight_decay = 0.01
    epochs = 1000
    n_personas = 1
    use_wandb = False
    discrete = True
    n_discrete = 10
    emb_kind = 'gpt2'
    print_every = 100
    k = 5

    if use_wandb:
        wandb.init(project='personality_classifier')

    if emb_kind == 'gpt2':
        tokenizer, decoder, embs, n_toks, padding = initalize_gpt2()
    elif emb_kind == 'glove':
        tokenizer, decoder, embs, n_toks, padding = initalize_glove('../glove/glove.6B.50d.txt')
    else:
        raise NotImplementedError
    if not discrete:
        model = PersonalityClassifier(emb_dim, h_dim, n_toks, padding, embs).to(device)
    else:
        model = DiscretePersonalityClassifier(n_discrete, emb_dim, h_dim, n_toks, padding, decoder, k, embs).to(device)
    train_dataset = PersuasionData(info_path, dialogue_path, tokenizer, padding, 'train', 0.75, discrete, n_discrete)
    eval_dataset = PersuasionData(info_path, dialogue_path, tokenizer, padding, 'eval', 0.75, discrete, n_discrete)
    # persona_dataset = PersonaChatData(persona_path, tokenizer, padding)
    train_data_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True, collate_fn=train_dataset.collate)
    eval_data_loader = DataLoader(eval_dataset, batch_size=bsize, shuffle=True, collate_fn=eval_dataset.collate)
    optim = AdamW(model.parameters(), lr, weight_decay=weight_decay)

    # all_classes = []
    # for _, _, classes in PersuasionData(info_path, dialogue_path, tokenizer, padding, 'train', 1.0):
    #     all_classes.append(classes)
    # all_classes = np.array(all_classes)
    # dataset_avg = np.mean(all_classes, axis=0)
    # dataset_std = np.std(all_classes, axis=0)
    # print('dataset average:', listtobig5(dataset_avg.tolist()))
    # print('dataset std:', listtobig5(dataset_std.tolist()))

    for epoch in range(epochs):
        for tokens, classes in tqdm(train_data_loader):
            tokens, classes = tokens.to(device), classes.to(device)
            # print(classes)
            predictions = model(tokens)
            loss = -predictions.log_prob(classes).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
        
        train_logs, train_print_out = model.eval(train_data_loader)
        eval_logs, eval_print_out = model.eval(eval_data_loader)

        if (epoch+1) % print_every == 0:
            if train_print_out is not None:
                print('='*25)
                print('train outputs:')
                print('='*25)
                print(train_print_out)
                print('='*25)
            if eval_print_out is not None:
                print('='*25)
                print('eval outputs:')
                print('='*25)
                print(eval_print_out)
                print('='*25)

        # persona_matches = []
        # for idx in random.sample(range(len(persona_dataset)), n_personas):
        #     persona, persona_tokens, text, tokens = persona_dataset[idx]
        #     tokens, persona_tokens = torch.tensor(tokens).to(device), torch.tensor(persona_tokens).to(device)
        #     predictions = model(tokens.unsqueeze(0)).loc[0].detach().cpu().tolist()
        #     persona_predictions = model(persona_tokens.unsqueeze(0)).loc[0].detach().cpu().tolist()
        #     persona_matches.append({
        #                             'predictions': listtobig5(predictions), 
        #                             'persona_predictions': listtobig5(persona_predictions), 
        #                             'persona': persona, 
        #                             'utterances': text})
        print('epoch:', epoch)
        results = {'train': train_logs, 
                    'eval': eval_logs, 
                  }
        if use_wandb:
            wandb.log({'epoch': epoch, **results})
        print('train:', results['train'])
        print('eval:', results['eval'])
        # for match in persona_matches:
        #     print(match['predictions'], match['persona_predictions'], match['persona'])

