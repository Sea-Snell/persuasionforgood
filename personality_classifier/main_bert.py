import csv
from collections import defaultdict
from transformers import RobertaModel, RobertaTokenizer
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
import json
import random
from torch.distributions.normal import Normal
import wandb
import numpy as np
import spacy

def listtobig5(x):
    return {'extrovert': x[0], 'agreeable': x[1], 'conscentious': x[2], 'neurotic': x[3], 'open': x[4]}

def big5tolist(x):
    return [x['extrovert'], x['agreeable'], x['conscentious'], x['neurotic'], x['open']]

def initalize_roberta():
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    roberta = RobertaModel.from_pretrained('roberta-base')
    return lambda x: tokenizer(x)['input_ids'], roberta, len(tokenizer), tokenizer.pad_token_id

class PersonalityClassifier(nn.Module):
    def __init__(self, emb_dim, h_dim, pad_token, pretrained_model):
        super().__init__()
        self.emb_dim = emb_dim
        self.h_dim = h_dim
        self.pad_token = pad_token
        self.pretrained_model = pretrained_model
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
        embs = []
        with torch.no_grad():
            for item in tokens:
                mask = ((item != self.pad_token).float().sum(dim=1) > 0).float()
                embs.append((self.pretrained_model(item, attention_mask=(item != self.pad_token).long()).pooler_output * mask.unsqueeze(1)).sum(dim=0) / mask.sum())
            embs = torch.stack(embs, dim=0)
        return Normal(self.mean_mlp(embs), torch.exp(self.log_uncertainty_mlp(embs)))

class PersuasionData(Dataset):
    def __init__(self, info_path, dialogue_path, tokenizer, pad_token, split, frac_train) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_token = pad_token

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
            text = [item['text'] for item in data_item['dialogue'] if item['role'] == 0]
            tokens = [self.tokenizer(item) for item in text]
            examples.append((text, tokens, classes,))
        if 'big5_1' in data_item:
            classes = big5tolist(data_item['big5_1'])
            text = [item['text'] for item in data_item['dialogue'] if item['role'] == 1]
            tokens = [self.tokenizer(item) for item in text]
            examples.append((text, tokens, classes,))
        return examples
    
    def __getitem__(self, i):
        return self.datapoints[i]
    
    def __len__(self):
        return len(self.datapoints)
    
    def collate(self, items):
        _, tokens, classes = list(zip(*items))
        tokens = [nn.utils.rnn.pad_sequence(list(map(lambda x: torch.tensor(x), tok)), 
                                            batch_first=False, 
                                            padding_value=self.pad_token) for tok in tokens]
        max_bsize = max(map(lambda x: x.shape[1], tokens))
        for i in range(len(tokens)):
            tokens[i] = torch.cat((tokens[i], torch.full((tokens[i].shape[0], max_bsize-tokens[i].shape[1],), self.pad_token, dtype=torch.long)), dim=1)
        tokens = nn.utils.rnn.pad_sequence(tokens, 
                                           batch_first=True, 
                                           padding_value=self.pad_token)
        tokens = tokens.permute(0, 2, 1)
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
        text = [utterance['candidates'][-1] for utterance in item['utterances']]
        tokens = [self.tokenizer(item) for item in text]
        persona_tokens = self.tokenizer('\n'.join(item['personality']))
        return persona, persona_tokens, text, tokens
    
    def __getitem__(self, i):
        return self.datapoints[i]
    
    def __len__(self):
        return len(self.datapoints)
    
    def collate(self, items):
        _, tokens, classes = list(zip(*items))
        tokens = [nn.utils.rnn.pad_sequence(list(map(lambda x: torch.tensor(x), tok)), 
                                            batch_first=False, 
                                            padding_value=self.pad_token) for tok in tokens]
        max_bsize = max(map(lambda x: x.shape[1], tokens))
        for i in range(len(tokens)):
            tokens[i] = torch.cat((tokens[i], torch.full((tokens[i].shape[0], max_bsize-tokens[i].shape[1],), self.pad_token, dtype=torch.long)), dim=1)
        tokens = nn.utils.rnn.pad_sequence(tokens, 
                                           batch_first=True, 
                                           padding_value=self.pad_token)
        tokens = tokens.permute(0, 2, 1)
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
    use_wandb = True

    if use_wandb:
        wandb.init(project='personality_classifier')

    tokenizer, pretrained_model, n_toks, padding = initalize_roberta()
    model = PersonalityClassifier(emb_dim, h_dim, padding, pretrained_model).to(device)
    train_dataset = PersuasionData(info_path, dialogue_path, tokenizer, padding, 'train', 0.75)
    eval_dataset = PersuasionData(info_path, dialogue_path, tokenizer, padding, 'eval', 0.75)
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
            predictions = model(tokens)
            loss = -predictions.log_prob(classes).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
        
        train_loss = 0.0
        train_entropy = 0.0
        train_entropy_std = 0.0
        train_mean = np.zeros((5,))
        train_mean_std = np.zeros((5,))
        train_std = np.zeros((5,))
        train_std_std = np.zeros((5,))
        train_total = 0
        eval_loss = 0.0
        eval_entropy = 0.0
        eval_entropy_std = 0.0
        eval_mean = 0.0
        eval_mean_std = 0.0
        eval_std = 0.0
        eval_std_std = 0.0
        eval_total = 0
        for tokens, classes in tqdm(train_data_loader):
            tokens, classes = tokens.to(device), classes.to(device)
            predictions = model(tokens)
            loss = -predictions.log_prob(classes).mean()
            train_loss += loss.item() * tokens.shape[0]
            train_entropy += predictions.entropy().mean().item() * tokens.shape[0]
            train_entropy_std += predictions.entropy().std().item() * tokens.shape[0]
            train_mean += predictions.loc.mean(dim=0).detach().cpu().numpy() * tokens.shape[0]
            train_mean_std += predictions.loc.std(dim=0).detach().cpu().numpy() * tokens.shape[0]
            train_std += predictions.scale.mean(dim=0).detach().cpu().numpy() * tokens.shape[0]
            train_std_std += predictions.scale.std(dim=0).detach().cpu().numpy() * tokens.shape[0]
            train_total += tokens.shape[0]
        for tokens, classes in tqdm(eval_data_loader):
            tokens, classes = tokens.to(device), classes.to(device)
            predictions = model(tokens)
            loss = -predictions.log_prob(classes).mean()
            eval_loss += loss.item() * tokens.shape[0]
            eval_entropy += predictions.entropy().mean().item() * tokens.shape[0]
            eval_entropy_std += predictions.entropy().std().item() * tokens.shape[0]
            eval_mean += predictions.loc.mean(dim=0).detach().cpu().numpy() * tokens.shape[0]
            eval_mean_std += predictions.loc.std(dim=0).detach().cpu().numpy() * tokens.shape[0]
            eval_std += predictions.scale.mean(dim=0).detach().cpu().numpy() * tokens.shape[0]
            eval_std_std += predictions.scale.std(dim=0).detach().cpu().numpy() * tokens.shape[0]
            eval_total += tokens.shape[0]
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
        results = {'train': {'loss': train_loss / train_total, 
                             'entropy': train_entropy / train_total, 
                             'entropy_std': train_entropy_std / train_total, 
                             'mean': listtobig5((train_mean / train_total).tolist()), 
                             'mean_std': listtobig5((train_mean_std / train_total).tolist()),
                             'std': listtobig5((train_std / train_total).tolist()), 
                             'std_std': listtobig5((train_std_std / train_total).tolist())}, 
                    'eval': {'loss': eval_loss / eval_total, 
                             'entropy': eval_entropy / eval_total, 
                             'entropy_std': eval_entropy_std / eval_total, 
                             'mean': listtobig5((eval_mean / eval_total).tolist()), 
                             'mean_std': listtobig5((eval_mean_std / eval_total).tolist()), 
                             'std': listtobig5((eval_std / eval_total).tolist()), 
                             'std_std': listtobig5((eval_std_std / eval_total).tolist())}, 
                  }
        if use_wandb:
            wandb.log({'epoch': epoch, **results})
        print('train:', results['train'])
        print('eval:', results['eval'])
        # for match in persona_matches:
        #     print(match['predictions'], match['persona_predictions'], match['persona'])



