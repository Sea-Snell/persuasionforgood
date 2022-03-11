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
        self.mlp = nn.Sequential(
            # nn.Linear(self.emb_dim, self.h_dim), 
            # nn.ReLU(), 
            nn.Linear(self.h_dim, 5), 
        )

    def forward(self, tokens):
        mask = tokens != self.pad_token
        avg_emb = (self.emb(tokens) * mask.unsqueeze(2)).sum(dim=1) / mask.sum(dim=1).unsqueeze(1)
        return self.mlp(avg_emb)


class PersuasionData(Dataset):
    def __init__(self, info_path, dialogue_path, tokenizer, split, frac_train) -> None:
        super().__init__()
        self.tokenizer = tokenizer

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
            classes = [data_item['big5_0']['extrovert'], 
                    data_item['big5_0']['agreeable'], 
                    data_item['big5_0']['conscentious'], 
                    data_item['big5_0']['neurotic'], 
                    data_item['big5_0']['open']]
            text = '\n'.join([item['text'] for item in data_item['dialogue'] if item['role'] == 0])
            tokens = self.tokenizer(text)['input_ids']
            examples.append((text, tokens, classes,))
        if 'big5_1' in data_item:
            classes = [data_item['big5_1']['extrovert'], 
                    data_item['big5_1']['agreeable'], 
                    data_item['big5_1']['conscentious'], 
                    data_item['big5_1']['neurotic'], 
                    data_item['big5_1']['open']]
            text = '\n'.join([item['text'] for item in data_item['dialogue'] if item['role'] == 1])
            tokens = self.tokenizer(text)['input_ids']
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
                                           padding_value=self.tokenizer.pad_token_id)
        classes = torch.tensor(classes)
        return tokens, classes

class PersonaChatData(Dataset):
    def __init__(self, path, tokenizer):
        self.tokenizer = tokenizer
        with open(path, 'r') as f:
            data = json.load(f)
        self.datapoints = [self.prepare_item(item) for item in data]
    
    def prepare_item(self, item):
        persona = item['personality']
        text = '\n'.join([utterance['candidates'][-1] for utterance in item['utterances']])
        tokens = self.tokenizer(text)['input_ids']
        return persona, text, tokens
    
    def __getitem__(self, i):
        return self.datapoints[i]
    
    def __len__(self):
        return len(self.datapoints)
    
    def collate(self, items):
        _, tokens, classes = list(zip(*items))
        tokens = nn.utils.rnn.pad_sequence(list(map(lambda x: torch.tensor(x), tokens)), 
                                           batch_first=True, 
                                           padding_value=self.tokenizer.pad_token_id)
        classes = torch.tensor(classes)
        return tokens, classes

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    info_path = '../data/FullData/full_info.csv'
    dialogue_path = '../data/FullData/full_dialog.csv'
    persona_path = '../personachat_truecased/personachat_truecased_full_train.json'
    bsize = 256
    emb_dim = 768
    h_dim = 256
    lr = 1e-4
    weight_decay = 0.001
    epochs = 1000
    n_personas = 1

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    gpt2_embs = GPT2Model.from_pretrained('gpt2').resize_token_embeddings(len(tokenizer))
    for param in gpt2_embs.parameters():
        param.requires_grad = False
    model = PersonalityClassifier(emb_dim, h_dim, len(tokenizer), tokenizer.pad_token_id, gpt2_embs).to(device)
    train_dataset = PersuasionData(info_path, dialogue_path, tokenizer, 'train', 0.75)
    eval_dataset = PersuasionData(info_path, dialogue_path, tokenizer, 'eval', 0.75)
    persona_dataset = PersonaChatData(persona_path, tokenizer)
    train_data_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True, collate_fn=train_dataset.collate)
    eval_data_loader = DataLoader(eval_dataset, batch_size=bsize, shuffle=True, collate_fn=eval_dataset.collate)
    optim = AdamW(model.parameters(), lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        for tokens, classes in tqdm(train_data_loader):
            tokens, classes = tokens.to(device), classes.to(device)
            predictions = model(tokens)
            loss = ((classes - predictions) ** 2).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
        
        train_loss = 0.0
        train_total = 0
        eval_loss = 0.0
        eval_total = 0
        for tokens, classes in tqdm(train_data_loader):
            tokens, classes = tokens.to(device), classes.to(device)
            predictions = model(tokens)
            loss = ((classes - predictions) ** 2).mean()
            train_loss += loss.item() * tokens.shape[0]
            train_total += tokens.shape[0]
        for tokens, classes in tqdm(eval_data_loader):
            tokens, classes = tokens.to(device), classes.to(device)
            predictions = model(tokens)
            loss = ((classes - predictions) ** 2).mean()
            eval_loss += loss.item() * tokens.shape[0]
            eval_total += tokens.shape[0]
        persona_matches = []
        for idx in random.sample(range(len(persona_dataset)), n_personas):
            persona, text, tokens = persona_dataset[idx]
            tokens = torch.tensor(tokens).to(device)
            predictions = model(tokens.unsqueeze(0))[0].detach().cpu().tolist()
            persona_matches.append({'predictions': {'extrovert': predictions[0], 
                                                    'agreeable': predictions[1], 
                                                    'conscentious': predictions[2], 
                                                    'neurotic': predictions[3], 
                                                    'open': predictions[4]}, 
                                    'persona': persona, 
                                    'utterances': text})
        print('epoch:', epoch)
        print('train loss:', train_loss / train_total)
        print('eval loss:', eval_loss / eval_total)
        for match in persona_matches:
            print(match['predictions'], match['persona'])



