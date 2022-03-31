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
from main import EssayData, initalize_gpt2, initalize_glove, PersonalityClassifier, DiscretePersonalityClassifier, PersuasionData, PersonaChatData

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    info_path = '../data/FullData/full_info.csv'
    dialogue_path = '../data/FullData/full_dialog.csv'
    persona_path = '../personachat_truecased/personachat_truecased_full_valid.json'
    essay_path = '../essays/essays.csv'
    bsize = 32
    emb_dim = 768
    h_dim = 128
    n_personas = 1
    use_wandb = True
    discrete = True
    n_discrete = 10
    emb_kind = 'gpt2'
    k = 5
    checkpoint_path = './model.pkl'

    if emb_kind == 'gpt2':
        tokenizer, decoder, embs, n_toks, padding = initalize_gpt2()
    elif emb_kind == 'glove':
        tokenizer, decoder, embs, n_toks, padding = initalize_glove('../glove/glove.6B.50d.txt')
    else:
        raise NotImplementedError
    if not discrete:
        model = PersonalityClassifier(device, emb_dim, h_dim, n_toks, padding, embs).to(device)
    else:
        model = DiscretePersonalityClassifier(device, n_discrete, emb_dim, h_dim, n_toks, padding, decoder, k, embs).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    train_dataset = PersuasionData(info_path, dialogue_path, tokenizer, padding, 'train', 0.75, discrete, n_discrete)
    eval_dataset = PersuasionData(info_path, dialogue_path, tokenizer, padding, 'eval', 0.75, discrete, n_discrete)
    persona_dataset = PersonaChatData(persona_path, tokenizer, padding)
    essay_dataset = EssayData(essay_path, tokenizer, padding)
    train_data_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True, collate_fn=train_dataset.collate)
    eval_data_loader = DataLoader(eval_dataset, batch_size=bsize, shuffle=True, collate_fn=eval_dataset.collate)
    persona_data_loader = DataLoader(persona_dataset, batch_size=bsize, shuffle=True, collate_fn=persona_dataset.collate)
    essay_data_loader = DataLoader(essay_dataset, batch_size=bsize, shuffle=True, collate_fn=essay_dataset.collate)
    evaluators = {
                  'train': lambda: model.eval(train_data_loader), 
                  'eval': lambda: model.eval(eval_data_loader), 
                  'persona': lambda: model.eval_personachat(persona_data_loader), 
                  'essay': lambda: model.eval(essay_data_loader), 
                 }

    for name, evaluator in evaluators.items():
        logs, str_ = evaluator()
        print('evaluator %s:' % (name))
        print(str_)
        print(logs)
        print('\n'*2)
