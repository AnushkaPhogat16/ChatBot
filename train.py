import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np 

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

with open('prompts.json', 'r') as prompt:
    prompts = json.load(prompt)


all_words = []
tags = []
xy = []

# HERE WE ARE TOKENIZING OUR SENTENCE

for a_prompt in prompts['prompts']:
    tag = a_prompt['tag']
    tags.append(tag)
    for pattern in a_prompt['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))


ignore_words = ['?', ",", "!", ":", ".", ")", "("]
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(tags))

print(tags) 

# TRAINING DATA

X_train = []
y_train = []
for (pattern_sent, tag) in xy:
    bag = bag_of_words(pattern_sent, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataSet(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train


    def __getitem__(self, index):
       return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples
    
batch_size  = 8

dataset = ChatDataSet()
train_loader = DataLoader(dataset= dataset, batch_size= batch_size, shuffle=True, num_workers=2)
