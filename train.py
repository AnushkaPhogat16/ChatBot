import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

class ChatDataSet(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Use this block to ensure the script is run directly
if __name__ == "__main__":  # Added to guard the entry point
    with open('prompts.json', 'r') as prompt:
        prompts = json.load(prompt)

    all_words = []
    tags = []
    xy = []

    # Tokenize the sentence
    for a_prompt in prompts['prompts']:
        tag = a_prompt['tag']
        tags.append(tag)
        for pattern in a_prompt['patterns']:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))

    ignore_words = ['?', ",", "!", ":", ".", ")", "("]
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    # Training data
    X_train = []
    y_train = []
    for (pattern_sent, tag) in xy:
        bag = bag_of_words(pattern_sent, all_words)
        X_train.append(bag)

        label = tags.index(tag)
        y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Hyperparameters
    batch_size = 8
    hidden_size = 8
    output_size = len(tags)
    input_size = len(X_train[0])
    learning_rate = 0.001
    num_epochs = 1000

    dataset = ChatDataSet()
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # Set num_workers to 0

    print(input_size, len(all_words))
    print(output_size, tags)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss = None  # Initialize loss variable

    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device).long()

            # Forward
            outputs = model(words)
            loss = criterion(outputs, labels)

            # Backward and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'epoch {epoch + 1}/{num_epochs}, loss={loss.item():.4f}')

    if loss is not None:
        print(f'final loss, loss={loss.item():.4f}')


#TO SAVE THE DATA

data = {
    "model_state": model.state_dict(),
    "input_size":  input_size,
    "output_size": output_size ,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')