import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import json


with open('cleaned_vocab.json', 'r', encoding='utf-8') as vocab_file:
    vocab = json.load(vocab_file)

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        return self.fc(output)

class CustomDataset(Dataset):
    def __init__(self, csv_path, max_seq_length, vocab_size):
        self.data = pd.read_csv(csv_path)
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_text = self.data.loc[idx, 'Input']
        output_text = self.data.loc[idx, 'Output']
        
        input_tokens = [vocab.get(token, vocab['<unk>']) for token in input_text.split()]
        output_tokens = [vocab.get(token, vocab['<unk>']) for token in output_text.split()]
        
        if len(input_tokens) > self.max_seq_length:
            input_tokens = input_tokens[:self.max_seq_length]
        if len(output_tokens) > self.max_seq_length:
            output_tokens = output_tokens[:self.max_seq_length]
        
        padded_input_sequence = input_tokens + [0] * (self.max_seq_length - len(input_tokens))
        padded_output_sequence = output_tokens + [0] * (self.max_seq_length - len(output_tokens))
        
        input_sequence = torch.tensor(padded_input_sequence[:-1])
        target_sequence = torch.tensor(padded_output_sequence[1:])
        return input_sequence, target_sequence


class CustomCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    
    def __call__(self, batch):
        input_seqs, target_seqs = zip(*batch)
        padded_input_seqs = pad_sequence(input_seqs, batch_first=True, padding_value=self.pad_idx)
        padded_target_seqs = pad_sequence(target_seqs, batch_first=True, padding_value=self.pad_idx)
        return padded_input_seqs, padded_target_seqs

max_sequence_length = 30
train_dataset = CustomDataset('train_dataset.csv', max_sequence_length, len(vocab))
validation_dataset = CustomDataset('validation_dataset.csv', max_sequence_length, len(vocab))

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=CustomCollate(0))
validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False, collate_fn=CustomCollate(0))

vocab_size = 50255
embedding_dim = 128
hidden_dim = 256
rnn_model = RNNModel(vocab_size, embedding_dim, hidden_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)

num_epochs = 100

best_validation_loss = float('inf')

for epoch in range(num_epochs):
    for input_batch, target_batch in train_dataloader:
        optimizer.zero_grad()
        output = rnn_model(input_batch)
        loss = criterion(output.transpose(1, 2), target_batch)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        validation_loss = 0
        for input_batch, target_batch in validation_dataloader:
            output = rnn_model(input_batch)
            loss = criterion(output.transpose(1, 2), target_batch)
            validation_loss += loss.item()
        validation_loss /= len(validation_dataloader)

    print(f'Epoch [{epoch+1}/{num_epochs}] - Validation Loss: {validation_loss:.4f}')

    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss

print("Training completed.")
