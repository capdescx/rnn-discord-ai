import discord
from discord.ext import commands
import json
import torch
import torch.nn as nn
import numpy as np

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

intents = discord.Intents.default()
bot = commands.Bot(command_prefix='!', intents=intents)
intents.message_content = True

with open('cleaned_vocab.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)


vocab_size = 50255
embedding_dim = 128
hidden_dim = 256
rnn_model = RNNModel(vocab_size, embedding_dim, hidden_dim)

rnn_model.load_state_dict(torch.load('rnn_model.pth', map_location=torch.device('cpu')))
rnn_model.eval()  


beam_width = 20
max_beam_length = 100


@bot.event
async def on_ready():
    print(f'✅✅✅✅✅ Logged in as {bot.user.name} ✅✅✅✅✅')


@bot.event
async def on_message(message):
    await bot.change_presence(activity=discord.Game(name="Axe 2.0"))
    if message.author == bot.user:
        return

    content = message.content


    input_tokens = content.split()  
    input_sequence = torch.tensor([vocab.get(token, vocab['<unk>']) for token in input_tokens]).unsqueeze(0)
    output = rnn_model(input_sequence)

    
    beam = [{'sequence': [vocab['<start>']], 'prob': 1.0}]
    for _ in range(max_beam_length):
        new_beam = []
        for entry in beam:
            prev_word_idx = entry['sequence'][-1]
            input_sequence = torch.tensor(entry['sequence']).unsqueeze(0)
            output = rnn_model(input_sequence)
            next_word_probs = torch.softmax(output[0, -1, :], dim=-1).detach().numpy()
            top_indices = np.argsort(next_word_probs)[-beam_width:]
            for idx in top_indices:
                new_entry = {
                    'sequence': entry['sequence'] + [idx],
                    'prob': entry['prob'] * next_word_probs[idx]
                }
                new_beam.append(new_entry)
        new_beam.sort(key=lambda x: -x['prob'])
        beam = new_beam[:beam_width]
    
    
    probs = np.array([entry['prob'] for entry in beam])
    normalized_probs = probs / np.sum(probs)  
    sampled_index = np.random.choice(len(beam), p=normalized_probs)
    sampled_sequence = beam[sampled_index]['sequence']

    response_words = [next((word for word, idx in vocab.items() if idx == token_idx), 'shayen = gay') for token_idx in sampled_sequence]
    response = ' '.join(response_words[1:])  

    await message.channel.send(response)  


bot.run('PASSWORD')
