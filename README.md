# rnn-discord-ai
This project implements a text-sequence learning system using PyTorch, integrated with a Discord bot for interactive predictions. It processes text data, trains an RNN model, and serves responses through Discord.

File Overview

main.py – Entry point; initializes a Discord bot, loads the trained RNN model, and handles message-based inference.
rnn_model.py – Defines the RNN architecture with embedding, recurrent, and linear layers for sequence modeling.
training_model.py – Handles dataset loading, training, padding, and model optimization using PyTorch. Saves the trained model as rnn_model.pth (Omitted as it is not a readable file).
create_mapper.py – Builds token and word mappings from text data for preprocessing and decoding.

Data and Artifacts

vocab.json, tokenizer.json, word_to_token_mapping.json: Store word-token mappings.

dataset.csv, train_dataset.csv, validation_dataset.csv: Contain training and validation data. (These have been omitted due to large file size)

rnn_model.pth: Trained RNN weights. (Omitted as it is not a readable file)

Workflow

Run create_mapper.py to generate mappings.
Train the model with training_model.py.
Deploy and interact with the model via Discord using main.py.
