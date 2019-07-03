import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_
from data_utils import Dictionary, Corpus
from model import RNNLM
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
embed_size = 100
hidden_size = 64
num_layers = 1
num_epochs = 50
batch_size = 1
seq_length = 35
learning_rate = 0.003
input_dropout = 0.7
weight_decay = -4
# Load "Penn Treebank" dataset
corpus = Corpus()
ids = corpus.get_data('data/train.txt', batch_size)
vocab_size = len(corpus.dictionary)
num_batches = ids.size(1) // seq_length


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
              torch.zeros(num_layers, batch_size, hidden_size).to(device))

    with torch.no_grad():
        # Get mini-batch inputs and targets
        inputs = data_source[:, 0: -2].to(device)
        targets = data_source[:, 1: -1].to(device)

        # Forward pass
        states = detach(states)
        outputs, states = model(inputs, states)
        loss = criterion(outputs, targets.reshape(-1))
    return np.exp(loss.item())


# Truncated backpropagation
def detach(states):
    return [state.detach() for state in states]


def train():
    # Train the model
    for epoch in range(num_epochs):
        # Set initial hidden and cell states
        states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
                  torch.zeros(num_layers, batch_size, hidden_size).to(device))

        for i in range(0, ids.size(1) - seq_length, seq_length):
            # Get mini-batch inputs and targets
            inputs = ids[:, i:i + seq_length].to(device)
            targets = ids[:, (i + 1):(i + 1) + seq_length].to(device)

            # Forward pass
            states = detach(states)
            outputs, states = model(inputs, states)
            loss = criterion(outputs, targets.reshape(-1))

            # Backward and optimize
            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            step = (i + 1) // seq_length
            if step % 200 == 0:
                print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                      .format(epoch + 1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item())))


model = RNNLM(1987987).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Save the model checkpoints
# torch.save(model.state_dict(), 'model.ckpt')

test = corpus.get_data('data/ptb.train.txt', batch_size)
score = evaluate(test)
print(score)





