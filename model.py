import torch
import torch.nn as nn
import numpy as np
import random
from noise import SharedNoiseTable


embed_size = 10
vocab_size = 10000
hidden_size = 64

noise = SharedNoiseTable()


class RNNLM(nn.Module):
    def __init__(self, init_seed=None, mutate_seeds=None):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

        self.size = sum([i.size().numel() for i in self.state_dict().values()])


        if init_seed:
            self.seeds = [init_seed]
        else:
            self.seeds = [random.randint(0, self.size)]

        theta = noise.get(init_seed, self.size)

        if mutate_seeds:
            for power, idk in mutate_seeds:
                theta = theta + (power * noise.get(idk, self.size))

        state_dict = {}
        idx = 0
        for name, tensor in self.state_dict().items():
            layer_size = tensor.size().numel()
            layer = torch.tensor(theta[idx: idx+layer_size]).resize_(tensor.size())
            #print(layer_size, layer.size())
            state_dict[name] = layer
            idx+=layer_size
        nn.Module.load_state_dict(self, state_dict=state_dict, strict=True)


    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.embed(x)

        # Forward propagate LSTM
        out, (h, c) = self.lstm(x, h)

        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.reshape(out.size(0) * out.size(1), out.size(2))


        # Decode hidden states of all time steps
        out = self.linear(out)
        return out, (h, c)

    def mutate(self, sigma, seed):
        self.seeds.append((sigma, seed))
