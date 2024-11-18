import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import Dataset

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1 + nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, xs):
        x, a = xs
        out = self.fc1(x)
        out = self.relu(out)
        # debug()
        out = self.fc2(torch.cat([out, a], 1))  # for Q(s, a) computation
        out = self.relu(out)
        out = self.fc3(out)
        return out



### CNN-based autoencoder for pretraining
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1), # Preserve T, F -> 16 channels
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0), # downsample T -> T/2

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), # Preserve T/2, F -> 32 channels
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0), # downsample T/2 -> T/4

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # Preserve T/4, F -> 64 channels
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0), # downsample T/4 -> T/8
        )

    def forward(self, x):
        # x is assumed to be NxTxF
        x = x.permute(0, 2, 1) # NxTxF -> NxFxT for Conv1d
        x = self.encoder(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1), # T/8 -> T/4
            nn.ReLU(),

            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1), # T/4 -> T/2
            nn.ReLU(),

            nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1), # T/2 -> T
        )

    def forward(self, x):
        x = self.decoder(x)
        x = x.permute(0, 2, 1) # NxFxT -> NxTxF
        return x

class CnnAutoencoder(nn.Module):
    def __init__(self):
        super(CnnAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)