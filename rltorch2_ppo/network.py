from functools import partial
import torch
from torch import nn


def init_fn(m, gain=1):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, gain=gain)
        nn.init.constant_(m.bias.data, 0)


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class ResidualBlock(nn.Module):

    def __init__(self, num_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_channels, num_channels, 3, 1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_channels, num_channels, 3, 1, padding=1),
        )

    def forward(self, x):
        return self.net(x).add_(x)


class ConvSequence(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
            nn.MaxPool2d(3, 2, padding=1),
            ResidualBlock(out_channels),
            ResidualBlock(out_channels)
        )

    def forward(self, x):
        return self.net(x)


class ImpalaCNNBody(nn.Module):

    def __init__(self, num_channels, num_initial_blocks=1, depths=(16, 32)):
        super().__init__()
        assert 1 <= num_initial_blocks <= len(depths)

        self._hidden_size = 256

        self.feature_dim = depths[num_initial_blocks-1] * \
            (64 // 2 ** num_initial_blocks) ** 2

        in_channels = num_channels
        nets = []
        for out_channels in depths:
            nets.append(ConvSequence(in_channels, out_channels))
            in_channels = out_channels

        current_dim = depths[-1] * (64 // 2 ** len(depths)) ** 2
        nets.append(
            nn.Sequential(
                nn.LeakyReLU(0.2),
                Flatten(),
                nn.Linear(current_dim, self._hidden_size),
                nn.LeakyReLU(0.2, inplace=True),
            )
        )

        self.initial_net = nn.Sequential(
            *[nets.pop(0) for _ in range(num_initial_blocks)]
        )
        self.net = nn.Sequential(*nets)

        # self.critic_linear = nn.Linear(self._hidden_size, 1).apply(init_fn)
        Nb = 50
        self.critic_linear = nn.Linear(self._hidden_size, Nb).apply(init_fn)
        self.cluster_linear = nn.Linear(Nb, Nb).apply(init_fn)
        self.softmax = nn.Softmax(dim=1)
        self.mu_linear = nn.Linear(Nb, Nb).apply(init_fn)

    @property
    def output_size(self):
        return self._hidden_size

    def forward(self, states):
        features = self.initial_net(states / 255)
        return self.net(features)

    def forward(self, states):
        x = self.initial_net(states / 255)
        x = self.net(x)

        y = self.critic_linear(x)
        a = self.cluster_linear(y)
        a = self.softmax(a)
        mu = self.mu_linear(y)

        value = (a * mu).sum(axis=1).unsqueeze(1)

        return value, x