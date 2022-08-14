import random

import torch
from torch import nn


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state):
        q_value = self.forward(state)
        return torch.argmax(q_value).item()

    def epsilon_act(self, state, epsilon):
        return (
            self.act(state) if random.random() > epsilon else env.action_space.sample()
        )
