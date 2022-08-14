import torch
from torch import nn

from network.base_network import BaseNetwork


class Network(BaseNetwork):
    def __init__(self, action_space):
        super(Network, self).__init__(action_space)
        self.action_space = action_space
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, action_space.n),
        )

    def forward(self, x):
        conv_out = self.conv_layers(x)
        return self.linear_layers(conv_out.view(conv_out.size(0), -1))

    def act(self, state):
        q_value = self.forward(state)
        return torch.argmax(q_value).item()
