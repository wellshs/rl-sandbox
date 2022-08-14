from random import random

from torch import nn


class BaseNetwork(nn.Module):
    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space

    def act(self, state):
        raise NotImplementedError()

    def epsilon_act(self, state, epsilon):
        return self.act(state) if random() > epsilon else self.action_space.sample()
