from torch.utils.tensorboard import SummaryWriter


class BaseAgent:
    def __init__(self):
        self.writer = SummaryWriter(log_dir="./logs")

    def train(self):
        raise NotImplementedError

    def load_model(self):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError
