from torch.utils.tensorboard import SummaryWriter


class BaseAgent:
    def __init__(self):
        self.writer = SummaryWriter(log_dir=f"./logs/{self.__class__.__name__}")

    def train(self):
        raise NotImplementedError

    def load_model(self):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError
