import gym
import torch

from agent.dqn import DQN
from network.atari_network import Network

env = gym.make("ALE/Breakout-v5", frameskip=1)
env = gym.wrappers.AtariPreprocessing(env)
env = gym.wrappers.FrameStack(env, 4)

evaluate_env = gym.make("ALE/Breakout-v5", frameskip=1, render_mode="human")
evaluate_env = gym.wrappers.AtariPreprocessing(evaluate_env)
evaluate_env = gym.wrappers.FrameStack(evaluate_env, 4)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Network(action_space=env.action_space).to(device)
agent = DQN(
    env=env,
    model=model,
    replay_buffer_size=1000000,
    num_frames=10000000,
    batch_size=32,
    gamma=0.99,
    epsilon_start=1,
    epsilon_end=0.1,
    epsilon_decay=1000000,
    sync_period=10000,
)
agent.train()
