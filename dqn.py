import random

import gym
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch import optim

from common.greedy_function import get_epsilon_function
from common.replay_buffer import ReplayBuffer, Transition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

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


class DQNSettings:
    replay_buffer_size = 1000
    num_frames = 10000
    batch_size = 32
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 1000


def compute_td_loss(batch_size, gamma, optimizer, replay_buffer):
    transitions = replay_buffer.sample(batch_size=batch_size)

    state = torch.stack([t.state for t in transitions])
    next_state = torch.stack([t.next_state for t in transitions])
    action = torch.tensor([t.action for t in transitions])
    reward = torch.tensor([t.reward for t in transitions])
    done = torch.tensor([t.done for t in transitions])

    q_values = model(state)
    next_q_values = model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1).values
    expected_q_value = reward + gamma * next_q_value * (1 - done.int())

    loss = (q_value - expected_q_value.data).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def evaluate_model(env, model: DQN, num_episodes=10, epsilon=0.05, render=False):
    rewards = []
    for _ in range(num_episodes):
        episode_reward = 0
        state = env.reset()
        while True:
            if render:
                env.render()
            action = model.epsilon_act(torch.tensor(state), epsilon)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                env.reset()
                rewards.append(episode_reward)
                break
    return sum(rewards) / len(rewards)


def main(
    env,
    model,
    settings: DQNSettings,
):
    optimizer = optim.Adam(model.parameters())
    replay_buffer = ReplayBuffer(settings.replay_buffer_size)
    epsilon_by_frame = get_epsilon_function(
        settings.epsilon_start,
        settings.epsilon_end,
        settings.epsilon_decay,
    )
    losses = []
    all_rewards = []
    episode_reward = 0
    evaluated_rewards = []

    state = env.reset()
    for frame_idx in range(1, settings.num_frames + 1):
        epsilon = epsilon_by_frame(frame_idx)
        action = model.epsilon_act(torch.tensor(state), epsilon)

        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(
            Transition(
                torch.tensor(state), action, torch.tensor(next_state), reward, done
            )
        )

        state = next_state
        episode_reward += reward

        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0

        if len(replay_buffer) > settings.batch_size:
            loss = compute_td_loss(
                settings.batch_size, settings.gamma, optimizer, replay_buffer
            )
            losses.append(loss.data)

        if frame_idx % 200 == 0:
            evaluated_rewards.append(evaluate_model(env, model))
            plt.plot(evaluated_rewards)
            plt.show()

    print(evaluate_model(env, model, render=True))


if __name__ == "__main__":
    model = DQN().to(device)
    env_id = "CartPole-v1"
    env = gym.make(env_id)
    settings = DQNSettings()
    main(env=env, model=model, settings=settings)
