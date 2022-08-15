import copy

import torch
from torch import optim
from tqdm import tqdm

from agent.base_agent import BaseAgent
from common.replay_buffer import ReplayBuffer, Transition
from network.base_network import BaseNetwork


class DQN(BaseAgent):
    def __init__(
        self,
        env,
        model: BaseNetwork,
        replay_buffer_size,
        num_frames,
        batch_size,
        gamma,
        epsilon_start,
        epsilon_end,
        epsilon_decay,
        sync_period,
        optimizer=optim.Adam,
    ):
        super().__init__()
        # train parameter
        self.env = env
        self.behaviour_network = copy.deepcopy(model)
        self.target_network = copy.deepcopy(model)
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sync_period = sync_period
        self.optimizer = optimizer(self.behaviour_network.parameters())

    def update_model(self):
        transitions = self.replay_buffer.sample(batch_size=self.batch_size)

        state = torch.stack([t.state for t in transitions])
        next_state = torch.stack([t.next_state for t in transitions])
        action = torch.tensor([t.action for t in transitions])
        reward = torch.tensor([t.reward for t in transitions])
        done = torch.tensor([t.done for t in transitions])

        q_values = self.target_network(state)
        next_q_values = self.target_network(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1).values
        expected_q_value = reward + self.gamma * next_q_value * (1 - done.int())

        loss = (q_value - expected_q_value.data).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def sync_model(self):
        self.target_network.load_state_dict(self.behaviour_network.state_dict())

    def train(self):
        state = self.env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32)
        episode_reward = 0
        episode_index = 0
        for frame_idx in tqdm(range(1, self.num_frames + 1)):

            epsilon = self.get_epsilon(frame_idx)
            action = self.behaviour_network.epsilon_act(
                torch.unsqueeze(state_tensor, dim=0),
                epsilon=epsilon,
            )
            next_state, reward, done, _ = self.env.step(action)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
            self.replay_buffer.push(
                Transition(
                    state_tensor,
                    action,
                    next_state_tensor,
                    reward,
                    done,
                )
            )
            state_tensor = next_state_tensor
            episode_reward += reward

            if done:
                episode_index += 1
                self.writer.add_scalar("episode_reward", episode_reward, episode_index)
                episode_reward = 0
                state = self.env.reset()
                state_tensor = torch.tensor(state, dtype=torch.float32)

            if len(self.replay_buffer) > self.batch_size:
                loss = self.update_model()
                self.writer.add_scalar("loss", loss.item(), frame_idx)

            if frame_idx % self.sync_period == 0:
                self.sync_model()

    def evaluate(self, env, num_episodes=1, epsilon=0.05):
        rewards = []
        for _ in range(num_episodes):
            episode_reward = 0
            state = env.reset()
            while True:
                action = self.target_network.epsilon_act(
                    torch.unsqueeze(torch.tensor(state, dtype=torch.float32), dim=0),
                    epsilon,
                )
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                if done:
                    env.reset()
                    rewards.append(episode_reward)
                    break
        return sum(rewards) / len(rewards)

    def get_epsilon(self, frame_idx):
        return (
            self.epsilon_start
            - frame_idx * (self.epsilon_start - self.epsilon_end) / self.epsilon_decay
        )

    def load_model(self):
        pass

    def save_model(self):
        pass
