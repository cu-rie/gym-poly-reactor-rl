import torch
from collections import deque
import torch.nn as nn
import numpy as np
import random


def update_target(target_params, source_params, tau=0.1):
    for t, s in zip(target_params, source_params):
        pass


class OUNoise:

    """
    Taken from https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
    """

    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=1000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space
        self.reset()

        self.epsilon = 1.0
        self.epsilon_decay = 0.000005
        self.epsilon_min = 0.0

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):

        ou_state = self.evolve_state() * self.epsilon

        self.epsilon -= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, -1.0, 1.0)


class ReplayMemory:
    def __init__(self, length):
        self.memory = deque(maxlen=length)

    def __len__(self):
        return len(self.memory)

    def save_memory(self, transition):
        s, a, r, s_, t = transition

        s = s.numpy()
        # s_ = s_.numpy()

        transition = [s, a, r, s_, t]
        self.memory.append(transition)

    def sample(self, size):
        sample = random.sample(self.memory, size)

        state = [i[0] for i in sample]
        action = [i[1] for i in sample]
        reward = [i[2] for i in sample]
        next_state = [i[3] for i in sample]
        terminal = [i[4] for i in sample]

        state = np.stack(state)
        state = torch.Tensor(state).squeeze()

        next_state = np.stack(next_state)
        next_state = torch.Tensor(next_state).squeeze()

        reward = np.array(reward)
        reward = torch.tensor(reward, dtype=torch.float32).reshape(-1, 1)

        terminal = np.array(terminal).astype(int)
        terminal = torch.tensor(terminal).reshape(-1, 1)

        action = np.array(action)
        # action = torch.tensor(action, dtype=torch.float32).reshape(-1, 1)
        action = torch.tensor(action).squeeze().float()

        return state, action, reward, next_state, terminal


class DDPGAgent(nn.Module):
    def __init__(self, state_dim, action_dim, action_min: list, action_max: list, gamma=0.99):
        super(DDPGAgent, self).__init__()
        self.action_min = np.array(action_min)
        self.action_max = np.array(action_max)
        self.gamma = gamma
        self.ou_noise = OUNoise(3)

        self.critic = Critic(state_dim, action_dim)
        self.actor = Actor(state_dim, action_dim)

        self.critic_target = Critic(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)

        self.update_model(self.critic, self.critic_target, tau=1.0)
        self.update_model(self.actor, self.actor_target, tau=1.0)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.memory = ReplayMemory(50000)
        self.batch_size = 50
        self.tau = 0.005

        self.loss_ftn = nn.MSELoss()

    def get_action(self, state, t=0):
        action_before_norm = self.actor(state).detach().numpy()
        action_before_norm = self.ou_noise.get_action(action_before_norm, t)

        action_after_norm = (action_before_norm + 1) / 2 * (self.action_max - self.action_min) + self.action_min

        return action_before_norm, action_after_norm, self.ou_noise.epsilon

    def save_transition(self, transition):
        self.memory.save_memory(transition)

    def train_start(self):
        return len(self.memory) > self.batch_size

    def update_model(self, source, target, tau):
        for src_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * src_param.data + (1.0 - tau) * target_param.data)

    def fit(self):
        state, action, reward, next_state, terminal = self.memory.sample(self.batch_size)

        next_q_val = self.critic_target(next_state, self.actor_target(next_state))
        target_q = reward + self.gamma * (1 - terminal) * next_q_val
        q = self.critic(state, action)

        # Critic loss
        value_loss = self.loss_ftn(q, target_q)

        # Critic update
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        policy_loss = -self.critic(state, self.actor(state))
        policy_loss = policy_loss.mean()

        # Actor update
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.update_model(self.critic, self.critic_target, tau=self.tau)
        self.update_model(self.actor, self.actor_target, tau=self.tau)

        return value_loss.item(), policy_loss.item()


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=[64, 64], output_dim=1):
        super(Critic, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        input_dims = [state_dim + action_dim] + hidden_dim
        output_dims = hidden_dim + [output_dim]

        for in_dim, out_dim in zip(input_dims, output_dims):
            self.layers.append(nn.Linear(in_dim, out_dim))

        for i in range(len(hidden_dim)):
            self.activations.append(nn.LeakyReLU())

        self.activations.append(nn.Identity())

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        for l, act in zip(self.layers, self.activations):
            x = l(x)
            x = act(x)

        return x


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=[64, 64]):
        super(Actor, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        input_dims = [state_dim] + hidden_dim
        output_dims = hidden_dim + [action_dim]

        for in_dim, out_dim in zip(input_dims, output_dims):
            self.layers.append(nn.Linear(in_dim, out_dim))

        for i in range(len(hidden_dim)):
            self.activations.append(nn.LeakyReLU())

        self.activations.append(nn.Tanh())

    def forward(self, state):
        x = state
        for l, activation in zip(self.layers, self.activations):
            x = l(x)
            x = activation(x)
        return x
