import numpy as np
import torch
import torch.nn as nn
import collections


class Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, activation=nn.functional.relu):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.act = activation

    def forward(self, x):
        x = self.act(self.layer1(x))
        x = self.act(self.layer2(x))
        out = self.layer3(x)

        return out


class ReplayMemory():
    def __init__(self, batch_size=10000):
        self.states = []
        self.actions = []
        self.rewards = []
        self.rewards_togo = []
        self.advantages = []
        self.values = []
        self.log_probs = []
        self.batch_size = batch_size

    def push(self, state, action, reward, reward_togo, advantage, value, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.rewards_togo.append(reward_togo)
        self.advantages.append(advantage)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def sample(self):
        num_states = len(self.states)
        batch_start = torch.arange(0, num_states, self.batch_size)
        indices = torch.randperm(num_states)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return (torch.tensor(self.states),
                torch.tensor(self.actions),
                torch.tensor(self.rewards),
                torch.tensor(self.rewards_togo),
                torch.tensor(self.advantages),
                torch.tensor(self.values),
                torch.tensor(self.log_probs),
                batches)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.rewards_togo = []
        self.advantages = []
        self.values = []
        self.log_probs = []
