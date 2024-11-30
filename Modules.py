import collections
import torch.nn as nn
import torch
import numpy as np


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
        self.rewards_togo.append(reward_togo.detach(
        ) if torch.is_tensor(reward_togo) else reward_togo)
        self.advantages.append(
            advantage.detach() if torch.is_tensor(advantage) else advantage)
        self.values.append(value.detach() if torch.is_tensor(value) else value)
        self.log_probs.append(
            log_prob.detach() if torch.is_tensor(log_prob) else log_prob)

    def sample(self):
        # Convert lists to NumPy arrays for efficient processing
        # Shape: [num_samples, state_dim]
        states_np = np.array(self.states, dtype=np.float32)
        # Shape: [num_samples, action_dim]
        actions_np = np.array(self.actions, dtype=np.float32)
        # Shape: [num_samples]
        rewards_np = np.array(self.rewards, dtype=np.float32)

        # Handle tensors that might require gradients
        rewards_togo_np = np.array([r.detach().cpu().numpy() if torch.is_tensor(r) else r
                                    for r in self.rewards_togo], dtype=np.float32)
        advantages_np = np.array([a.detach().cpu().numpy() if torch.is_tensor(a) else a
                                  for a in self.advantages], dtype=np.float32)
        values_np = np.array([v.detach().cpu().numpy() if torch.is_tensor(v) else v
                              for v in self.values], dtype=np.float32)
        log_probs_np = np.array([l.detach().cpu().numpy() if torch.is_tensor(l) else l
                                for l in self.log_probs], dtype=np.float32)

        # Convert NumPy arrays to PyTorch tensors
        # [num_samples, state_dim]
        states_tensor = torch.from_numpy(states_np)
        # [num_samples, action_dim]
        actions_tensor = torch.from_numpy(actions_np)
        rewards_tensor = torch.from_numpy(
            rewards_np)              # [num_samples]
        rewards_togo_tensor = torch.from_numpy(
            rewards_togo_np)    # [num_samples]
        advantages_tensor = torch.from_numpy(
            advantages_np)        # [num_samples]
        values_tensor = torch.from_numpy(
            values_np)                # [num_samples]
        log_probs_tensor = torch.from_numpy(
            log_probs_np)         # [num_samples]

        # Shuffle and create batches
        num_samples = len(self.states)
        indices = torch.randperm(num_samples)
        batches = [indices[i:i+self.batch_size]
                   for i in range(0, num_samples, self.batch_size)]

        return (states_tensor,
                actions_tensor,
                rewards_tensor,
                rewards_togo_tensor,
                advantages_tensor,
                values_tensor,
                log_probs_tensor,
                batches)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.rewards_togo = []
        self.advantages = []
        self.values = []
        self.log_probs = []

    def push_batch(self, states, actions, rewards, rtgs, advantages, values, log_probs):
        # Convert tensors to numpy/CPU if they're on GPU
        states = states.detach().cpu().numpy() if torch.is_tensor(states) else states
        actions = actions.detach().cpu().numpy() if torch.is_tensor(actions) else actions
        rewards = rewards.detach().cpu().numpy() if torch.is_tensor(rewards) else rewards
        rtgs = rtgs.detach().cpu().numpy() if torch.is_tensor(rtgs) else rtgs
        advantages = advantages.detach().cpu().numpy(
        ) if torch.is_tensor(advantages) else advantages
        values = values.detach().cpu().numpy() if torch.is_tensor(values) else values
        log_probs = log_probs.detach().cpu().numpy(
        ) if torch.is_tensor(log_probs) else log_probs

        # Add all transitions to memory
        for t in range(len(states)):
            self.push(
                states[t],
                actions[t],
                rewards[t],
                rtgs[t],
                advantages[t],
                values[t],
                log_probs[t]
            )
