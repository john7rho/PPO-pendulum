from torch.distributions import MultivariateNormal
from Modules import Net, ReplayMemory
import gym
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import os
import multiprocessing as mp
import torch.nn.functional as F

# Set multiprocessing start method
mp.set_start_method('spawn', force=True)


# Sample hyperparameters
num_timesteps = 200  # T
num_trajectories = 10  # N
num_iterations = 110
epochs = 100

batch_size = 10
learning_rate = 3e-4
eps = 0.2  # clipping

# Move tensor device configuration to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to calculate the (discounted) reward-to-go from a sequence of rewards


def calc_reward_togo(rewards, gamma=0.99):
    n = len(rewards)
    reward_togo = np.zeros(n)
    reward_togo[-1] = rewards[-1]
    for i in reversed(range(n-1)):
        reward_togo[i] = rewards[i] + gamma * reward_togo[i+1]

    reward_togo = torch.tensor(reward_togo, dtype=torch.float)
    return reward_togo

# Compute advantage estimates (as done in PPO paper)


def calc_advantages(rewards, values, gamma=0.99, lambda_=1):
    advantages = torch.zeros_like(torch.as_tensor(rewards))
    sum = 0
    for t in reversed(range(len(rewards)-1)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        sum = delta + gamma * lambda_ * sum
        advantages[t] = sum

    return advantages


class PPO:
    def __init__(self, clipping_on, advantage_on, gamma=0.99, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # Move networks to GPU if available
        self.policy_net = Net(3, 1).to(device)
        self.critic_net = Net(3, 1).to(device)

        self.optimizer = torch.optim.Adam([
            {'params': self.policy_net.parameters(), 'lr': learning_rate},
            {'params': self.critic_net.parameters(), 'lr': learning_rate}
        ])

        self.memory = ReplayMemory(batch_size)

        self.gamma = gamma
        self.lambda_ = 1
        self.vf_coef = 1  # c1
        self.entropy_coef = 0.01  # c2

        self.clipping_on = clipping_on
        self.advantage_on = advantage_on

        # Use fixed std
        self.std = torch.diag(torch.full(size=(1,), fill_value=0.5)).to(device)

        # Add tracking for best model
        self.best_reward = float('-inf')

        # Save seed
        self.seed = seed

        # Create environment and seed it
        self.env = gym.make('Pendulum-v1')
        if seed is not None:
            self.env.reset(seed=seed)

    def generate_trajectory(self):
        # Basic PPO trajectory collection
        current_state, _ = self.env.reset()
        states = []
        actions = []
        rewards = []
        log_probs = []

        for t in range(num_timesteps):
            mean = self.policy_net(torch.as_tensor(current_state))
            normal = MultivariateNormal(mean, self.std)
            action = normal.sample().detach()
            log_prob = normal.log_prob(action).detach()

            next_state, reward, terminated, truncated, info = self.env.step(
                action)

            states.append(current_state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)

            current_state = next_state

        # Process batch and store in memory
        states_tensor = torch.as_tensor(states, device=device)
        rtg = calc_reward_togo(torch.as_tensor(
            rewards, device=device), self.gamma)
        values = self.critic_net(states_tensor).squeeze()
        advantages = calc_advantages(
            rewards, values.detach(), self.gamma, self.lambda_)

        for t in range(len(rtg)):
            self.memory.push(states[t], actions[t], rewards[t],
                             rtg[t], advantages[t], values[t], log_probs[t])

    def save_checkpoint(self, filename):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'critic_net_state_dict': self.critic_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_reward': self.best_reward,
        }, filename)

    def train(self):

        train_actor_loss = []
        train_critic_loss = []
        train_total_loss = []
        train_reward = []

        for iteration in range(num_iterations):

            # Collect a number of trajectories and save the transitions in replay memory
            for _ in range(num_trajectories):
                self.generate_trajectory()

            # Sample from replay memory
            states, actions, rewards, rewards_togo, advantages, values, log_probs, batches = self.memory.sample()

            # Move tensors to device in batch
            states = states.to(device)
            actions = actions.to(device)
            rewards_togo = rewards_togo.to(device)
            advantages = advantages.to(device)
            values = values.to(device)
            log_probs = log_probs.to(device)

            actor_loss_list = []
            critic_loss_list = []
            total_loss_list = []
            reward_list = []
            for _ in range(epochs):

                # Calculate the new log prob
                mean = self.policy_net(states)
                normal = MultivariateNormal(mean, self.std)
                new_log_probs = normal.log_prob(actions.unsqueeze(-1))

                r = torch.exp(new_log_probs - log_probs)

                if self.clipping_on:
                    clipped_r = torch.clamp(r, 1 - eps, 1 + eps)
                else:
                    clipped_r = r

                new_values = self.critic_net(states).squeeze()
                returns = (advantages + values).detach()

                if self.advantage_on:
                    actor_loss = (-torch.min(r * advantages,
                                  clipped_r * advantages)).mean()
                    critic_loss = nn.MSELoss()(new_values.float(), returns.float())
                else:
                    actor_loss = (-torch.min(r * rewards_togo,
                                  clipped_r * rewards_togo)).mean()
                    critic_loss = nn.MSELoss()(new_values.float(), rewards_togo.float())

                # Calculate total loss
                total_loss = actor_loss + \
                    (self.vf_coef * critic_loss) - \
                    (self.entropy_coef * normal.entropy().mean())

                # Update policy and critic network
                self.optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                self.optimizer.step()

                actor_loss_list.append(actor_loss.item())
                critic_loss_list.append(critic_loss.item())
                total_loss_list.append(total_loss.item())
                reward_list.append(sum(rewards))

            # Clear replay memory
            self.memory.clear()

            avg_actor_loss = sum(actor_loss_list) / len(actor_loss_list)
            avg_critic_loss = sum(critic_loss_list) / len(critic_loss_list)
            avg_total_loss = sum(total_loss_list) / len(total_loss_list)
            avg_reward = sum(reward_list) / len(reward_list)

            train_actor_loss.append(avg_actor_loss)
            train_critic_loss.append(avg_critic_loss)
            train_total_loss.append(avg_total_loss)
            train_reward.append(avg_reward)

            if iteration % 10 == 0:
                print(f"PPO Iteration {iteration}, Seed {self.seed}")
                print('Actor loss = ', avg_actor_loss)
                print('Critic loss = ', avg_critic_loss)
                print('Total Loss = ', avg_total_loss)
                print('Reward = ', avg_reward)
                print("")

        # Save final weights
        self.save_checkpoint(
            f'./results/ppo_final_model_{self.clipping_on}_{self.advantage_on}_seed{self.seed}.pt')

        # Plotting
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].plot(range(len(train_actor_loss)),
                     train_actor_loss, 'r', label='Actor Loss')
        axes[0].set_title('Actor Loss', fontsize=18)

        axes[1].plot(range(len(train_critic_loss)),
                     train_critic_loss, 'b', label='Critic Loss')
        axes[1].set_title('Critic Loss', fontsize=18)

        axes[2].plot(range(len(train_total_loss)),
                     train_total_loss, 'm', label='Total Loss')
        axes[2].set_title('Total Loss', fontsize=18)

        axes[3].plot(range(len(train_reward)), train_reward,
                     'orange', label='Accumulated Reward')
        axes[3].set_title('Accumulated Reward', fontsize=18)

        fig.suptitle(
            f'Results for clipping_on={self.clipping_on} and advantage_on={self.advantage_on}\nSeed={self.seed}', fontsize=20)
        fig.tight_layout()
        plt.savefig(
            f'./results/ppo_figure1_{self.clipping_on}_{self.advantage_on}_seed{self.seed}.png')
        plt.close(fig)

        self.show_value_grid()

        return train_reward

    def show_value_grid(self):

        # Sweep theta and theta_dot and find all states
        theta = torch.linspace(-np.pi, np.pi, 100)
        theta_dot = torch.linspace(-8, 8, 100)
        values = torch.zeros((len(theta), len(theta_dot)))

        for i, t in enumerate(theta):
            for j, td in enumerate(theta_dot):
                state = (torch.cos(t), torch.sin(t), td)
                values[i, j] = self.critic_net(torch.as_tensor(state))

        # Display the resulting values using imshow
        fig2 = plt.figure(figsize=(5, 5))
        plt.imshow(values.detach().numpy(), extent=[
                   theta[0], theta[-1], theta_dot[0], theta_dot[-1]], aspect=0.4)
        plt.title('Value grid', fontsize=18)
        plt.xlabel('angle', fontsize=18)
        plt.ylabel('angular velocity', fontsize=18)

        plt.savefig(
            f'./results/ppo_figure2_{self.clipping_on}_{self.advantage_on}_seed{self.seed}.png')
        plt.close(fig2)

    def test(self):
        # Update to load the best model
        checkpoint = torch.load(
            f'./results/ppo_best_model_{self.clipping_on}_{self.advantage_on}_seed{self.seed}.pt')
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        current_state, _ = self.env.reset()

        for i in range(200):
            mean = self.policy_net(torch.as_tensor(current_state))
            normal = MultivariateNormal(mean, self.std)
            action = normal.sample().detach().numpy()

            next_state, reward, terminated, truncated, info = self.env.step(
                action)

            self.env.render()
            current_state = next_state


class PPOPlusPlus(PPO):
    def __init__(self, clipping_on, advantage_on, gamma=0.99, seed=None):
        super().__init__(clipping_on, advantage_on, gamma=gamma, seed=seed)

        # Load guide policy
        self.guide_policy = Net(3, 1).to(device)
        checkpoint = torch.load(
            './results/expert_policy.pt', weights_only=True)
        self.guide_policy.load_state_dict(checkpoint['policy_net_state_dict'])
        self.guide_policy.eval()  # Set to evaluation mode

        # Hyperparameters
        self.initial_beta = 0.5
        self.beta = self.initial_beta
        self.min_beta = 0.1
        self.beta_decay = 0.995

        # BC parameters
        self.bc_coef = 0.1  # Reduced from 1.0
        self.bc_decay = 0.99
        self.min_bc_coef = 0.01  # Reduced from 0.1

        # Adjust value coefficients
        self.vf_coef = 0.5  # Reduced from 1.0
        self.entropy_coef = 0.01

        # Normalize advantages flag
        self.normalize_advantages = True

    def generate_trajectory(self):
        states = []
        actions = []
        rewards = []
        log_probs = []

        current_state, _ = self.env.reset()

        # Use guide policy with probability beta
        if random.random() < self.beta:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(current_state).to(device)
                mean = self.guide_policy(state_tensor)
                normal = MultivariateNormal(mean, self.std)
                action = normal.sample()
                next_state, _, _, _, _ = self.env.step(action.cpu().numpy())
                current_state = next_state

        # Collect trajectory
        for t in range(num_timesteps):
            state_tensor = torch.FloatTensor(current_state).to(device)

            with torch.no_grad():
                mean = self.policy_net(state_tensor)
                normal = MultivariateNormal(mean, self.std)
                action = normal.sample()
                log_prob = normal.log_prob(action)

            # Keep action as 3-dimensional
            action_np = action.cpu().numpy()
            next_state, reward, terminated, truncated, _ = self.env.step(
                action_np)

            # Store with consistent shapes
            # shape [3]
            states.append(np.array(current_state, dtype=np.float32))
            actions.append(np.array(action_np, dtype=np.float32)
                           )     # shape [3]
            rewards.append(reward)
            log_probs.append(log_prob.item())

            current_state = next_state

            if terminated or truncated:
                break

        # Process batch
        states_tensor = torch.FloatTensor(states).to(device)

        with torch.no_grad():
            values = self.critic_net(states_tensor).squeeze()
            returns = calc_reward_togo(torch.tensor(
                rewards, device=device), self.gamma)
            advantages = calc_advantages(
                rewards, values, self.gamma, self.lambda_)

            if self.normalize_advantages:
                advantages = (advantages - advantages.mean()) / \
                    (advantages.std() + 1e-8)

        # Store in memory
        for t in range(len(returns)):
            self.memory.push(
                states[t],      # shape [3]
                actions[t],     # shape [3]
                rewards[t],
                returns[t].item(),
                advantages[t].item(),
                values[t].item(),
                log_probs[t]
            )

    # In PPOPlusPlus class
    def train(self):
        train_actor_loss = []
        train_critic_loss = []
        train_total_loss = []
        train_reward = []

        for iteration in range(num_iterations):
            trajectory_rewards = []  # Track full trajectory rewards
            actor_loss_list = []
            critic_loss_list = []
            total_loss_list = []
            reward_list = []

            # Collect trajectories
            for _ in range(num_trajectories):
                self.generate_trajectory()
                trajectory_rewards.append(
                    sum(self.memory.rewards[-num_timesteps:]))

            # Sample from memory
            states, actions, rewards, returns, advantages, values, old_log_probs, batches = self.memory.sample()
            if len(batches) == 0:
                continue

            # Move tensors to device
            states = states.to(device)
            actions = actions.to(device)
            returns = returns.to(device)
            advantages = advantages.to(device)
            values = values.to(device)
            old_log_probs = old_log_probs.to(device)

            # Training loop
            for _ in range(epochs):
                mean = self.policy_net(states)
                normal = MultivariateNormal(mean, self.std)
                new_log_probs = normal.log_prob(actions)

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages

                actor_loss = -torch.min(surr1, surr2).mean()

                new_values = self.critic_net(states).squeeze()
                critic_loss = F.mse_loss(new_values, returns)

                with torch.no_grad():
                    guide_mean = self.guide_policy(states)
                bc_loss = F.mse_loss(mean, guide_mean)

                total_loss = (
                    actor_loss +
                    self.vf_coef * critic_loss +
                    self.bc_coef * bc_loss -
                    self.entropy_coef * normal.entropy().mean()
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy_net.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(
                    self.critic_net.parameters(), 0.5)
                self.optimizer.step()

                actor_loss_list.append(actor_loss.item())
                critic_loss_list.append(critic_loss.item())
                total_loss_list.append(total_loss.item())
                reward_list.append(np.mean(trajectory_rewards))

            # Update coefficients
            self.beta = max(self.min_beta, self.beta * self.beta_decay)
            self.bc_coef = max(self.min_bc_coef, self.bc_coef * self.bc_decay)

            # Clear memory
            self.memory.clear()

            # Calculate averages
            avg_actor_loss = sum(actor_loss_list) / len(actor_loss_list)
            avg_critic_loss = sum(critic_loss_list) / len(critic_loss_list)
            avg_total_loss = sum(total_loss_list) / len(total_loss_list)
            avg_reward = sum(reward_list) / len(reward_list)

            # Store metrics
            train_actor_loss.append(avg_actor_loss)
            train_critic_loss.append(avg_critic_loss)
            train_total_loss.append(avg_total_loss)
            train_reward.append(avg_reward)

            # Log progress
            if iteration % 10 == 0:
                print(f"PPO++ Iteration {iteration}, Seed {self.seed}")
                print('Actor loss = ', avg_actor_loss)
                print('Critic loss = ', avg_critic_loss)
                print('Total Loss = ', avg_total_loss)
                print('Reward = ', avg_reward)
                print("")

        return train_reward


# Add new constants
NUM_SEEDS = 5  # Number of different random seeds to try
SAVE_RESULTS = True  # Whether to save the results


def run_experiment_for_seed(seed):
    # Initialize results dictionary
    seed_results = {}

    print(f"\nRunning experiments with seed {seed} in process {os.getpid()}")

    # Train PPO
    ppo_agent = PPO(clipping_on=True, advantage_on=True, seed=seed)
    ppo_rewards = ppo_agent.train()
    seed_results['PPO'] = ppo_rewards

    # Train PPO++
    ppo_plus_agent = PPOPlusPlus(
        clipping_on=True, advantage_on=True, seed=seed)
    ppo_plus_rewards = ppo_plus_agent.train()
    seed_results['PPO++'] = ppo_plus_rewards

    return seed_results


def run_experiments():
    results = defaultdict(list)

    seeds = range(NUM_SEEDS)

    with mp.Pool() as pool:
        all_seed_results = pool.map(run_experiment_for_seed, seeds)

    # Collect results
    for seed_result in all_seed_results:
        for model in seed_result:
            results[model].append(seed_result[model])

    # Plot comparison
    plt.figure(figsize=(10, 6))

    # Plot each seed's results
    for model in ['PPO', 'PPO++']:
        # for model in ['PPO++']:
        rewards_array = np.array(results[model])
        mean_rewards = np.mean(rewards_array, axis=0)
        std_rewards = np.std(rewards_array, axis=0)

        x = range(len(mean_rewards))
        plt.plot(x, mean_rewards, label=model)
        plt.fill_between(x,
                         mean_rewards - std_rewards,
                         mean_rewards + std_rewards,
                         alpha=0.2)

    plt.title('PPO vs PPO++ Training Rewards (Multiple Seeds)')
    plt.xlabel('Iteration')
    plt.ylabel('Average Reward')
    plt.legend()

    if SAVE_RESULTS:
        plt.savefig('./results/ppo_comparison.png')
    plt.show()


if __name__ == '__main__':
    run_experiments()
