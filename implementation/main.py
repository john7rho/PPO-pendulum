import gym
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from Modules import Net, ReplayMemory
from torch.distributions import MultivariateNormal

env = gym.make('Pendulum-v1')

# sample hyperparameters
num_timesteps = 200  # T
num_trajectories = 10  # N
num_iterations = 250
epochs = 100

batch_size = 10
learning_rate = 3e-4
eps = 0.2  # clipping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# function to calculate the (discounted) reward-to-go from a sequence of rewards


def calc_reward_togo(rewards, gamma=0.99):
    n = len(rewards)
    reward_togo = np.zeros(n)
    reward_togo[-1] = rewards[-1]
    for i in reversed(range(n-1)):
        reward_togo[i] = rewards[i] + gamma * reward_togo[i+1]

    reward_togo = torch.tensor(reward_togo, dtype=torch.float)
    return reward_togo


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

        # Move networks to GPU if available
        self.policy_net = Net(3, 1).to(device)
        self.critic_net = Net(3, 1).to(device)

        # self.policy_opt = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        # self.critic_opt = torch.optim.Adam(self.critic_net.parameters(), lr=learning_rate)

        self.optimizer = torch.optim.Adam([  # Update both models together
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

        # use fixed std
        self.std = torch.diag(torch.full(size=(1,), fill_value=0.5)).to(device)

        # Add tracking for best model
        self.best_reward = float('-inf')

    def generate_trajectory(self):
        # Basic PPO trajectory collection
        current_state, _ = env.reset()
        states = []
        actions = []
        rewards = []
        log_probs = []

        for t in range(num_timesteps):
            mean = self.policy_net(torch.as_tensor(current_state))
            normal = MultivariateNormal(mean, self.std)
            action = normal.sample().detach()
            log_prob = normal.log_prob(action).detach()

            next_state, reward, terminated, truncated, info = env.step(action)

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

        for iteration in range(num_iterations):  # k

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

                # Calculate the new log probs
                mean = self.policy_net(states)
                normal = MultivariateNormal(mean, self.std)
                new_log_probs = normal.log_prob(actions.unsqueeze(-1))

                r = torch.exp(new_log_probs - log_probs)

                if self.clipping_on == True:
                    clipped_r = torch.clamp(r, 1 - eps, 1 + eps)
                else:
                    clipped_r = r

                new_values = self.critic_net(states).squeeze()
                returns = (advantages + values).detach()

                if self.advantage_on == True:
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

                self.optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                self.optimizer.step()

                actor_loss_list.append(actor_loss.item())
                critic_loss_list.append(critic_loss.item())
                total_loss_list.append(total_loss.item())
                reward_list.append(sum(rewards))

            self.memory.clear()

            avg_actor_loss = sum(actor_loss_list) / len(actor_loss_list)
            avg_critic_loss = sum(critic_loss_list) / len(critic_loss_list)
            avg_total_loss = sum(total_loss_list) / len(total_loss_list)
            avg_reward = sum(reward_list) / len(reward_list)

            train_actor_loss.append(avg_actor_loss)
            train_critic_loss.append(avg_critic_loss)
            train_total_loss.append(avg_total_loss)
            train_reward.append(avg_reward)

            print('Actor loss = ', avg_actor_loss)
            print('Critic loss = ', avg_critic_loss)
            print('Total Loss = ', avg_total_loss)
            print('Reward = ', avg_reward)
            print("")

            # Save best model weights
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                self.save_checkpoint(
                    f'./results/ppo_best_model_{self.clipping_on}_{self.advantage_on}.pt')

            # Save checkpoint weights every 50 iterations
            if iteration % 50 == 0:
                self.save_checkpoint(
                    f'./results/ppo_checkpoint_model_{iteration}_{self.clipping_on}_{self.advantage_on}.pt')

        # Save final weights
        self.save_checkpoint(
            f'./results/ppo_final_model_{self.clipping_on}_{self.advantage_on}.pt')

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
            f'Results for clipping_on={self.clipping_on} and advantage_on={self.advantage_on}\n', fontsize=20)
        fig.tight_layout()
        plt.savefig(
            f'./results/ppo_figure1_{self.clipping_on}_{self.advantage_on}.png')
        fig.show()

        self.show_value_grid()

        # Modify return to include rewards for plotting
        if iteration == num_iterations - 1:
            return train_reward

        return train_reward

    def show_value_grid(self):
        theta = torch.linspace(-np.pi, np.pi, 100)
        theta_dot = torch.linspace(-8, 8, 100)
        values = torch.zeros((len(theta), len(theta_dot)))

        for i, t in enumerate(theta):
            for j, td in enumerate(theta_dot):
                state = (torch.cos(t), torch.sin(t), td)
                values[i, j] = self.critic_net(torch.as_tensor(state))

        fig2 = plt.figure(figsize=(5, 5))
        plt.imshow(values.detach().numpy(), extent=[
                   theta[0], theta[-1], theta_dot[0], theta_dot[-1]], aspect=0.4)
        plt.title('Value grid', fontsize=18)
        plt.xlabel('angle', fontsize=18)
        plt.ylabel('angular velocity', fontsize=18)

        plt.savefig(
            f'./results/ppo_figure2_{self.clipping_on}_{self.advantage_on}.png')
        plt.show()

    def test(self):
        # Update to load the best model
        checkpoint = torch.load(
            f'./results/ppo_best_model_{self.clipping_on}_{self.advantage_on}.pt')
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        current_state, _ = env.reset()

        for i in range(200):
            mean = self.policy_net(torch.as_tensor(current_state))
            normal = MultivariateNormal(mean, self.std)
            action = normal.sample().detach().numpy()

            next_state, reward, terminated, truncated, info = env.step(action)

            env.render()
            current_state = next_state


class PPOPlusPlus(PPO):
    def __init__(self, clipping_on, advantage_on, gamma=0.99, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        super().__init__(clipping_on, advantage_on, gamma)

        # Load guide policy
        self.guide_policy = Net(3, 1).to(device)
        checkpoint = torch.load('./results/expert_policy.pt')
        self.guide_policy.load_state_dict(checkpoint['policy_net_state_dict'])

        # Make beta decay over time (optional)
        self.initial_beta = 0.5
        self.beta_decay = 1
        self.min_beta = 0.1
        self.beta = self.initial_beta

    def generate_trajectory(self):
        # PPO++ specific trajectory collection with rollin/rollout
        if np.random.random() < self.beta:
            # Follow guide policy to get initial state
            with torch.no_grad():
                current_state, _ = env.reset()
                mean = self.guide_policy(torch.as_tensor(current_state))
                normal = MultivariateNormal(mean, self.std)
                action = normal.sample().detach()
                current_state, _, _, _, _ = env.step(action)
        else:
            # Use learned policy's state distribution
            current_state, _ = env.reset()

        # Rest of the trajectory collection is the same as PPO
        states = []
        actions = []
        rewards = []
        log_probs = []

        # Rollout with learned policy
        for t in range(num_timesteps):
            mean = self.policy_net(torch.as_tensor(current_state))
            normal = MultivariateNormal(mean, self.std)
            action = normal.sample().detach()
            log_prob = normal.log_prob(action).detach()

            next_state, reward, terminated, truncated, info = env.step(action)

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

    def train_iteration(self):
        # Decay beta over time
        self.beta = max(self.min_beta, self.beta * self.beta_decay)

        # Rest of training as normal
        return super().train_iteration()


# Add new constants
NUM_SEEDS = 5  # Number of different random seeds to try
SAVE_RESULTS = True  # Whether to save the results


def run_experiments():
    results = defaultdict(list)

    for seed in range(NUM_SEEDS):
        print(f"\nRunning experiments with seed {seed}")

        # Train PPO
        ppo_agent = PPO(clipping_on=True, advantage_on=True, seed=seed)
        ppo_rewards = ppo_agent.train()
        results['PPO'].append(ppo_rewards)

        # Train PPO++
        ppo_plus_agent = PPOPlusPlus(
            clipping_on=True, advantage_on=True, seed=seed)
        ppo_plus_rewards = ppo_plus_agent.train()
        results['PPO++'].append(ppo_plus_rewards)

    # Plot comparison
    plt.figure(figsize=(10, 6))

    # Plot each seed's results
    for model in ['PPO', 'PPO++']:
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
