import numpy as np
from homework2 import Hw2Env
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import math
import random
import matplotlib.pyplot as plt
import os

GAMMA = 0.99
EPSILON = 1.0  # random action stuff
EPSILON_DECAY = 0.999  # decay epsilon by 0.999 every EPSILON_DECAY_ITER
EPSILON_DECAY_ITER = 10  # decay epsilon every 10 updates
MIN_EPSILON = 0.1  # minimum epsilon
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
UPDATE_FREQ = 4  # update the network every 4 steps
TARGET_NETWORK_UPDATE_FREQ = 1000  # update the target network every 100 steps
BUFFER_LENGTH = 10000
N_ACTIONS = 8

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__() # 128x128x3
        self.conv1 = nn.Conv2d(3, 32, 4, 2, 1) # 64*64*32
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1) # 32*32*64
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1) # 16*16*128
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1) # 8*8*256
        self.conv5 = nn.Conv2d(256, 512, 4, 2, 1) # 4*4*512
        self.avg = nn.AvgPool2d(kernel_size=[2, 2]) # 2*2*512
        self.fc1 = nn.Linear(2048, 256) # 8
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 8)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.avg(x)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x)) 
        x = F.relu(self.fc3(x)) 
        y = self.fc4(x)   

        return y

class ReplayMemory(object):
    def __init__(self, capacity=BUFFER_LENGTH):
        self.memory = deque(maxlen=capacity)
        
    def push(self, data):
        self.memory.append(data)

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return [] 
        else:
            return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, n_actions=N_ACTIONS):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        
        # Initialize networks
        self.online_net = DeepQNetwork().to(self.device)
        self.target_net = DeepQNetwork().to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        
        # Initialize memory and optimizer
        self.memory = ReplayMemory(BUFFER_LENGTH)
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LEARNING_RATE)
        
        # Initialize counters
        self.steps_done = 0
        self.epsilon = EPSILON
        
        # Create directory for plots
        if not os.path.exists('hw2_plots'):
            os.makedirs('hw2_plots')
            print("Created directory: hw2_plots")
    
    def select_action(self, state):
        self.steps_done += 1
        if random.random() > self.epsilon:
            with torch.no_grad():  # disable gradient 
                state = state.unsqueeze(0) if len(state.shape) == 3 else state
                return self.online_net(state).max(1).indices.view(1, 1)
        else:  # return random action
            action = torch.tensor(np.random.randint(self.n_actions), dtype=torch.long)
            return action
    
    def update_epsilon(self):
        if self.steps_done % EPSILON_DECAY_ITER == 0:
            self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)
        return self.epsilon
    
    def optimize_model(self):
        batch = self.memory.sample(BATCH_SIZE)  # random batch
        if not batch:  # If the batch is empty, return None
            return None
            
        state_batch = []
        action_batch = []
        next_state_batch = []
        reward_batch = []
        done_batch = []

        for transition in batch:
            if transition.state is not None:
                state_batch.append(torch.tensor(transition.state.detach().cpu().numpy(), dtype=torch.float32))
                action_batch.append(transition.action)
                next_state_batch.append(torch.tensor(transition.next_state.detach().cpu().numpy(), dtype=torch.float32))
                reward_batch.append(torch.tensor(transition.reward.item(), dtype=torch.float32))
                done_batch.append(torch.tensor(transition.done, dtype=torch.bool))

        state_batch = torch.stack(state_batch)
        action_batch = torch.tensor(action_batch)
        next_state_batch = torch.stack(next_state_batch)
        reward_batch = torch.stack(reward_batch)
        done_batch = torch.stack(done_batch)

        # Calculate Q-values for current states using the online network
        Q_values = self.online_net(state_batch).gather(1, action_batch.unsqueeze(1))
        # Calculate Q-values for next states using the target network
        next_state_actions = self.online_net(next_state_batch).max(1)[1].unsqueeze(1)
        next_Q_values = self.target_net(next_state_batch).gather(1, next_state_actions)
        # Calculate expected Q-values using the Bellman equation
        expected_Q_values = reward_batch + (GAMMA * next_Q_values.squeeze() * (~done_batch))
        # Calculate loss between Q-values and expected Q-values
        loss = F.smooth_l1_loss(Q_values, expected_Q_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()  # Return the loss value
    
    def train(self, num_episodes=300):
        """Train the DQN agent for a specified number of episodes"""
        env = Hw2Env(n_actions=self.n_actions, render_mode="offscreen")
        
        # Initialize lists to store metrics for plotting
        episode_rewards = []
        episode_losses = []
        epsilons = []
        episode_steps = []
        reward_per_step = []
        
        # Fill the memory with initial transitions
        env.reset()
        state = env.state()
        done = False
        i = 0
        while not done and i < 34:
            action = np.random.randint(self.n_actions)
            new_state, reward, is_terminal, is_truncated = env.step(action)
            done = is_terminal or is_truncated
            transition = Transition(state, action, new_state, reward, done)
            if i > 1:
                self.memory.push(transition)
            state = new_state
            i += 1
        
        # Main training loop
        for episode in range(num_episodes):
            env.reset()
            state = env.state()
            total_reward = 0
            episode_loss = 0
            loss_count = 0
            step_count = 0
            
            # Take initial random action
            action = np.random.randint(self.n_actions)
            new_state, reward, is_terminal, is_truncated = env.step(action)
            done = is_terminal or is_truncated
            
            while True:
                action = self.select_action(state)
                new_state, reward, is_terminal, is_truncated = env.step(action.item())
                total_reward += reward
                done = is_terminal or is_truncated
                
                transition = Transition(state, action, new_state, reward, done)
                self.memory.push(transition)
                
                # Optimize model and get loss
                loss = self.optimize_model()
                if loss is not None:
                    episode_loss += loss
                    loss_count += 1
                
                state = new_state
                epsilon = self.update_epsilon()
                step_count += 1
                
                # Update online network periodically
                if self.steps_done % UPDATE_FREQ == 0:
                    self.online_net.load_state_dict(self.target_net.state_dict())
                
                # Update target network periodically
                if self.steps_done % TARGET_NETWORK_UPDATE_FREQ == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())
                
                if done:
                    break
            
            # Store metrics for this episode
            avg_loss = episode_loss / max(1, loss_count)
            episode_rewards.append(total_reward)
            episode_losses.append(avg_loss)
            epsilons.append(epsilon)
            episode_steps.append(step_count)
            rps = total_reward / max(1, step_count)
            reward_per_step.append(rps)
            
            print(f"Episode {episode + 1}, Total Reward: {total_reward}, RPS: {rps:.4f}, Steps: {step_count}, Avg Loss: {avg_loss}, Epsilon: {epsilon}")
            
            # Plot and save every 10 episodes
            if (episode + 1) % 10 == 0:
                self._save_training_plots(episode, episode_rewards, episode_losses, epsilons)
        
        # Save final plots
        self._save_final_plots(episode_rewards, reward_per_step, episode_losses, epsilons)
        
        # Save models
        torch.save(self.online_net.state_dict(), "online_model.pth")
        torch.save(self.target_net.state_dict(), "target_model.pth")
        
        print("Training completed")
        env.close()
        
        return episode_rewards, reward_per_step, episode_losses, epsilons
    
    def test(self, num_episodes=100, render_mode="gui", model_path="online_model.pth"):
        """Test the trained DQN agent using a saved model"""
        env = Hw2Env(n_actions=self.n_actions, render_mode=render_mode)
        test_rewards = []
        
        # Load the saved model
        test_net = DeepQNetwork().to(self.device)
        test_net.load_state_dict(torch.load(model_path))
        test_net.eval()  # Set to evaluation mode
        
        print(f"Loaded model from {model_path}")
        
        for i in range(num_episodes):
            env.reset()
            state = env.state()
            done = False 
            cum_reward = 0.0
            
            while not done:
                with torch.no_grad():
                    state = state.unsqueeze(0) if len(state.shape) == 3 else state
                    action = test_net(state).max(1).indices.view(1, 1)
                
                state, reward, is_terminal, is_truncated = env.step(action.item())
                done = is_terminal or is_truncated
                cum_reward += reward
            
            test_rewards.append(cum_reward)
            print(f"Test episode {i+1}, Total Reward: {cum_reward}")
        
        env.close()
        
        # Plot test rewards
        plt.figure(figsize=(10, 5))
        plt.plot(test_rewards)
        plt.xlabel('Test Episode')
        plt.ylabel('Total Reward')
        plt.title('Test Performance')
        plt.savefig('hw2_plots/test_performance.png')
        plt.close()
        
        print(f"Average test reward: {sum(test_rewards)/len(test_rewards)}")
        print(f"Test plot has been saved to the 'hw2_plots' directory.")
        
        return test_rewards
    
    def _save_training_plots(self, episode, episode_rewards, episode_losses, epsilons):
        """Save training plots during training"""
        plt.figure(figsize=(15, 10))
        
        # Plot rewards
        plt.subplot(3, 1, 1)
        plt.plot(episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Reward per Episode')
        
        # Plot losses
        plt.subplot(3, 1, 2)
        plt.plot(episode_losses)
        plt.xlabel('Episode')
        plt.ylabel('Average Loss')
        plt.title('Loss per Episode')
        
        # Plot epsilon
        plt.subplot(3, 1, 3)
        plt.plot(epsilons)
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.title('Epsilon Decay')
        
        plt.tight_layout()
        plt.savefig(f'hw2_plots/training_progress_episode_{episode+1}.png')
        plt.close()
    
    def _save_final_plots(self, episode_rewards, reward_per_step, episode_losses, epsilons):
        """Save final plots after training"""
        # Plot 1: Rewards over episodes
        plt.figure(figsize=(10, 6))
        plt.plot(episode_rewards, 'b-', label='Episode Reward')
        # Moving average of rewards
        window_size = min(10, len(episode_rewards))
        if window_size > 0:
            moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(episode_rewards)), moving_avg, 'r-', label=f'{window_size}-episode Moving Average')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Reward per Episode')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('hw2_plots/reward_per_episode.png')
        plt.close()

        # Plot 2: Reward Per Step (RPS) over episodes
        plt.figure(figsize=(10, 6))
        plt.plot(reward_per_step, 'g-', label='Reward Per Step (RPS)')
        # Moving average of RPS
        window_size = min(10, len(reward_per_step))
        if window_size > 0:
            moving_avg_rps = np.convolve(reward_per_step, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(reward_per_step)), moving_avg_rps, 'r-', label=f'{window_size}-episode Moving Average')
        plt.xlabel('Episode')
        plt.ylabel('Reward Per Step')
        plt.title('Reward Per Step (RPS) over Episodes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('hw2_plots/reward_per_step.png')
        plt.close()

        # Additional training metrics (all in one figure)
        plt.figure(figsize=(15, 15))

        # Plot rewards
        plt.subplot(4, 1, 1)
        plt.plot(episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Reward per Episode')
        plt.grid(True, alpha=0.3)

        # Plot RPS
        plt.subplot(4, 1, 2)
        plt.plot(reward_per_step)
        plt.xlabel('Episode')
        plt.ylabel('Reward Per Step')
        plt.title('Reward Per Step (RPS)')
        plt.grid(True, alpha=0.3)

        # Plot losses
        plt.subplot(4, 1, 3)
        plt.plot(episode_losses)
        plt.xlabel('Episode')
        plt.ylabel('Average Loss')
        plt.title('Loss per Episode')
        plt.grid(True, alpha=0.3)

        # Plot epsilon
        plt.subplot(4, 1, 4)
        plt.plot(epsilons)
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.title('Epsilon Decay')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('hw2_plots/all_training_metrics.png')
        plt.close()

# Main execution
if __name__ == "__main__":
    # Create agent
    agent = DQNAgent(n_actions=N_ACTIONS)
    
    # Option to train or load a pre-trained model
    train_new_model = True  # Set to False to skip training and only test
    
    if train_new_model:
        # Train the agent
        print("Starting training...")
        episode_rewards, reward_per_step, episode_losses, epsilons = agent.train(num_episodes=300)
        print("Training completed.")
    
    # Test the agent using the saved model
    print("Starting testing with saved model...")
    test_rewards = agent.test(num_episodes=100, model_path="online_model.pth")
    
    print(f"All plots have been saved to the 'hw2_plots' directory.")