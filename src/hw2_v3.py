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
    
    def act(self):
        pass

class ReplayMemory(object):

    def __init__(self, capacity = BUFFER_LENGTH):
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

def update_epsilon(iteration):

    global EPSILON, EPSILON_DECAY, EPSILON_DECAY_ITER, MIN_EPSILON
    if iteration % EPSILON_DECAY_ITER == 0:
        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
    return EPSILON

def select_action(state):
    global steps_done, EPSILON
    steps_done += 1
    if random.random() > EPSILON:
        with torch.no_grad():  # disable gradient 
            state = state.unsqueeze(0)
            # print("not random")
            return online_net(state).max(1).indices.view(1, 1)  # Get max from output, reshapes it into a 1x1 tensor
            
    else:  # return random action
        action = torch.tensor(np.random.randint(N_ACTIONS), dtype=torch.long)
        # print("random")
        return action

def optimize_model(memory, online_net, target_net, optimizer):

    global BATCH_SIZE, GAMMA
   
    batch = memory.sample(BATCH_SIZE)  # random batch
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
    Q_values = online_net(state_batch).gather(1, action_batch.unsqueeze(1))
    # Calculate Q-values for next states using the target network
    next_state_actions = online_net(next_state_batch).max(1)[1].unsqueeze(1)  # Unsqueeze to maintain dimension
    next_Q_values = target_net(next_state_batch).gather(1, next_state_actions)
    # Calculate expected Q-values using the Bellman equation
    expected_Q_values = reward_batch + (GAMMA * next_Q_values.squeeze() * (~done_batch))  # Apply the mask (~done_batch) to exclude terminal states
    # Calculate loss between Q-values and expected Q-values
    loss = F.smooth_l1_loss(Q_values, expected_Q_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()  # Return the loss value

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = Hw2Env(n_actions=N_ACTIONS, render_mode="offscreen")

online_net = DeepQNetwork().to(device)
target_net = DeepQNetwork().to(device)
target_net.load_state_dict(online_net.state_dict())

memory = ReplayMemory(BUFFER_LENGTH)
optimizer = optim.Adam(online_net.parameters(), lr=LEARNING_RATE)

loss_fn = nn.MSELoss()

steps_done = 0
env.reset()
state = env.state()
done = False
i = 0

# Initialize lists to store metrics for plotting
episode_rewards = []
episode_losses = []
epsilons = []
episode_steps = []  # Track steps per episode for RPS calculation
reward_per_step = []  # Track reward per step (RPS)

while not done and i < 34:
    action = np.random.randint(N_ACTIONS)
    new_state, reward, is_terminal, is_truncated = env.step(action)
    done = is_terminal or is_truncated
    transition = Transition(state, action, new_state, reward, done)
    if(i > 1):    
        memory.push(transition)
    state = new_state
    i += 1

for episode in range(300):  # Number of episodes
    env.reset()
    state = env.state()
    total_reward = 0  # Initialize total reward for this episode
    episode_loss = 0  # Initialize total loss for this episode
    loss_count = 0    # Count number of loss updates
    step_count = 0    # Count steps in this episode
    
    action = np.random.randint(N_ACTIONS)
    new_state, reward, is_terminal, is_truncated = env.step(action)
    done = is_terminal or is_truncated
    while True:
        # print("steps_done: ", steps_done)
        action = select_action(state)  # Select action using epsilon-greedy strategy
        
        new_state, reward, is_terminal, is_truncated = env.step(action.item())  # Take action in the environment
        total_reward += reward  # Accumulate total reward for this episode
        done = is_terminal or is_truncated
        transition = Transition(state, action, new_state, reward, done)
        memory.push(transition)
        
        # Optimize model and get loss
        loss = optimize_model(memory, online_net, target_net, optimizer)
        if loss is not None:
            episode_loss += loss
            loss_count += 1
            
        state = new_state  # Update current state
        epsilon = update_epsilon(steps_done)
        step_count += 1  # Increment step counter for this episode

        # Update online network every 4 steps
        if steps_done % UPDATE_FREQ == 0:
            #print("Updating online network")
            online_net.load_state_dict(target_net.state_dict())
        
        # Update target network every 1000 steps
        if steps_done % TARGET_NETWORK_UPDATE_FREQ == 0:
            #print("Updating target network")
            target_net.load_state_dict(online_net.state_dict())

        if done:
            break  # Break out of the loop if episode is done

    # Store metrics for this episode
    episode_rewards.append(total_reward)
    avg_loss = episode_loss / max(1, loss_count)  # Avoid division by zero
    episode_losses.append(avg_loss)
    epsilons.append(epsilon)
    episode_steps.append(step_count)
    # Calculate reward per step (RPS)
    rps = total_reward / max(1, step_count)  # Avoid division by zero
    reward_per_step.append(rps)
    
    print(f"Episode {episode + 1}, Total Reward: {total_reward}, RPS: {rps:.4f}, Steps: {step_count}, Avg Loss: {avg_loss}, Epsilon: {epsilon}")
    
    # Create hw2_plots directory if it doesn't exist
    import os
    if not os.path.exists('hw2_plots'):
        os.makedirs('hw2_plots')
        print("Created directory: hw2_plots")
    
    # Plot and save every 10 episodes (but don't display)
    if (episode + 1) % 10 == 0:
        # Create figure with 3 subplots
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

# Make sure the hw2_plots directory exists
import os
if not os.path.exists('hw2_plots'):
    os.makedirs('hw2_plots')
    print("Created directory: hw2_plots")

# Plot 1: Rewards over episodes (as requested)
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

# Plot 2: Reward Per Step (RPS) over episodes (as requested)
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

plt.tight_layout()
plt.savefig('dqn_training_metrics.png')  # Save figure to file
plt.show()

# Save models
torch.save(online_net.state_dict(), "online_model.pth")
torch.save(target_net.state_dict(), "target_model.pth")

print("done with traininggg")

# Test the trained model
env2 = Hw2Env(n_actions=N_ACTIONS, render_mode="gui")
test_rewards = []

for i in range(100):
    env2.reset()
    state = env2.state()
    done = False 
    cum_reward = 0.0
    
    while not done:
        with torch.no_grad():
            state = state.unsqueeze(0) if len(state.shape) == 3 else state
            action = online_net(state).max(1).indices.view(1, 1)
        
        state, reward, is_terminal, is_truncated = env2.step(action.item())
        done = is_terminal or is_truncated
        cum_reward += reward
    
    test_rewards.append(cum_reward)
    print(f"Test episode {i+1}, Total Reward: {cum_reward}")

env2.close()

# Plot test rewards
plt.figure(figsize=(10, 5))
plt.plot(test_rewards)
plt.xlabel('Test Episode')
plt.ylabel('Total Reward')
plt.title('Test Performance')
plt.savefig('hw2_plots/test_performance.png')
plt.close()

print(f"Average test reward: {sum(test_rewards)/len(test_rewards)}")
print(f"All plots have been saved to the 'hw2_plots' directory.")