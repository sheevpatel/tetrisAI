import gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    print("MPS device not found.")

# Define the Deep Q-Network class using PyTorch
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 22 * 31, 64)  # Adjusted input size based on conv layers
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = x / 255.0  # Normalize input
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #x = x.view(x.size(0), -1)  # Flatten the tensor
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Define the DQN Agent class
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)  # Replay memory
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        self.last_action_was_rotation = False
        self.rotation_actions = [0, 5]  # Assuming 1 and 2 are 'A' and 'B' for rotation


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
        else :

            state = torch.FloatTensor(state).unsqueeze(0).permute(0, 3, 1, 2)  # Adjust for convolutional layers
            with torch.no_grad():
                act_values = self.model(state)
            action = torch.argmax(act_values).item()

        if self.last_action_was_rotation and action in self.rotation_actions:
            action = random.choice([a for a in range(self.action_size) if a not in self.rotation_actions])

        self.last_action_was_rotation = action in self.rotation_actions

        return action


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state.copy()).unsqueeze(0).permute(0, 3, 1, 2)  # Make a copy to avoid stride issues
            #next_state = torch.FloatTensor(next_state.copy()).permute(2, 0, 1).unsqueeze(0)
            next_state = torch.FloatTensor(next_state.copy()).unsqueeze(0).permute(0, 3, 1, 2)
            target = reward
            if not done:
                print(next_state)
                #target += self.gamma * torch.max(self.model(next_state)).item()n
                next_state_value = self.model(next_state)
                max_next_state_value = torch.max(next_state_value, dim=1)[0].item()  # Adjust dimension for max operation
                target += self.gamma * max_next_state_value
                #target += self.gamma * torch.max(self.model(state)).item()
            target_f = self.model(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()

    def load(self, name):
        self.model.load_state_dict(torch.load(name))
        self.model.eval()

    def save(self, name):
        torch.save(self.model.state_dict(), name)


def train_dqn(agent, env, num_episodes, max_steps_per_episode, batch_size, gamma, epsilon, epsilon_decay, min_epsilon):
    rewards = []
    max_reward = -1.0

    for episode in range(num_episodes):
        state = env.reset()
        
        total_reward = 0
        for step in range(max_steps_per_episode):
            '''
            action = agent.act(state)
            state_step = env.step(action)
            env.render()       
            observation, reward, done, info = state_step
            total_reward += reward
            agent.remember(observation, action, reward, info.get('next_piece'), done)  
            state = observation
            '''

            action = agent.act(state)
            next_state_step = env.step(action)
            env.render()
            next_observation, reward, done, info = next_state_step
            if reward > 0:
                reward *= 1000
            total_reward += reward
            agent.remember(state, action, reward, next_observation, done)
            state = next_observation
        
            if done:
                break

        #agent.replay(batch_size)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        rewards.append(total_reward)

        if total_reward > max_reward:
            max_reward = total_reward
            print(f"New max reward {max_reward} achieved with epsilon {epsilon:.2f}")
        if episode % 50 == 0:
            agent.save('tetrisAINew.pth')
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    print(f"Max reward was: {max(rewards)}")
    return rewards

'''
def compute_reward(info):
    
    height_reward = 0
    board_height = info.get('board_height')
    if board_height > 10 and board_height <= 13:
        height_reward = 0.5 * board_height
    elif board_height > 13 and board_height < 17:
        height_reward = 0.8 * board_height
    elif board_height >= 17:
        height_reward = 1.5 * board_height
    else:
        if board_height < 5:
            height_reward = -0.1 * board_height #will become postive
        else :
            height_reward = 0.1 * board_height
    
    
    score_reward = info.get('score') * 10
    
    line_score_rule = {
        1: 10,
        2:40,
        3:120,
        4:400
    }
    line_count = info.get('number_of_lines', 0)
    lines_cleared = line_score_rule.get(line_count, 0)
    return score_reward + lines_cleared - height_reward
'''

def observation_to_grid(observation, threshold=128):
    height, width, _ = observation.shape
    grid = np.zeros((height, width), dtype=int)
    for i in range(height):
        for j in range(width):
            if observation[i, j, 0] < threshold:
                grid[i, j] = 1
    return grid

def compute_reward(info, grid):
    def get_aggregate_height(grid):
        heights = [0] * len(grid[0])
        for col in range(len(grid[0])):
            for row in range(len(grid)):
                if grid[row][col] != 0:
                    heights[col] = len(grid) - row
                    break
        return sum(heights)

    def get_complete_lines(grid):
        complete_lines = 0
        for row in grid:
            if all(row):
                complete_lines += 1
        return complete_lines

    def get_holes(grid):
        holes = 0
        for col in range(len(grid[0])):
            block_found = False
            for row in range(len(grid)):
                if grid[row][col] != 0:
                    block_found = True
                elif block_found and grid[row][col] == 0:
                    holes += 1
        return holes

    def get_bumpiness(grid):
        heights = [0] * len(grid[0])
        for col in range(len(grid[0])):
            for row in range(len(grid)):
                if grid[row][col] != 0:
                    heights[col] = len(grid) - row
                    break
        bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))
        return bumpiness

    aggregate_height = get_aggregate_height(grid)
    complete_lines = get_complete_lines(grid)
    holes = get_holes(grid)
    bumpiness = get_bumpiness(grid)

    a, b, c, d = -0.5, 1.0, -0.7, -0.2  # Example weights

    score = (a * aggregate_height) + (b * complete_lines) + (c * holes) + (d * bumpiness)

    return score
