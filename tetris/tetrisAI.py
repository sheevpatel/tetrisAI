import gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



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
        self.fc1 = nn.Linear(22528, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = x / 255.0  # Normalize input
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        #print(f"Shape after conv layers: {x.shape}")
        x = F.relu(self.fc1(x))
        return self.fc2(x)

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

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).permute(0, 3, 1, 2)  # Adjust for convolutional layers
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        
        
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).permute(2, 0, 1).unsqueeze(0)
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state)).item()

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
            action = agent.act(state)
            state_step = env.step(action)
            observation, reward, terminated,truncated, _ = state_step
            #print(observation[1])
            
            #print(reward)
            
            #reward = compute_custom_reward(state, observation, reward, terminated)
            
            agent.remember(state, action, reward, observation, terminated)
            state = observation
            total_reward += reward
            #print('total_reward on this step b4: ', total_reward)
            total_reward +=1
            #print('total_reward on this step after: ', total_reward)
            if terminated:
                #print(observation)
                #print(observation.ndim)
                #time.sleep(5)
                total_reward -= 100
                break
            
                
        #agent.replay(batch_size)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        rewards.append(total_reward)

        if total_reward > max_reward:
            max_reward = total_reward
            print(f"New max reward {max_reward} achieved with epsilon {epsilon:.2f}")
        if episode % 200 == 0:
            agent.save('tetrisAI.pth')
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")
    
    print(f"Max reward was: {max(rewards)}")
    return rewards

def compute_custom_reward(state, observation, reward, terminated):

    #lines_cleared_reward = get_lines_cleared(observation)  
    
    game_over_penalty = -10 if terminated else 0

    #height_penalty = -0.1 * get_max_height(observation)
    
    #hole_penalty = 0 #-0.5 * count_holes(observation)

    # Combine all reward components
    #custom_reward = lines_cleared_reward + game_over_penalty + height_penalty + hole_penalty
    custom_reward = game_over_penalty
    return custom_reward

def get_max_height(observation):
    raise Exception("Need to implement")

def count_holes(observation):
    raise Exception("Need to implement")

def get_lines_cleared(observation):
    raise Exception("Need to implement")
