import gym
import numpy as np
import time

from tetrisAI import DQNAgent, train_dqn

def main():
    # Initialize Tetris environment
    env = gym.make('ALE/Tetris-v5', obs_type='rgb', render_mode = None)
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    num_episodes = 10000
    max_steps_per_episode = 5000
    batch_size = 32
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 1 - 0.00029
    min_epsilon = 0.1

    agent.load('tetrisAI.pth')
    rewards = train_dqn(agent, env, num_episodes, max_steps_per_episode, batch_size, gamma, epsilon, epsilon_decay, min_epsilon)
    agent.save('tetrisAI.pth')
    env.close()
if __name__ == "__main__":
    main()
