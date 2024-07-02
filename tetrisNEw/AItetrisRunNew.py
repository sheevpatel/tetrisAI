import gym_tetris
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import SIMPLE_MOVEMENT  # or use MOVEMENT for more actions
import numpy as np
import time
 
from tetrisAINew import DQNAgent, train_dqn

def main():
    # Initialize Tetris environment
    env = gym_tetris.make('TetrisA-v3')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    print(SIMPLE_MOVEMENT)
    

    state_size = (240, 256, 3)  # Typical NES screen resolution
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    num_episodes = 100
    max_steps_per_episode = 10**5 #2 ** 31 - 1
    batch_size = 32
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 1 - 0.03
    min_epsilon = 0.3

    
    #agent.load('tetrisAINew.pth')
    rewards = train_dqn(agent, env, num_episodes, max_steps_per_episode, batch_size, gamma, epsilon, epsilon_decay, min_epsilon)
    agent.save('tetrisAINew.pth')

    env.close()


if __name__ == "__main__":
    main()
