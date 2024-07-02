import gym_tetris
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import SIMPLE_MOVEMENT  # or use MOVEMENT for more actions
import numpy as np
from pynput import keyboard
import threading
import time

from sheev_tetris_AI import DQNAgent, train_dqn, save_human_data, load_human_data

# Map keyboard keys to actions
key_action_map = {
    keyboard.KeyCode.from_char('d'): 3,   # Move right
    keyboard.KeyCode.from_char('a'): 4,   # Move right
    keyboard.KeyCode.from_char('s'): 2,   # Move down
    keyboard.KeyCode.from_char('w'): 1,   # Drop
    keyboard.Key.space: 5,                # Rotate counterclockwise
    keyboard.Key.shift: 0                 # Rotate clockwise
}

current_action = None
auto_drop_speed = 0.1  # Adjust this value for the auto drop speed

def on_press(key):
    global current_action
    if key in key_action_map:
        current_action = key_action_map[key]

def on_release(key):
    global current_action
    if key in key_action_map and current_action == key_action_map[key]:
        current_action = None

def get_key_action():
    global current_action
    while True:
        if current_action is not None:
            return current_action
        #time.sleep(auto_drop_speed)  # Auto drop slowly if no key is pressed

def collect_human_data(env, num_games):
    human_data = []
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    for game in range(num_games):
        state = env.reset()
        done = False
        while not done:
            env.render()
            action = get_key_action()
            next_state, reward, done, info = env.step(action)
            human_data.append((state, action, reward, next_state, done))
            state = next_state
    save_human_data(human_data)
    return human_data

def main():
    # Initialize Tetris environment
    env = gym_tetris.make('TetrisA-v3')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    print(SIMPLE_MOVEMENT)

    state_size = (240, 256, 3)  # Typical NES screen resolution
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    # Collect human data
    num_human_games = 1  # Number of human-played games for pre-training
    human_data = collect_human_data(env, num_human_games)

    # Pre-train the agent with human data
    agent.pretrain(human_data, batch_size=32)

    # Train the agent with reinforcement learning
    num_episodes = 10
    max_steps_per_episode = 1000000
    batch_size = 32
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01

    rewards = train_dqn(agent, env, num_episodes, max_steps_per_episode, batch_size, gamma, epsilon, epsilon_decay, min_epsilon)
    agent.save('tetrisAISheev.pth')

    env.close()

if __name__ == "__main__":
    main()
