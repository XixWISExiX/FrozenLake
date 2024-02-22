import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
# import pyautogui

def seeModel(model, env_seed):
    env = gym.make("FrozenLake-v1", desc=env_seed, map_name="8x8", is_slippery=False, render_mode="human")
    state = env.reset()[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
    terminated = False      # True when fall in hole or reached goal
    truncated = False       # True when actions > 200
    while(not terminated and not truncated):
        action = np.argmax(model[state,:])
        new_state,reward,terminated,truncated,_ = env.step(action)
        state = new_state
    env.close()

def testModel(model, model_name, env_seed, tests, line_style=None):
    rewards_per_episode = np.zeros(tests)
    env = gym.make("FrozenLake-v1", desc=env_seed, map_name="8x8", is_slippery=False, render_mode=None)
    for i in range(tests):
        state = env.reset()[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
        terminated = False      # True when fall in hole or reached goal
        truncated = False       # True when actions > 200
        while(not terminated and not truncated):
            action = np.argmax(model[state,:])
            new_state,reward,terminated,truncated,_ = env.step(action)
            state = new_state

            if reward == 1:
                reward = 0
            elif terminated:
                reward = -99
            else:
                reward = -1

        if reward == 0:
            rewards_per_episode[i] = 1
    env.close()
    plt.plot(rewards_per_episode, label=model_name, linestyle=line_style)
    plt.legend()

def seeModelState(model, env_seed):
    env = gym.make("FrozenLake-v1", desc=env_seed, map_name="8x8", is_slippery=False, render_mode="human")
    done = False
    state = env.reset()[0]
    while not done:
        action = model[state]
        state, _, done, _, _= env.step(action)
    env.close()

def visualize_policy(policy, size):
    policy_grid = policy.astype(int).reshape(size, size)  # Convert policy values to integers and reshape into a 12x12 grid
    plt.imshow(np.zeros((size, size)), cmap='Greys', interpolation='nearest')  # Display a blank grid
    
    # Add annotations for the arrows
    for i in range(policy_grid.shape[0]):
        for j in range(policy_grid.shape[1]):
            action = policy_grid[i, j]
            if action == 1:
                arrow = '\u2193'  # Down arrow
            elif action == 2:
                arrow = '\u2192'  # Right arrow
            elif action == 3:
                arrow = '\u2191'  # Up arrow
            elif action == 0:
                arrow = '\u2190'  # Left arrow
            else:
                arrow = ''  # No arrow for other values
            plt.text(j, i, arrow, ha='center', va='center', color='black', fontsize=20)

    plt.title('Policy Visualization')
    plt.xlabel('Column')
    plt.ylabel('Row')
    # Add lines between grid points to separate squares
    for i in range(1, size):
        plt.axhline(y=i - 0.5, color='black', linewidth=1)
        plt.axvline(x=i - 0.5, color='black', linewidth=1)
    plt.savefig('images/frozen_lake_DynamicProgramming.png')
    plt.show()

def seeEnv(env_seed):
    env = gym.make("FrozenLake-v1", desc=env_seed, map_name="12x12", is_slippery=False, render_mode='ansi')
    env.reset()
    print(env.render())
    env.close()