import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

def visualize_policy(policy, name, size, env_array, env_seed):
    env_array = findPolicyPath(env_array, policy, env_seed, size)
    env_array[size-1][size-1] = 2
    policy_grid = policy.astype(int).reshape(size, size)  # Convert policy values to integers and reshape into a 12x12 grid
    colors = ['black', 'white', 'green', 'red', 'cyan'] # Position corallates with values [-1, 0, 1, 2]
    cmap = plt.cm.colors.ListedColormap(colors)
    plt.imshow(env_array, cmap=cmap, interpolation='nearest')  # Display a blank grid
    
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

    env_array[size-1][size-1] = 2
    plt.title(name)
    plt.xlabel('Column')
    plt.ylabel('Row')
    # Add lines between grid points to separate squares
    for i in range(1, size):
        plt.axhline(y=i - 0.5, color='black', linewidth=1)
        plt.axvline(x=i - 0.5, color='black', linewidth=1)
    labels = {0: 'Normal State', 1: 'Start State', 2: 'End State', -1: 'Hole State', 3: 'Path of Agent'}
    legend_patches = [mpatches.Patch(color=colors[i+1], label=labels[i]) for i in range(-1,4)]
    plt.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.5, 1))
    plt.savefig('images/frozen_lake_{name}_policy.png'.format(name=name))
    plt.show()
    plt.close()

def visualize_QTable(QTable, name, size, env_array, env_seed):
    env_array = findQTablePath(env_array, QTable, env_seed, size)
    env_array[size-1][size-1] = 2
    policy_grid = np.zeros((size, size))
    colors = ['black', 'white', 'green', 'red', 'cyan'] # Position corallates with values [-1, 0, 1, 2]
    cmap = plt.cm.colors.ListedColormap(colors)
    plt.imshow(env_array, cmap=cmap, interpolation='nearest')  # Display a blank grid
    
    # Add annotations for the arrows
    for i in range(policy_grid.shape[0]):
        for j in range(policy_grid.shape[1]):
            action = np.argmax(QTable[i*size+j,:])
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

    plt.title(name)
    plt.xlabel('Column')
    plt.ylabel('Row')
    # Add lines between grid points to separate squares
    for i in range(1, size):
        plt.axhline(y=i - 0.5, color='black', linewidth=1)
        plt.axvline(x=i - 0.5, color='black', linewidth=1)
    labels = {0: 'Normal State', 1: 'Start State', 2: 'End State', -1: 'Hole State', 3: 'Path of Agent'}
    legend_patches = [mpatches.Patch(color=colors[i+1], label=labels[i]) for i in range(-1,4)]
    plt.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.5, 1))
    plt.savefig('images/frozen_lake_{name}_policy.png'.format(name=name))
    plt.show()
    plt.close()

def seeEnv(env_seed, size):
    plt_array = getBoard(env_seed, size)
           
    # Show the environment in a plot
    colors = ['black', 'white', 'green', 'red'] # Position corallates with values [-1, 0, 1, 2]
    cmap = plt.cm.colors.ListedColormap(colors)
    plt.imshow(plt_array, cmap=cmap, interpolation='nearest')  # Display a blank grid
    for i in range(1, size):
        plt.axhline(y=i - 0.5, color='black', linewidth=1)
        plt.axvline(x=i - 0.5, color='black', linewidth=1)
    plt.title("Frozen Lake Environment")
    plt.xlabel('Column')
    plt.ylabel('Row')
    labels = {0: 'Normal State', 1: 'Start State', 2: 'End State', -1: 'Hole State'}
    legend_patches = [mpatches.Patch(color=colors[i+1], label=labels[i]) for i in range(-1,3)]
    plt.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.5, 1))
    plt.savefig('images/frozen_lake_environment.png')
    plt.show()
    plt.close()
    return plt_array

def getBoard(env_seed, size):
    env = gym.make("FrozenLake-v1", desc=env_seed, map_name="12x12", is_slippery=False, render_mode='ansi')
    env.reset()

    # Make the environment array to plot out
    plt_array = np.zeros((size, size))
    hole_states = []
    for s in range(env.observation_space.n):
        # Check if the state is a hole state
        if env.desc.flatten()[s].decode('utf-8') == "H":
            hole_states.append(s)
    plt_array[0, 0] = 1  # Start state
    plt_array[size-1, size-1] = 2  # Goal state
    for s in hole_states:
        row = s // size
        col = s % size
        plt_array[row, col] = -1  # Mark hole state
    env.close()
    return plt_array
     

def findQTablePath(env_array, model, env_seed, size):
    env = gym.make("FrozenLake-v1", desc=env_seed, map_name="8x8", is_slippery=False, render_mode=None)
    state = env.reset()[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
    terminated = False      # True when fall in hole or reached goal
    truncated = False       # True when actions > 200
    while(not terminated and not truncated):
        action = np.argmax(model[state,:])
        new_state,reward,terminated,truncated,_ = env.step(action)
        env_array[new_state // size][new_state % size] = 3
        state = new_state
    env.close()
    return env_array


def findPolicyPath(env_array, model, env_seed, size):
    env = gym.make("FrozenLake-v1", desc=env_seed, map_name="8x8", is_slippery=False, render_mode=None)
    done = False
    state = env.reset()[0]
    while not done:
        action = model[state]
        state, _, done, _, _= env.step(action)
        env_array[state // size][state % size] = 3
    env.close()
    return env_array