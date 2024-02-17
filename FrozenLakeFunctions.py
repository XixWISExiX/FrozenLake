import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

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

