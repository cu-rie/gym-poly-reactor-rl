import matplotlib.pyplot as plt
import numpy as np

from gym_poly_reactor.envs.poly_reactor_env import PolyReactor
from gym_poly_reactor.agents.random_agent import RandomAgent

if __name__ == '__main__':
    # Action space

    env = PolyReactor()

    state_dim = env.observation_space.shape[0]

    action_low = env.action_space.low
    action_high = env.action_space.high

    agent = RandomAgent(state_dim=state_dim, action_low=action_low, action_high=action_high)

    state = env.reset()

    state_trajectory = []
    action_trajectory = []

    # while True:
    for _ in range(10000):
        action = agent.get_action(state)
        next_state, _, done, _ = env.step(action)

        state_trajectory.append(next_state)
        action_trajectory.append(action)

        if done:
            break

    # plotting state
    state_dim = env.observation_space.shape[0]

    full_trajectory = np.array(state_trajectory)
    fig, axs = plt.subplots(state_dim, 1, figsize=(5, 10))
    # fig, axs = plt.subplots(state_dim, 1, figsize=(10, 25))

    for i, ax in enumerate(axs):
        ax.plot(full_trajectory[:, i])

    plt.savefig('state.png')

    # plotting action
    action_dim = env.action_space.shape[0]
    action_trajectory = np.array(action_trajectory)

    fig, axs = plt.subplots(action_dim, 1, figsize=(5, 10))

    for i, ax in enumerate(axs):
        ax.plot(action_trajectory[:, i])

    plt.savefig('action.png')
