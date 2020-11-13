import matplotlib.pyplot as plt
import numpy as np

from gym_poly_reactor.envs.poly_reactor_env import PolyReactor

if __name__ == '__main__':

    env = PolyReactor()
    _ = env.reset()
    len_episode = 10000

    state_trajectory = []
    action_trajectory = []

    step = 0

    # while True:
    for _ in range(len_episode):
        action = env.action_space.sample()
        next_state, _, done, _ = env.step(action)

        state_trajectory.append(next_state)
        action_trajectory.append(action)

        step += 1

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
