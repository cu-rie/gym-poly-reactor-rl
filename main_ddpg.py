import matplotlib.pyplot as plt
import numpy as np
import torch

from gym_poly_reactor.envs.poly_reactor_env import PolyReactor
from gym_poly_reactor.agents.ddpg_agent import DDPGAgent

if __name__ == '__main__':
    # Action space
    abs_zero = 273.15
    m_DOT_F_MIN, m_DOT_F_MAX = 0, 30000  # [kgh^-1]
    T_IN_M_MIN, T_IN_M_MAX = 60 + abs_zero, 100 + abs_zero  # [Kelvin]
    T_IN_AWT_MIN, T_IN_AWT_MAX = 60 + abs_zero, 100 + abs_zero  # [Kelvin]

    action_min = [m_DOT_F_MIN, T_IN_M_MIN, T_IN_AWT_MIN]
    action_max = [m_DOT_F_MAX, T_IN_M_MAX, T_IN_AWT_MAX]

    env = PolyReactor()

    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]

    agent = DDPGAgent(state_dim=10, action_dim=len(action_min), action_min=action_min, action_max=action_max)

    state = env.reset()

    reward_trajectory = []

    critic_loss_traj = []
    actor_loss_traj = []

    # while True:
    for _ in range(100):
        state = env.reset()
        sum_reward = 0
        state_trajectory = []
        action_trajectory = []
        while True:

            state_tensor = torch.Tensor(state).reshape(1, -1)
            nn_action, action = agent.get_action(state_tensor)
            action_numpy = action.numpy()
            next_state, reward, done, _ = env.step(action_numpy.reshape(-1, 1))

            state_trajectory.append(state.reshape(-1))
            action_trajectory.append(action.reshape(-1))

            agent.save_transition((state_tensor, nn_action, reward, next_state, done))

            sum_reward += reward
            state = next_state

            if agent.train_start():
                critic_loss, actor_loss = agent.fit()

                critic_loss_traj.append(critic_loss)
                actor_loss_traj.append(actor_loss)

            if done:
                reward_trajectory.append(sum_reward)
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

    action_trajectory = torch.stack(action_trajectory).squeeze().detach().numpy()

    fig, axs = plt.subplots(action_dim, 1, figsize=(5, 10))

    for i, ax in enumerate(axs):
        ax.plot(action_trajectory[:, i])

    plt.savefig('action.png')

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(critic_loss_traj)
    plt.savefig('critic_loss.png')

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(actor_loss_traj)
    plt.savefig('actor_loss.png')

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(reward_trajectory)
    plt.savefig('reward.png')
