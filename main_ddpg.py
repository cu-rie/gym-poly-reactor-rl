import matplotlib.pyplot as plt
import numpy as np
import torch

from gym_poly_reactor.envs.poly_reactor_env import PolyReactor
from gym_poly_reactor.agents.ddpg_agent import DDPGAgent

if __name__ == '__main__':
    # Action & State list
    action_list = ["m_DOT_F", "T_IN_M", "T_IN_AWT"]
    state_list = ["m_w", "m_a", "m_p", "T_R", "T_S", "T_M", "T_EK", "T_AWT", "T_adiab", "m_acc"]
    init_state = np.array([10000, 853, 26.5, 363.15, 363.15, 363.15, 308.15, 308.15, 378.04682, 300])

    env = PolyReactor()

    # state space
    state_dim = env.observation_space.shape[0]

    # Action space
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high

    agent = DDPGAgent(state_dim=10, action_dim=action_dim, action_low=action_low, action_high=action_high)

    state = env.reset()

    reward_trajectory = []

    critic_loss_traj = []
    actor_loss_traj = []

    len_episodes = []
    step = 0

    # while True:
    agent.ou_noise.reset()

    # for i in range(1000):
    #     state = env.reset()
    #
    #     sum_reward = 0
    #     state_trajectory = [init_state]
    #     action_trajectory = []
    #     t = 0
    #
    #     while True:
    #
    #         state_tensor = torch.Tensor(state).reshape(1, -1)
    #         nn_action, action, epsilon = agent.get_action(state_tensor, step)
    #         # action_numpy = action.numpy()
    #         next_state, reward, done, info = env.step(action.reshape(-1, 1))
    #
    #         unnormalized_state = info['unnormalized_state']
    #         state_trajectory.append(unnormalized_state.reshape(-1))
    #         action_trajectory.append(action.reshape(-1))
    #
    #         agent.save_transition((state_tensor, nn_action, reward, next_state, done))
    #
    #         sum_reward += reward[0]
    #         state = next_state
    #         t += 1
    #         step += 1
    #
    #         if agent.train_start():
    #             critic_loss, actor_loss = agent.fit()
    #             critic_loss_traj.append(critic_loss)
    #             actor_loss_traj.append(actor_loss)
    #
    #         if done:
    #             reward_trajectory.append(sum_reward)
    #             len_episodes.append(t)
    #             print(i, "%.3f, %.3f, %.3f" % (sum_reward, epsilon, reward))
    #             break
    #
    # # plotting state
    # state_dim = env.observation_space.shape[0]
    #
    # full_trajectory = np.array(state_trajectory)
    # # fig, axs = plt.subplots(state_dim, 1, figsize=(5, 10), sharex=True)
    # fig, axs = plt.subplots(state_dim, 1, figsize=(5, 20), sharex=True)
    #
    # for i, ax in enumerate(axs):
    #     ax.plot(full_trajectory[:, i])
    #     ax.set_title(state_list[i])
    #
    # plt.savefig('state.png')
    #
    # # plotting action
    # action_trajectory = np.array(action_trajectory)
    # # action_trajectory = torch.stack(action_trajectory).squeeze().detach().numpy()
    #
    # fig, axs = plt.subplots(action_dim, 1, figsize=(5, 10), sharex=True)
    #
    # for i, ax in enumerate(axs):
    #     ax.plot(action_trajectory[:, i])
    #     ax.set_title(action_list[i])
    #
    # plt.savefig('action.png')
    #
    # fig, ax = plt.subplots()
    # ax.plot(critic_loss_traj)
    # plt.savefig('critic_loss.png')
    #
    # fig, ax = plt.subplots()
    # ax.plot(actor_loss_traj)
    # plt.savefig('actor_loss.png')
    #
    # fig, ax = plt.subplots()
    # ax.plot(reward_trajectory)
    # plt.savefig('reward.png')
    #
    # fig, ax = plt.subplots()
    # ax.plot(len_episodes)
    # plt.savefig('len_episode.png')
