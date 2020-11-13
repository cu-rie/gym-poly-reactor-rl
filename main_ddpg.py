import matplotlib.pyplot as plt
import numpy as np
import torch
import time

from gym_poly_reactor.envs.poly_reactor_env import PolyReactor
from gym_poly_reactor.agents.ddpg_agent import DDPGAgent

if __name__ == '__main__':

    # Action & State list
    # action_list = ["m_DOT_F", "T_IN_M", "T_IN_AWT"]
    # state_list = ["m_w", "m_a", "m_p", "T_R", "T_S", "T_M", "T_EK", "T_AWT", "T_adiab", "m_acc"]
    # init_state = np.array([10000, 853, 26.5, 363.15, 363.15, 363.15, 308.15, 308.15, 378.04682, 300])

    env = PolyReactor()

    # state space
    state_dim = env.observation_space.shape[0]

    # Action space
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high

    agent = DDPGAgent(state_dim=state_dim, action_dim=action_dim, action_low=action_low, action_high=action_high)

    state = env.reset()

    reward_trajectory = []

    critic_loss_traj = []
    actor_loss_traj = []

    len_episodes = []

    # Hyperparameters
    epsilon_min = 0
    num_ep = 500
    batch_size = 50

    agent.batch_size = batch_size
    agent.ou_noise.epsilon_min = epsilon_min


    moving_avg_rwd = [0]

    for i in range(num_ep):
        state = env.reset()

        sum_reward = 0
        t = 0

        while True:

            state_tensor = torch.Tensor(state).reshape(1, -1)
            nn_action, action, epsilon = agent.get_action(state_tensor, t)
            next_state, reward, done, info = env.step(action.reshape(-1, 1))

            agent.save_transition((state_tensor, nn_action, reward, next_state, done))

            state = next_state
            t += 1

            # save step result
            sum_reward += reward[0]

            if agent.train_start():
                critic_loss, actor_loss = agent.fit()

            if done:
                agent.ou_noise.reset()

                # save results
                reward_trajectory.append(sum_reward)
                moving_avg = moving_avg_rwd[-1] * 0.8 + sum_reward * 0.2
                moving_avg_rwd.append(moving_avg)
                if agent.train_start():
                    critic_loss_traj.append(critic_loss)
                    actor_loss_traj.append(actor_loss)

                if sum_reward > 20000:
                    len_episodes.append(t)

                print("EP:%d, RWD:%.3f, EPS:%.3f" % (i, sum_reward, epsilon))
                break

    # plotting state
    state_dim = env.observation_space.shape[0]

    fig, ax = plt.subplots()
    ax.plot(critic_loss_traj)
    plt.savefig('critic_loss.png')

    fig, ax = plt.subplots()
    ax.plot(actor_loss_traj)
    plt.savefig('actor_loss.png')

    fig, ax = plt.subplots()
    ax.plot(reward_trajectory, label='reward')
    ax.plot(moving_avg_rwd[1:], label='moving average reward')
    ax.legend()
    plt.savefig('reward.png')

    fig, ax = plt.subplots()
    ax.plot(len_episodes)
    plt.savefig('Episode length of WELLDONE.png')
