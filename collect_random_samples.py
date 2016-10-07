""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import gym
import pickle as pickle
import numpy as np

import tensorflow as tf
import sys
print(sys.executable)

import matplotlib.pyplot as plt

# hyperparameters
render = False
total_episodes = 3

gamma = 0.7
pickle_frequency = 2
file_header = 'samples_episodes/samples_random_policy_small.pickle'

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    # I = I[:, :, 0] #only get the red color
    return I


def discount_rewards(r):
    """ take 1D float/int array of rewards and compute discounted reward """
    discounted_r = np.zeros(r.shape, dtype=float)
    running_add = 0.0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

env = gym.make("Pong-v0")
observation = env.reset()
states, actions, rewards, returns = [], [], [], []
s_epi, a_epi, rew_epi, ret_epi = [], [], [], []
episode_number = 0

time_step = 0
total_games = 0

def show_this_dump():
    game_length = 0.0
    for i in range(len(returns)):
        game_length += len(returns[i])
    print('average game length is %f ' % (game_length / len(returns)))

    num_successes = 0
    num_misses = 0
    for epi in range(len(returns)):
        for rew in rewards[epi]:
            if np.abs(rew - 1.0) < 1e-5:
                num_successes += 1
            elif np.abs(rew - (-1.0)) < 1e-5:
                num_misses += 1
        plt.plot(rewards[epi])
    assert (total_games == num_successes + num_misses)
    print('winning %d games out of total %d games in total %d episodes' %
          (num_successes, total_games, total_episodes))

    # plt.show()

while episode_number < total_episodes:
    if render: env.render()

    # preprocess the observation, set input to network to be difference image
    x = prepro(observation)

    if np.random.uniform() < 0.5:
        a = 2
    else:
        a = 3

    s_epi.append(x)  # observation
    a_epi.append(a)

    # step the environment and get new measurements
    observation, rew, done, info = env.step(a)

    rew_epi.append(rew)

    if rew == 0:
        print('episode %d#, step %d: reward is 0' % (episode_number, time_step))
    else:
        print('episode %d#, step %d: reward is ' % (episode_number, time_step) + ('-1' if rew == -1 else '1!!!!'))

    # a game is over
    if rew != 0:
        states.append(s_epi)
        actions.append(a_epi)
        rewards.append(rew_epi)
        returns.append(discount_rewards(np.array(rew_epi)))
        s_epi, a_epi, rew_epi, ret_epi = [], [], [], []
        total_games += 1

    # an episode is over
    if done:
        episode_number += 1
        time_step = 0
        observation = env.reset()
    else:
        time_step += 1

    # dump data every 2 episodes
    if done and episode_number > 0 and episode_number % pickle_frequency == 0:
        show_this_dump()
        total_games = 0
        samples = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'returns': returns
        }
        with open(file_header + str(int(episode_number/pickle_frequency)), 'wb') as handler:
            pickle.dump(samples, handler)

        states, actions, rewards, returns = [], [], [], []


assert(len(states) == len(actions))
assert(len(rewards) == len(states))
assert(len(returns) == len(states))




