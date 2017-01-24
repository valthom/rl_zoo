#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
    rl_zoo.simple_bandits
    ~~~~~~~~~~~~~~~~~~~~~

    Simple k-armed bandits examples

"""

import numpy as np
import matplotlib.pyplot as plt
import ipdb


# more like class bandit and function pull
class Bandit:

    """Bandit class to generate the bandit problem and the pull arm function"""

    def __init__(self, K, episodes):
        """TODO: to be defined1. """
        self.K = K
        self.means = np.random.randn(K,)*5
        self.var = np.random.rand(K,)*1
        self.episodes = episodes
        self.Q = np.zeros(K,)
        self.N = np.zeros(K,)


    def pull(self, k):
        """Pull arm k out of K and return observed reward

        :k: arm to pull
        :returns: observed reward

        """
        r =  np.random.randn()*self.var[k]+self.means[k]
        self.update_arms()
        return r

    def update_arms(self):
        """
        Moves the means of the arms according to a brownian motion
        """
        self.means += np.random.standard_normal(self.means.shape)*.0

    def e_greedy_Q_learning(self, eps, Q):
        if eps > np.random.rand():
            a = np.random.randint(0, K)
        else:
            a = np.argmax(Q)
        return a

    def ucb(self, Q, N, i):
        a = np.argmax(Q + np.sqrt(np.log(i)/N))
        return a

    def run_round(self, eps, alpha, decrease_eps = False, init_q = 0):
        Q = np.ones(K,)*init_q
        N = np.zeros(K,)
        rewards = np.zeros(self.episodes,)
        for i in range(self.episodes):
            if decrease_eps:
                eps = eps/np.sqrt(i+1)
            # Select action
            a = self.e_greedy_Q_learning(eps, Q)
            # a = self.ucb(Q, N, i)
            r = bandit.pull(a)
            #ipdb.set_trace()
            Q[a] = Q[a] + alpha*(r - Q[a])
            N[a] += 1
            rewards[i] = r
        return np.cumsum(rewards)/np.arange(1, episodes+1)#, rewards

# class eps_greedy_value_table:
#
# class eps_greedy_value_table(object):
#
#     """Class for the epsilon-greedy policy using tabular values"""
#
#     def __init__(self, bandit):
#         """TODO: to be defined1. """
        



K, episodes = 10, 20000
bandit = Bandit(K, episodes)


best_reward = np.max(bandit.means)
#reward_per_episode = np.cumsum(rewards)/np.arange(1, episodes+1)

alpha = .5
for eps in [.1, 1e-3]:
    plt.semilogx(np.arange(episodes)+1, bandit.run_round(eps, alpha), label=r'$\epsilon = {}$'.format(eps))
plt.semilogx(np.arange(episodes)+1, np.ones(episodes,)*best_reward, label='Best expected reward')
plt.semilogx(np.arange(episodes)+1, bandit.run_round(50, alpha, decrease_eps=True),
        label=r'$\epsilon = \min(1, \frac{50}{t})$')
plt.semilogx(np.arange(episodes)+1, bandit.run_round(1e-3, alpha, init_q = 50), label=r'$\epsilon = 0.001, Q_0 = 50$')
plt.legend()
plt.show()
