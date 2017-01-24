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

    def __init__(self, K, episodes, means=None, var=None):
        """TODO: to be defined1. """
        self.K = K
        self.means = np.random.randn(K,)*1 if means is None else means
        self.var = np.ones(K,)*1 if means is None else means
        self.episodes = episodes
        # self.Q = np.zeros(K,)
        # self.N = np.zeros(K,)
        self.strategy = None


    def pull(self, a):
        """Pull arm k out of K and return observed reward

        :k: arm to pull
        :returns: observed reward

        """
        r =  np.random.randn()*self.var[a]+self.means[a]
        self.update_arms()
        return r

    def update_arms(self):
        """
        Moves the means of the arms according to a brownian motion
        """
        self.means += np.random.standard_normal(self.means.shape)*.0


    def run_round(self, n_round=1):
        M_reward = np.zeros(self.episodes,) #n x mean
        S_reward = np.zeros(self.episodes,) #n x var
        for i_round in range(1, n_round+1):
            rewards = np.zeros(self.episodes,)
            for i in range(self.episodes):

                # if decrease_eps:
                #     eps = eps/np.sqrt(i+1)

                # Select action
                a = self.strategy.action()

                r = bandit.pull(a)

                self.strategy.observe(a, r)
                rewards[i] = r
            x = np.cumsum(rewards)/np.arange(1, episodes+1)#, rewards
            S_reward += ((i_round)*x - M_reward)/(i_round*(i_round+1))
            M_reward += x
        return M_reward/n_round, np.sqrt(S_reward/n_round)


class UCB:

    """Class for the UCB algorithm"""

    def __init__(self, bandit, c, alpha):
        """TODO: to be defined1. """
        self.Q = np.zeros(bandit.K,)
        self.N = np.zeros(bandit.K,)
        self.c = c
        self.alpha = alpha

    def action(self):
        invN = np.zeros_like(self.N)
        invN[self.N>0] = self.N[self.N>0]
        invN[self.N==0] = np.inf
        sumN = self.N.sum() if self.N.sum()>0 else 1
        a = np.argmax(self.Q + self.c*np.sqrt(np.log(sumN)*invN))
        return a

    def observe(self, a, r):
        self.Q[a] = self.Q[a] + self.alpha*(r - self.Q[a])
        self.N[a] += 1

class eps_greedy_value_table:

    """Class for the epsilon-greedy policy using tabular values"""

    def __init__(self, bandit, eps, alpha):
        self.Q = np.zeros(bandit.K,)
        self.eps = eps
        self.alpha = alpha

    def action(self):
        if self.eps > np.random.rand():
            a = np.random.randint(0, bandit.K)
        else:
            a = np.argmax(self.Q)
        return a

    def observe(self, a, r):
        self.Q[a] = self.Q[a] + self.alpha*(r - self.Q[a])



K, episodes = 3, 20000
n_round = 10
means = np.random.rand(K,)*9
means[K-1] = 10
bandit = Bandit(K, episodes, means=means)
vec = np.arange(episodes)+1

# Strategy 1
strat = UCB(bandit, 1, .1)
bandit.strategy = strat

meanR, stdR = bandit.run_round(n_round)
plt.plot(vec, meanR, label=r'UCB')
plt.fill_between(vec, meanR-stdR, meanR+stdR, alpha = 0.5)

# Strategy 2
bandit2 = Bandit(K, episodes, means=means)
strat2 = eps_greedy_value_table(bandit2, .1, .8)
#strat2 = UCB(bandit2, 1, .8)
bandit.strategy = strat2

meanR, stdR = bandit.run_round(n_round)
plt.plot(vec, meanR, label=r'$\epsilon$-greedy, $\epsilon=0.1$')
plt.fill_between(vec, meanR-stdR, meanR+stdR, alpha = 0.5)

# Best reward
best_reward = np.max(bandit.means)
plt.plot(vec, np.ones(episodes,)*best_reward, label='Best expected reward')

plt.legend()
plt.xlabel('Episodes $(t)$')
plt.ylabel('$\mathbb{E} [R_t ]/t$')
plt.title('K-armed bandits $(K={})$'.format(K))
plt.show()
