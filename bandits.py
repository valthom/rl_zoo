#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    rl_zoo.simple_bandits
    ~~~~~~~~~~~~~~~~~~~~~

    Simple k-armed bandits examples
    3 algorithms implemented
        - UCB
        - epsilon-greedy with value table
        - bandit policy gradient

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
np.random.seed(3)

def softmax(z):
    zm = np.exp(z-z.max())
    return zm/zm.sum()

class Bandit:

    """Bandit class to generate the bandit problem and the pull arm function"""

    def __init__(self, K, means=None, var=None):
        """TODO: to be defined1. """
        self.K = K
        self.means = np.random.randn(K,)*5 if means is None else means
        self.var = np.ones(K,)*3 if means is None else means

    def pull(self, a):
        """Pull arm k out of K and return observed reward

        :k: arm to pull
        :returns: observed reward

        """
        r =  np.random.randn()*self.var[a]+self.means[a]
        #self.update_arms()
        return r

    def reinitialize(self):
        self.means = np.random.randn(K,)*5
        self.var = np.ones(K,)*3


    # def update_arms(self):
    #     """
    #     Moves the means of the arms according to a brownian motion
    #     """
    #     self.means += np.random.standard_normal(self.means.shape)*.0


class Strategy(object):

    """Virtual class for all strategies (ubc, eps-greedy, policy gradient..)"""

    def run_round(self, n_round=5, episodes=1000, plot=True):
        # we compute mean and variance recursively
        M_reward = np.zeros(episodes,) #n x mean
        S_reward = np.zeros(episodes,) #n x var
        for i_round in tqdm(range(1, n_round+1)):
            self.reinitialize()
            #self.bandit.reinitialize()
            rewards = np.zeros(episodes,)
            for i in range(episodes):
                # Select action
                a = self.action()
                # Get reward for that action
                r = self.bandit.pull(a)
                #r = 100*r/self.bandit.means.max()
                # Update strategy with given action/reward pair
                self.observe(a, r)

                rewards[i] = r
            x = np.cumsum(rewards)/np.arange(1, episodes+1)# Rt/t
            M_reward += x
            S_reward += ((i_round)*x - M_reward)**2/(i_round*(i_round+1))
        meanR, stdR = M_reward/n_round, np.sqrt(S_reward/n_round)
        if plot:
            self.plot(meanR, stdR)
        return meanR, stdR

    def plot(self, meanR, stdR):
        best_reward = bandit.means.max()
        vec = np.arange(episodes)
        meanR, stdR = 100*meanR/best_reward, 50*stdR/best_reward
        # Plot the average curve and fill 1 sigma around
        plt.plot(vec, meanR, label=self.label())
        plt.fill_between(vec, meanR-stdR, meanR+stdR, alpha = 0.25)

class UCB(Strategy):

    """Class for the UCB algorithm"""

    def __init__(self, bandit, c=None, alpha=None):
        self.bandit = bandit
        self.Q = np.ones(bandit.K,)*0
        self.N = np.zeros(bandit.K,)
        self.c = c if c is not None else 4
        self.alpha = alpha if alpha is not None else .1

    def action(self):
        if np.any(self.N==0):
            best_actions = np.argwhere(self.N==0)
            a = best_actions[np.random.randint(best_actions.size)]
        else:
            value = self.Q + self.c*np.sqrt(np.log(self.N.sum())/self.N)
            a = np.argmax(value)
        return a

    def observe(self, a, r):
        self.Q[a] = self.Q[a] + self.alpha*(r - self.Q[a])
        self.N[a] += 1

    def reinitialize(self):
        self.Q *= 0
        self.N *= 0

    def label(self):
        return r'UCB (c = {}, $\alpha = {}$)'.format(self.c, self.alpha)

class Eps_greedy_value_table(Strategy):

    """Class for the epsilon-greedy policy using tabular values"""

    def __init__(self, bandit, eps=1, alpha=.1):
        """init function for the epsilon greedy strategy

        :bandit: its associated bandit problem (contain pull_arm function)
        :eps: rate at which we select random actions
        :alpha: learning rate

        """
        self.Q = np.zeros(bandit.K,)
        self.eps = eps
        self.alpha = alpha
        self.bandit = bandit

    def action(self):
        """Return selected action"""
        # with probability epsilon we choose at random
        if self.eps > np.random.rand():
            a = np.random.randint(0, bandit.K)
        # otherwise we select our best action
        else:
            # --- to prevent from selecting same sequence of actions
            # each time when Q is 0 ---
            best_actions = np.argwhere(self.Q == np.max(self.Q))
            a = best_actions[np.random.randint(best_actions.size)]
        return a

    def observe(self, a, r):
        """ Update value function with new reward """
        self.Q[a] = self.Q[a] + self.alpha*(r - self.Q[a])

    def reinitialize(self):
        self.Q *= 0

    def label(self):
        return r'$\epsilon$-greedy ($\epsilon = {}, \alpha = {} $)'.format(self.eps, self.alpha)

class Policy_gradient(Strategy):

    """Policy based strategy"""

    def __init__(self, bandit, alpha=0.1):
        """Init for policy based strat.

        :alpha: learning rate

        """
        self.alpha = alpha
        self.bandit = bandit
        self.H = np.ones(self.bandit.K,)*0
        self.proba = np.ones_like(self.H,)/self.H.size
        self.r_m = 0
        self.counter = 0

    def action(self):
        self.proba = softmax(self.H)
        a = np.argmax(np.random.multinomial(1, self.proba))
        return a

    def observe(self, a, r):
        self.r_m = self.counter/(1+self.counter)*self.r_m + 1/(1+self.counter)*r
        self.counter += 1
        self.H -= self.alpha*(r - self.r_m)*self.proba
        self.H[a] += self.alpha*(r - self.r_m)

    def reinitialize(self):
        self.H *= 0
        self.proba = np.ones_like(self.H,)/self.H.size
        self.counter = 0
        self.r_m = 0

    def label(self):
        return r'Policy gradient ($\alpha = {}$)'.format(self.alpha)



K, episodes = 10, 2500
n_round = 100
bandit = Bandit(K)


best_reward = np.max(bandit.means)
print('Best value:', best_reward)

# Strategy 1
eps = 0.05
e_gr = Eps_greedy_value_table(bandit, eps)
e_gr.run_round(n_round, episodes=episodes)

# Strategy 2
eps = 0.2
e_gr2 = Eps_greedy_value_table(bandit, eps)
e_gr2.run_round(n_round, episodes=episodes)

# Strategy 3
pg = Policy_gradient(bandit)
pg.run_round(n_round, episodes=episodes)

# Strategy 4
ucb = UCB(bandit)
ucb.run_round(n_round, episodes=episodes)


# Add title and axis labels to the plot
plt.legend(loc='best')
axes=plt.gca()
axes.set_ylim([-10, 110])
plt.xlabel('Episodes $(t)$')
plt.ylabel('$\mathbb{E} [R_t]$ (as % of best expected reward)')
plt.title('K-armed bandits $(K={})$'.format(K))
plt.show()
