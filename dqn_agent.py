#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    rl_zoo.dqn_agent
    ~~~~~~~~~~~~~~~~

    Code for a Deep-Q network trained on some OpenAI's gym environments

    :copyright: (c) 2017 by Valentin Thomas.
"""

import gym
from tqdm import tqdm
import numpy as np
from keras import models
from keras.layers import Dense
from keras.optimizers import *
import ipdb

#2 tricks
# experience replay
# fixed Q targets (in the max we use old parameters)


class Agent(object):

    """Class for a reinforcement learning agent"""

    def __init__(self, env, gamma=.99, eps=.1, n_hidden=4, max_mem=1000,
            batch_size=64, episode_maxlength=250):
        """TODO: to be defined1. """
        self.env = env
        self.gamma = gamma
        self.n_act = env.action_space.n
        self.n_obs = env.observation_space.shape[0]
        self.net = self.make_model(n_hidden)
        self.model = None
        self.memory = []
        self.eps = eps
        self.batch_size = batch_size
        self.episode_maxlength = episode_maxlength
        self.memory_maxsize = max_mem

    def act(self, s):
        """Choose action following a eps-greedy policy
        :returns: action to take

        """
        if np.random.rand()<self.eps:
            a = env.action_space.sample()
        else:
            a = np.argmax(self.net.predict(s.reshape(-1, self.n_obs)))
        return a

    def run_episode(self):
        """Run an episode on the environment

        :returns: TODO

        """
        s = env.reset()
        tot_reward = 0
        episode_length = 0
        for i in range(self.episode_maxlength):
            a = self.act(s)
            ss, r, done, _ = env.step(a)

            # Add last step into memory
            if not done:
                self.add_mem((s, r, a, ss))
            else:
                self.add_mem((s, r, a, np.zeros((self.n_obs,))))
                break

            self.replay()

            # Update new state
            s = ss
            tot_reward += 1
            episode_length += 1
        return tot_reward, episode_length

    def train(self, x, y, batch_size=64, n_epoch=3):
        """Training routine

        :x: TODO
        :y: TODO

        """
        self.net.fit(x, y, batch_size=batch_size, nb_epoch=n_epoch, verbose=0)

    def make_model(self, n_hidden):
        net = models.Sequential()
        net.add(Dense(output_dim=n_hidden, activation='relu',
            input_dim=self.n_obs))
        net.add(Dense(output_dim=self.n_act, activation='linear'))
        opt = RMSprop(lr=5e-3)
        net.compile(loss='mse', optimizer=opt)
        return net

    def add_mem(self, x):
        """Add element to memory

        :x: TODO
        :returns: TODO

        """
        self.memory.append(x)
        if len(self.memory) > self.memory_maxsize:
            self.memory = self.memory[-self.memory_maxsize:]

    def make_targets(self, samples):
        """Function to compute the fixed targets for Q-network

        :sample: s, a, r, s'
        :returns: input/output pair for training

        """
        #s, a, r, ss = sample
        inputs = np.zeros((0,self.n_obs))
        outputs = []
        inputs = np.vstack((sample_i[0] for sample_i in samples))
        new_states = np.vstack((sample_i[3] for sample_i in samples))
        Q = self.net.predict(inputs)
        Q_s2 = self.net.predict(new_states)

        # can do better without a loop wit filters on lists
        for i, (s, r, a, ss) in enumerate(samples):
            t = Q[i]
            if np.allclose(ss, 0): # terminal state
                #ipdb.set_trace()
                t[a] = r
            else:
                t[a] = r + self.gamma*np.max(Q_s2[i,:])
            outputs.append(t)
        return inputs, np.array(outputs)

    def replay(self):
        """
        Function that samples (s, a, r, s') tuples from the memory and use them
        to train the function approximator
        TODO: clip rewards
        info:
        target = r + gamma*max_a' Q(s', a', w-) <- fixed target: w- = last w
        input = s
        loss L(w) = 1/2 E_x[(target - net(s; w))**2]
        """
        replay_size = min(len(self.memory), self.batch_size)
        indices = np.random.randint(0, len(self.memory), replay_size)
        batch_mem = [self.memory[i] for i in indices]
        inputs, outputs = self.make_targets(batch_mem)
        self.train(inputs, outputs)


import matplotlib.pyplot as plt
import seaborn

env = gym.make('CartPole-v0')
dqn_ag = Agent(env)
n_episodes = 500
rewards, epis_len = [], []
for i in tqdm(range(n_episodes)):
    r, l = dqn_ag.run_episode()
    rewards.append(r)
    epis_len.append(l)

plt.figure()
plt.subplot(211)
plt.xlabel('episode')
plt.ylabel('reward')
plt.plot(rewards, label='rewards')

plt.subplot(212)
plt.xlabel('episode')
plt.ylabel('episode_length')
plt.plot(rewards, label='length')

plt.legend()
plt.show()
