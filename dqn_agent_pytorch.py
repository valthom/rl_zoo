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
import ipdb
# import matplotlib.pyplot as plt
# import seaborn
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
np.random.seed(0)

#2 tricks
# experience replay
# fixed Q targets (in the max we use old parameters)


class Agent(object):

    """Class for a reinforcement learning agent"""

    def __init__(self, env, gamma=.96, eps=.5, n_hidden=64, max_mem=50000,
            batch_size=64, episode_maxlength=600):
        """TODO: to be defined1. """
        self.env = env
        self.gamma = gamma
        self.n_act = env.action_space.n
        if isinstance(env.observation_space, gym.spaces.discrete.Discrete):
            self.n_obs = env.observation_space.n
        else:
            self.n_obs = env.observation_space.shape[0]

        self.net = self.make_model(n_hidden)
        self.model = None
        self.memory = []
        self.eps = eps
        self.counter = 0
        self.batch_size = batch_size
        self.episode_maxlength = episode_maxlength
        self.memory_maxsize = max_mem
        self.criterion = nn.MSELoss()
        self.opt = optim.Adam(self.net.parameters(), lr=3e-4)

    def act(self, s):
        """Choose action following a eps-greedy policy
        :returns: action to take

        """
        if np.random.rand()<self.eps:
            a = env.action_space.sample()
        else:
            input = Variable(torch.from_numpy(s.reshape(-1,\
                self.n_obs))).float()
            a = np.argmax(self.net(input).data.cpu().numpy())
        return a

    def run_episode(self, render=False):
        """Run an episode on the environment

        :returns: TODO

        """
        s = env.reset()
        tot_reward = 0
        episode_length = 0
        for i in range(self.episode_maxlength):
            if render:
                env.render()
            a = self.act(s)
            ss, r, done, _ = env.step(a)
            #env.render()

            # Add last step into memory
            if not done:
                self.add_mem((s, r, a, ss))
            else:
                self.add_mem((s, r, a, np.zeros((self.n_obs,))))
                break


            # Update new state
            s = ss
            tot_reward += r
            episode_length += 1

            self.replay()
        self.counter += 1
        self.eps = (1-0.01)*np.exp(-5e-3*self.counter)+0.01
        return tot_reward, episode_length

    def train(self, x, y, n_epoch=1):
        """Training routine

        :x: states saved
        :y: Bellman expectation with fixed targets

        """
        #self.net.fit(x, y, batch_size=self.batch_size, nb_epoch=n_epoch, verbose=0)
        loss = self.criterion(x, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def make_model(self, n_hidden):
        net = nn.Sequential(nn.Linear(self.n_obs, n_hidden), nn.ReLU(),\
                nn.Linear(n_hidden,  self.n_act))
        return net

    def add_mem(self, x):
        """Add element to memory

        :x: 4-uplet (s, a, r, s')

        """
        self.memory.append(x)
        if len(self.memory) > self.memory_maxsize:
            self.memory = self.memory[-self.memory_maxsize:]

    def make_targets(self, samples):
        """Function to compute the fixed targets for Q-network

        :samples: 4-uplets of (s, a, r, s')
        :returns: input/output pair for training

        """
        #s, a, r, s' = sample
        outputs = []
        inputs = np.vstack((sample_i[0] for sample_i in samples))
        new_states = np.vstack((sample_i[3] for sample_i in samples))
        Q = self.net(Variable(torch.from_numpy(inputs)).float())
        Q_s2 = self.net(Variable(torch.from_numpy(new_states)).float())

        # can do better without a loop wit filters on lists
        for i, (s, r, a, ss) in enumerate(samples):
            t = Q[i].clone()
            if np.allclose(ss, 0): # terminal state
                t[a] = r
            else:
                t[a] = r + self.gamma*torch.max(Q_s2[i,:])
            outputs.append(t)
        return inputs, torch.stack(outputs).detach()

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
        indices[0] = len(self.memory)-1
        batch_mem = [self.memory[i] for i in indices]
        inputs, outputs = self.make_targets(batch_mem)
        inputs = Variable(torch.from_numpy(inputs)).float()
        pred = self.net(inputs)
        self.train(pred, outputs)


env = gym.make('CartPole-v0')

dqn_ag = Agent(env)
n_episodes = 1000
rewards, epis_len = [], []
best_r = - np.inf
for i in tqdm(range(n_episodes)):
    r, l = dqn_ag.run_episode()
    print(r)
    rewards.append(r)
    epis_len.append(l)
    if r > best_r:
        print(r)
        best_r = r

# plt.figure()
# plt.subplot(211)
# plt.xlabel('episode')
# plt.ylabel('reward')
# vec = np.arange(len(rewards))
# plt.plot(vec, np.cumsum(rewards)/(1+vec), label='cum rewards')
# plt.plot(vec, rewards, label='instant rewards')
# plt.legend()
#
# plt.subplot(212)
# plt.xlabel('episode')
# plt.ylabel('episode_length')
# plt.plot(epis_len, label='length')
#
# plt.show()
