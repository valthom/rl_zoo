#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
import theano
import theano.tensor as T



class Agent(object):

    """Class for a reinforcement learning agent"""

    def __init__(self, env):
        """TODO: to be defined1. """
        self.env = env
        # self.exploration = exploration
        # self.exploitation = exploitation
        # assert stuff here
        self.n_act = env.action_space.n
        self.n_obs = env.observation_space.shape[0]
        self.w = np.random.randn(self.n_obs, self.n_acts)
        self.memory = []
        self.eps = .05

    def act(self):
        """Choose action following a eps-greedy policy
        :returns: TODO

        """
        if np.random.rand()<self.eps:
            a = env.action_space.sample()
        else:
            a = 0
        return a

    def predict(self, s):
        return

    def train(self, batch):
        """Training routine

        :batch: TODO
        :returns: TODO

        """
        return

    def observe(self, s, a, r, s2):
        """Observe a state/action/reward vector and adds it to memory

        :s: TODO
        :a: TODO
        :r: TODO
        :s2: TODO
        :returns: TODO

        """
        return


    #def run_episode(self):


def episode(env, w, render=False):
    observation = env.reset()
    done = False
    tot_reward = 0
    for i in range(500): # run until episode is done
        if render==True:
            env.render()
        #Choose random action
        #action = env.action_space.sample()

        # Choose simple action
        #action = 1 if observation[2] > 0 else 0 # if angle if positive, move right. if angle is negative, move left

        # function approximator
        #action = 0 if w.T@observation < 0 else 1
        action = np.argmax(w.T@observation)

        observation, reward, done, _ = env.step(action)
        tot_reward += reward
        if done:
            break
    return tot_reward


env = gym.make('CartPole-v0')
for n_episode in (range(400)): # run 20 episodes
    points = 0 # keep track of the reward each episode
    render = False
    if n_episode>=390:
        render=False
        w = best_w
    tot_reward = episode(env, w, render)

    print(n_episode, tot_reward)
    if tot_reward > highscore: # record high score
        highscore = tot_reward
        best_w = w
    else:
        w = np.random.standard_normal(w.shape)


