import gym
from tqdm import tqdm
import numpy as np

env = gym.make('CartPole-v0')
highscore = 0
# Funciton apprixmator
w = np.random.standard_normal(env.observation_space.shape)

def episode(env, w):
    observation = env.reset()
    done = False
    while not done: # run until episode is done
        #env.render()
        #Choose random action
        #action = env.action_space.sample()

        # Choose simple action
        #action = 1 if observation[2] > 0 else 0 # if angle if positive, move right. if angle is negative, move left

        # function approximator
        action = 0 if w@observation < 0 else 1

        observation, reward, done, _ = env.step(action)
    return reward

for i_episode in tqdm(range(200)): # run 20 episodes
    points = 0 # keep track of the reward each episode
    reward = episode(env, w)
    points += reward
    if points > highscore: # record high score
        highscore = points
    #print(highscore)

