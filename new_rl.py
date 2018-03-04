import argparse
import gym
import random
import numpy as np
import datetime
from itertools import count
from collections import namedtuple
import ipdb
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.999, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--beta', type=float, default=4e-2, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--lr', type=float, default=3e-3, metavar='G',
                    help='discount factor (default: 0.99)')
args = parser.parse_args()
args_param = vars(args)

seed = random.randint(0, 1000)
log_interval = 1
render = False

use_cuda = torch.cuda.is_available()
#env_name = 'CartPole-v0'
env_name = 'Acrobot-v1'

now = datetime.datetime.now()
folder = f'/data/milatmp1/thomasva/proba_rl/logs/{env_name}/{now.day}_{now.month}/'
name = f'{now.hour}_{now.minute}'
for arg, val in args_param.items():
    name += f'_{arg}={val}'

name = name+'_deep'
print(name)
print(use_cuda)

writer = SummaryWriter(log_dir=folder+name)

def logsumexp(inputs, dim=-1, keepdim=False):
    return (inputs - F.log_softmax(inputs, dim=-1)).mean(dim, keepdim=keepdim)


#env = gym.make('Acrobot-v1')
env = gym.make(env_name)
env.seed(seed)
torch.manual_seed(seed)

n_obs = env.observation_space.shape[0]
n_act = env.action_space.n

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    def __init__(self, lanbda):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(n_obs, 128)
        self.affine2 = nn.Linear(128, 128)
        self.affine3 = nn.Linear(128, n_act)
        self.simple = nn.Linear(n_obs, n_act)
        self.lanbda = lanbda
        #self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        q_values = (self.affine3(x))
        #q_values = self.simple(x)
        probs = F.softmax(q_values/self.lanbda, dim=-1)
        v_value = logsumexp(q_values/self.lanbda)*self.lanbda
        #v_value = q_values.max()
        return probs, q_values, v_value

beta = args.beta
temp = 1/beta
model = Policy(temp)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay = 0e-1)
#optimizer = optim.SGD(model.parameters(), lr=3e-3, momentum=0.9)
if use_cuda:
    model.cuda()


def main():
    running_reward = 0
    step = 0
    for i_episode in count(1):
        state = env.reset()
        state = torch.from_numpy(state).float()
        tot_reward = 0
        running_delta = 0
        running_fdelta = 0
        running_loss = 0
        model.lanbda *= .999
        model.lanbda = max(model.lanbda, 10)
        beta = 1/model.lanbda
        loss_ep = 0
        kl_ep = 0
        for t in range(10000):  # Don't infinite loop while learning
            step += 1
            if use_cuda:
                state_var = Variable(state).cuda()
            else:
                state_var = Variable(state)

            p_sa, q_s, v_s = model(state_var)
            m = Categorical(p_sa)
            action = m.sample()
            q_sa = q_s[action.data[0]]
            #action = select_action(state)
            state, reward, done, _ = env.step(action.data[0])
            state = torch.from_numpy(state).float()
            if use_cuda:
                state_var = Variable(state).cuda()
            else:
                state_var = Variable(state)

            # ====  Update
            p_ss, q_ss, v_ss = model(state_var)
            delta = -beta*(q_sa - v_ss - reward)
            # if done:
            #     delta = beta*(q_sa - reward)
            # if t == 0:
            #     delta = beta*(- v_ss - reward)
                #delta = beta*(q_sa - v_ss - reward)
            #fdelta = F.smooth_l1_loss(delta, Variable(torch.zeros(1).cuda()))
            fdelta = delta
            running_delta = running_delta * 0.999 + delta.detach() * 0.001
            running_fdelta = running_delta * 0.999 + fdelta.detach() * 0.001

            loss_ep += fdelta.detach()
            kl_ep += delta.detach()

            #delta *= (1+.1*torch.exp(beta*(reward+v_s-q_sa)))
            loss = (fdelta.detach()-running_fdelta)*m.log_prob(action) + fdelta
            optimizer.zero_grad()
            (args.gamma**t*loss).backward()
            optimizer.step()
            tot_reward += reward


            info ={
                    'v_s': v_s.data[0],\
                    'v_ss-q_sa': (v_ss-q_sa).data[0],\
                    'v_ss': v_ss.data[0],\
                    'q_sa': q_sa.data[0]}

            for tag, value in info.items():
                writer.add_scalar(tag, value, step)

            if render:
                env.render()
            model.rewards.append(reward)
            if done:
                break

        running_reward = running_reward * 0.99 + t * 0.01
        #print(total_loss)
        if i_episode % log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
            print(delta.data[0])
            print(running_delta.data[0])
            info = {'reward': tot_reward,\
                    'running_delta': running_delta.data[0],\
                    'loss': loss_ep.data[0],\
                    'kl': kl_ep.data[0],\
                    'v_s': v_s.data[0],\
                    'v_ss-q_sa': (v_ss-q_sa).data[0],\
                    'v_ss': v_ss.data[0],\
                    'lanbda': model.lanbda,\
                    'q_sa': q_sa.data[0]}

            for tag, value in info.items():
                writer.add_scalar(tag, value, step)
        # if i_episode % 100 == 0:
        #     ipdb.set_trace()


if __name__ == '__main__':
    main()
