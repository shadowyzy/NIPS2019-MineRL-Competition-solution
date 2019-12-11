import gym
import math
import random
import minerl
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboard import TensorBoard
from collections import namedtuple
from metamodel import DQN
from rpm import rpm_meta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent(object):

    def __init__(self, **kwargs):
        self.lr = 3e-4
        self.updata_time = 0
        self.batch_size = 64
        self.gamma = 0.9
        self.epsilon = 0.9
        self.Vmin = -50
        self.Vmax = 50
        self.atoms = 51
        self.actions = 8
        self.policy = DQN(8, self.actions, self.atoms)
        self.target = DQN(8, self.actions, self.atoms)
        self.reward = []
        self.datasum = 0
        self.memory = rpm_meta(100000)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr = self.lr)
        self.support = torch.linspace(self.Vmin, self.Vmax, self.atoms).to(device)

    def adddata(self, inv1, act, inv2, rew, done):
        self.datasum += 1
        inv1 = inv_to(inv1)
        inv2 = inv_to(inv2)
        act = torch.tensor([act])
        rew = torch.tensor([rew])
        gam = torch.tensor([self.gamma])
        if done:
            done = torch.tensor([1.0])
        else:
            done = torch.tensor([0.0])
        important = rew > 0.001
        self.memory.push([inv1, act, inv2, rew, done, gam], important)

    def get_action(self, inv, test=False):
        if test:
            epsilon = 0.05
        else:
            epsilon = self.epsilon
        if random.random() < epsilon:
            return random.randint(0, self.actions-1)
        with torch.no_grad():
            self.eval()
            state = inv_to(inv)
            state = state.to(dtype=torch.float)
            state = state.reshape([1] + list(state.shape))
            tmp   = (self.policy(state) * self.support).sum(2).max(1)[1]
        return (int (tmp))

    def updata_target(self):
        self.target.load_state_dict(self.policy.state_dict())

    def updata_epsilon(self, rew):

        self.reward.append(rew)
        if len(self.reward) > 200:
            self.epsilon = 0.1
        elif len(self.reward) > 160:
            self.epsilon = 0.2
        elif len(self.reward) > 100:
            self.epsilon = 0.4
        elif np.sum(self.reward) > 0:
            self.epsilon = max(0.5, self.epsilon * 0.8)

    def projection_distribution(self, next_state, reward, done, gam):
        with torch.no_grad():
            batch_size = next_state.size(0)
            delta_z = float(self.Vmax - self.Vmin) / (self.atoms - 1)
            support = torch.linspace(self.Vmin, self.Vmax, self.atoms).to(device)

            next_dist   = self.target(next_state) * support
            next_action = next_dist.sum(2).max(1)[1]
            next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))

            #DoubleDQN
            next_dist   = self.policy(next_state).gather(1, next_action).squeeze(1)

            reward  = reward.expand_as(next_dist)
            done    = done.expand_as(next_dist)
            gam     = gam.expand_as(next_dist)
            support = support.unsqueeze(0).expand_as(next_dist)

            Tz = reward + (1 - done) * gam * support
            Tz = Tz.clamp(self.Vmin, self.Vmax)
            b  = (Tz - self.Vmin) / delta_z
            l  = b.floor().long()
            u  = b.ceil().long()
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            offset = torch.linspace(0, (batch_size - 1) * self.atoms, batch_size).long()\
                    .unsqueeze(1).expand(batch_size, self.atoms)
            offset = offset.to(device)

            proj_dist = torch.zeros(next_dist.size()).to(device)

            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

            return proj_dist


    def learn(self):

        self.train()
        _loss = 0
        _Q_pred = 0

        if len(self.memory) < self.batch_size:
            return _loss, _Q_pred

        state_batch, action_batch, next_state_batch, reward_batch, done_batch, gam_batch = self.memory.sample(self.batch_size)

        action_batch = action_batch.unsqueeze(1).expand(action_batch.size(0), 1, self.atoms)
        dist_pred    = self.policy(state_batch).gather(1, action_batch).squeeze(1)
        dist_true    = self.projection_distribution(next_state_batch, reward_batch, done_batch, gam_batch)

        dist_pred.data.clamp_(0.001, 0.999)
        loss = - (dist_true * dist_pred.log()).sum(1).mean()
        self.optimizer_policy.zero_grad()
        loss.backward()
        self.optimizer_policy.step()

        with torch.no_grad():
            _loss = float(loss)
            _Q_pred = float((dist_pred * torch.linspace(self.Vmin, self.Vmax, self.atoms).to(device)).sum(1).mean())
        return _loss, _Q_pred

    def train_data(self, time):
        loss = []
        Q = []
        for i in range(time):
            _loss, _Q = self.learn()
            loss.append(_loss)
            Q.append(_Q)
            self.updata_time += 1
            if self.updata_time % 1000 == 0:
                self.updata_target()
        return np.mean(loss), np.mean(Q)

    def save_model(self, path):
        torch.save(self.policy.state_dict(),path + 'meta_DQN.pkl')

    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path + 'meta_DQN.pkl'))

    def updata_device(self, device=torch.device("cuda")):
        self.policy = self.policy.to(device=device)
        self.target = self.target.to(device=device)

    def train(self):
        self.policy.train()
        self.target.train()

    def eval(self):
        self.policy.eval()
        self.target.eval()

def inv_to(inv):
    state = torch.ones(8, dtype=torch.float32, device=device)
    state[0] *= inv['log']
    state[1] *= inv['planks']
    state[2] *= inv['stick']
    state[3] *= inv['crafting_table']
    state[4] *= inv['wooden_pickaxe']
    state[5] *= inv['cobblestone']
    state[6] *= inv['stone_pickaxe']
    state[7] *= inv['furnace']
    return state


