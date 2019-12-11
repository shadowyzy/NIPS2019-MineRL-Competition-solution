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
from cdqn_model_res import DQN
from rpm import rpm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent(object):

    def __init__(self, **kwargs):
        self.lr = 3e-4
        self.updata_time = 0
        self.batch_size = 64
        self.gamma = 0.9
        self.epsilon = 0.9
        self.Vmin = -100
        self.Vmax = 100
        self.atoms = 51
        self.actions = 7
        self.policy = DQN(10, self.actions, self.atoms)
        self.target = DQN(10, self.actions, self.atoms)
        self.reward = []
        self.memory = rpm(200000,10)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr = self.lr)
        self.support = torch.linspace(self.Vmin, self.Vmax, self.atoms).to(device)

    def get_action(self, state, test=False):
        if test:
            epsilon = 0.05
        else:
            epsilon = self.epsilon
        if random.random() < epsilon:
            return random.randint(0, self.actions-1)
        with torch.no_grad():
            self.eval()
            state   = state.to(dtype=torch.float, device=device)
            state   = state.reshape([1] + list(state.shape))
            tmp     = (self.policy(state) * self.support).sum(2).max(1)[1]
        return (int (tmp))

    def updata_target(self):
        self.target.load_state_dict(self.policy.state_dict())

    def updata_epsilon(self, rew):

        self.reward.append(rew)
        if len(self.reward) > 100:
            self.epsilon = 0.1
        elif len(self.reward) > 60:
            self.epsilon = 0.2
        elif np.sum(self.reward) > 0:
            self.epsilon = max(0.4, self.epsilon * 0.8)

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
        torch.save(self.policy.state_dict(),path + 'wdpkaxe_DQN.pkl')

    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path + 'wdpkaxe_DQN.pkl'))

    def updata_device(self, device=torch.device("cuda")):
        self.policy = self.policy.to(device=device)
        self.target = self.target.to(device=device)

    def train(self):
        self.policy.train()
        self.target.train()

    def eval(self):
        self.policy.eval()
        self.target.eval()

    def step1(self, step, env, m_obs, m_inv, kind, test=False):
        _reward = 0
        done = False
        action = action_to(1)
        action['attack'] = 0
        time = 1

        if kind == 1:
            action['craft'] = 3 # plank
            time = 5
        elif kind == 6:
            action['craft'] = 4 # craft table
        elif kind == 7:
            action['craft'] = 2 # stick


        for i in range (time):
            obs, rew, done, info = envstep_done(env, action, done)
            _reward += rew

        if not done:
            m_inv[-1] = obs['inventory']
            m_obs[-1] = np2torch(obs['pov'])
        return _reward, done

    def step2(self, step, env, m_obs, m_inv, kind, test=False):
        TD_step = 2
        _reward = 0
        frame = 0
        uptime = 0
        done = False
        self.memory.clear_recent()
        m_reward = [0 for _ in range(10)]
        m_action = [torch.tensor([0]) for _ in range(10)]
        state = [state_to(m_obs[-3:], uptime) for _ in range(10)]

        while not done and frame < step:
            action_num = self.get_action(state[-1], test)
            obs, rew, done, info, t = envstep(env, action_num, kind)
            _reward += rew
            frame += t

            if action_num == 5:
                uptime = max(uptime-1, -4)
            elif action_num == 6:
                uptime = min(uptime+1, 4)

            for i in range(9):
                m_obs[i] = m_obs[i+1]
                state[i] = state[i+1]
                m_inv[i] = m_inv[i+1]
                m_reward[i] = m_reward[i+1]
                m_action[i] = m_action[i+1]

            if done :
                _done = torch.tensor([1.0])
            else:
                m_obs[-1] = np2torch(obs['pov'])
                m_inv[-1] = obs['inventory']
                state[-1] = state_to(m_obs[-3:], uptime)
                m_reward[-1] = rew
                m_action[-1] = torch.tensor([action_num])
                _done = torch.tensor([0.0])

            if test:
                continue

            reward, gam = 0.0, 1.0
            for i in range(TD_step):
                reward += gam * m_reward[i-TD_step]
                gam *= self.gamma
            reward = torch.tensor([reward])
            gam = torch.tensor([gam])
            important = reward > 0.001
            if frame >= TD_step:
                self.memory.push([state[-TD_step-1], m_action[-TD_step], state[-1], reward, _done, gam], important)

        while uptime < 0:
            uptime += 1
            action = action_to(6)
            obs, rew, done, info = envstep_done(env, action, done)
        while uptime > 0:
            uptime -= 1
            action = action_to(5)
            obs, rew, done, info = envstep_done(env, action, done)
        #if not done:
        #    for i in range(10):
        #        m_obs[i] = obs['pov']

        return _reward, done

def np2torch(s):
    state = torch.from_numpy(s.copy())
    return state.to(dtype=torch.float, device=device)

def state_to(pov, up):
    tmp = torch.ones(list(pov[0].shape[:-1])+[1], dtype = torch.float, device=device) * up
    state = torch.cat([pov[0], pov[1], pov[2], tmp], 2)
    state = state.permute(2, 0, 1)
    return state.to(torch.device('cpu'))


def state_to1(pov, up):
    pov = np.concatenate([pov[0], pov[1], pov[2]], axis=-1)
    up  = np.ones(shape = list(pov.shape[:-1]) + [1], dtype = np.uint8) * up
    tmp = np.concatenate([pov, up], axis=-1)
    tmp = tmp.tolist()
    state = torch.tensor(tmp, dtype=torch.uint8, device=device)
    state = state.permute(2, 0, 1)
    return state

def action_to(num):
    act = {
        "forward": 1,
        "back": 0,
        "left": 0,
        "right": 0,
        "jump": 0,
        "sneak": 0,
        "sprint": 0,
        "attack" : 0,
        "camera": [0,0],

        "place": 0,
        "craft": 0,
        "equip": 0,
        "nearbyCraft": 0,
        "nearbySmelt": 0,
    }
    if num == 1:
        act['forward'] = 0
    elif num == 2 :
        act['jump'] = 1
    elif num == 3:
        act['camera'] = [0, -30]
    elif num == 4:
        act['camera'] = [0, 30]
    elif num == 5:
        act['camera'] = [-22.5, 0]
    elif num == 6:
        act['camera'] = [22.5, 0]

    return act.copy()

def envstep(env, action_num, kind):
    reward = 0
    action = action_to(action_num)
    for i in range(4):
        obs, rew, done, info = env.step(action)
        reward += rew
        if done or action_num >= 3:
            return nearbycraft(env, reward, done, i+1, kind)
    return nearbycraft(env, reward, done, 4, kind)

def nearbycraft(env, _reward, done, fra, kind):
    action = action_to(0)

    action['place'] = 4 #craft table
    obs, rew, done, info = envstep_done(env, action, done)
    _reward += rew
    action['place'] = 0

    if kind == 0:       # wooden pickaxe
        action['nearbyCraft'] = 2
    elif kind == 1:     # furnace
        action['nearbyCraft'] = 7
    elif kind == 2:     # stone pickaxe
        action['nearbyCraft'] = 4

    obs, rew, done, info = envstep_done(env, action, done)
    _reward += rew
    return obs, _reward, done, info, fra+2


def envstep_done(env, act, done):
    if done:
        return 0, 0, done, 0
    else :
        return env.step(act)


