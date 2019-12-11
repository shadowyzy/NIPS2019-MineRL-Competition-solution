# Simple env test.
import json
import select
import time
import logging
import os

import aicrowd_helper
import gym
import minerl
import argparse

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboard import TensorBoard
from meta import Agent as metaagent
from treechop import Agent as Agent1
from craft import Agent as Agent2
from stone import Agent as Agent3
from rpm import rpm
from treechop import train as treechop_train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = TensorBoard('../train_log/metacontroller')

class invent(object):

    def __init__(self):
        self.log = 0
        self.plank = 0
        self.stick = 0
        self.crafttable = 0
        self.wdpkaxe = 0
        self.stone = 0
        self.stpkaxe = 0
        self.furnace = 0

        self.log1 = 0
        self.plank1 = 0
        self.stick1 = 0
        self.crafttable1 = 0
        self.wdpkaxe1 = 0
        self.stone1 = 0
        self.stpkaxe1 = 0
        self.furnace1 = 0

    def update(self, inv):
        self.log = max(self.log, inv['log'])
        self.plank = max(self.plank, inv['planks'])
        self.stick = max(self.stick, inv['stick'])
        self.crafttable = max(self.crafttable, inv['crafting_table'])
        self.wdpkaxe = max(self.wdpkaxe, inv['wooden_pickaxe'])
        self.stone = max(self.stone, inv['cobblestone'])
        self.stpkaxe = max(self.stpkaxe, inv['stone_pickaxe'])
        self.furnace = max(self.furnace, inv['furnace'])

        if inv['log'] == 0 and inv['planks'] == 0 and inv['stick'] == 0 and inv['stone_pickaxe'] == 0 and \
            inv['crafting_table'] == 0 and inv['wooden_pickaxe'] == 0 and inv['cobblestone'] == 0 and inv['furnace'] == 0:
                return
        self.log1 = inv['log']
        self.plank1 = inv['planks']
        self.stick1 = inv['stick']
        self.crafttable1 = inv['crafting_table']
        self.wdpkaxe1 = inv['wooden_pickaxe']
        self.stone1 = inv['cobblestone']
        self.stpkaxe1 = inv['stone_pickaxe']
        self.furnace1 = inv['furnace']

    def write(self, i_episode):
        writer.add_scalar('inventory/log', self.log, i_episode)
        writer.add_scalar('inventory/plank', self.plank, i_episode)
        writer.add_scalar('inventory/stick', self.stick, i_episode)
        writer.add_scalar('inventory/crafttable', self.crafttable, i_episode)
        writer.add_scalar('inventory/wdpkaxe', self.wdpkaxe, i_episode)
        writer.add_scalar('inventory/stone', self.stone, i_episode)
        writer.add_scalar('inventory/stpkaxe', self.stpkaxe, i_episode)
        writer.add_scalar('inventory/furnace', self.furnace, i_episode)

        writer.add_scalar('inventory/log1', self.log1, i_episode)
        writer.add_scalar('inventory/plank1', self.plank1, i_episode)
        writer.add_scalar('inventory/stick1', self.stick1, i_episode)
        writer.add_scalar('inventory/crafttable1', self.crafttable1, i_episode)
        writer.add_scalar('inventory/wdpkaxe1', self.wdpkaxe1, i_episode)
        writer.add_scalar('inventory/stone1', self.stone1, i_episode)
        writer.add_scalar('inventory/stpkaxe1', self.stpkaxe1, i_episode)
        writer.add_scalar('inventory/furnace1', self.furnace1, i_episode)


def np2torch(s):
    state = torch.from_numpy(s.copy())
    return state.to(dtype=torch.float, device=device)

def main(episode):
    """
    This function will be called for training phase.
    """

    #data = minerl.data.make(MINERL_GYM_ENV, data_dir=MINERL_DATA_ROOT)

    env = gym.make('MineRLObtainDiamond-v0')

    meta = metaagent()
    agent1 = Agent1()  # treechop
    agent2 = Agent2()  # craft woodpkaxe
    agent3 = Agent3()  # stone
    meta.updata_device()
    agent1.updata_device()
    agent2.updata_device()
    agent3.updata_device()
    agent1.load_model('train/')

    print("start train")
    sum_episodes = episode
    step = 0
    all_frame = 0

    for i_episode in range(sum_episodes):
        steptime = [0 for _ in range(8)]
        agent_reward = [0 for _ in range(8)]
        frame = 0
        meta_time = 0
        inv = invent()
        obs = env.reset()
        done = False

        m_obs = [np2torch(obs['pov']) for _ in range(10)]
        m_inv = [obs['inventory'] for _ in range(10)]
        preinv = m_inv[-1]
        while (not done) and np.sum(agent_reward) < 80:
            meta_time += 1
            subagent = meta.get_action(m_inv[-1])
            if subagent == 0:
                rew, done = agent1.step(300, env, m_obs, m_inv, test=True)
                steptime[0] += 300
            elif subagent == 1 or subagent == 6 or subagent == 7:
                rew, done = agent2.step1(5, env, m_obs, m_inv, subagent)
                steptime[subagent] += 5
            elif subagent == 2:
                rew, done = agent2.step2(20, env, m_obs, m_inv, 0)
                steptime[2] += 20
            elif subagent == 3:
                rew, done = agent2.step2(20, env, m_obs, m_inv, 1)
                steptime[3] += 20
            elif subagent == 4:
                rew, done = agent2.step2(20, env, m_obs, m_inv, 2)
                steptime[4] += 20
            elif subagent == 5:
                rew, done = agent3.step(300, env, m_obs, m_inv)
                steptime[5] += 300

            agent_reward[subagent] += rew

            meta.adddata(preinv, subagent, m_inv[-1], rew, done)
            preinv = m_inv[-1]
            inv.update(m_inv[-1])

        frame = np.sum(steptime)
        _reward = np.sum(agent_reward)
        all_frame += frame
        meta.updata_epsilon(_reward)
        agent2.updata_epsilon(np.sum(agent_reward[2:5]))
        agent3.updata_epsilon(agent_reward[5])

        if i_episode < 15:
            time1 = i_episode * 5 + 10
            time2 = time1
            time3 = time2
        else :
            time1 = meta_time * 3
            time2 = steptime[2] + steptime[3] + steptime[4]
            time3 = max(300, steptime[5] // 4)
        meta_loss, meta_Q = meta.train_data(time1)
        wood_loss, wood_Q = agent2.train_data(time2)
        ston_loss, ston_Q = agent3.train_data(time3)


        inv.write(i_episode)
        writer.add_scalar('reward/reward', _reward, i_episode)
        writer.add_scalar('reward/treechop', agent_reward[0], i_episode)
        writer.add_scalar('reward/craftitm', agent_reward[1], i_episode)
        writer.add_scalar('reward/wdpkaxe ', agent_reward[2], i_episode)
        writer.add_scalar('reward/furnace ', agent_reward[3], i_episode)
        writer.add_scalar('reward/stpkaxe ', agent_reward[4], i_episode)
        writer.add_scalar('reward/stone   ', agent_reward[5], i_episode)

        writer.add_scalar('meta/Q', meta_Q, i_episode)
        writer.add_scalar('meta/loss', meta_loss, i_episode)
        writer.add_scalar('wood/Q', wood_Q, i_episode)
        writer.add_scalar('wood/loss', wood_loss, i_episode)
        writer.add_scalar('ston/Q', ston_Q, i_episode)
        writer.add_scalar('stone/loss', ston_loss, i_episode)

        writer.add_scalar('steptime/tree   ', steptime[0], i_episode)
        writer.add_scalar('steptime/craft  ', steptime[1], i_episode)
        writer.add_scalar('steptime/wdpkaxe', steptime[2], i_episode)
        writer.add_scalar('steptime/furnace', steptime[3], i_episode)
        writer.add_scalar('steptime/stpkaxe', steptime[4], i_episode)
        writer.add_scalar('steptime/stone  ', steptime[5], i_episode)


        print("")
        print('epi %d all_frame %d frame %d (reward (%4.0f))'%(i_episode, all_frame, frame, _reward))
        print(' log %3d plank %3d stick %3d table %3d wdpkaxe %3d stone %3d stpkaxe %3d furnace %3d'%\
                (inv.log, inv.plank, inv.stick, inv.crafttable, inv.wdpkaxe, inv.stone, inv.stpkaxe, inv.furnace))
        print(' log %3d plank %3d stick %3d table %3d wdpkaxe %3d stone %3d stpkaxe %3d furnace %3d'%\
                (inv.log1, inv.plank1, inv.stick1, inv.crafttable1, inv.wdpkaxe1, inv.stone1, inv.stpkaxe1, inv.furnace1))
        print('treechop step %5d reward %3d'%(steptime[0], agent_reward[0]))
        print('plank    step %5d reward %3d'%(steptime[1], agent_reward[1]))
        print('stick    step %5d reward %3d'%(steptime[6], agent_reward[6]))
        print('crafttbl step %5d reward %3d'%(steptime[7], agent_reward[7]))
        print('wdpkaxe  step %5d reward %3d'%(steptime[2], agent_reward[2]))
        print('furnace  step %5d reward %3d'%(steptime[3], agent_reward[3]))
        print('stpkaxe  step %5d reward %3d'%(steptime[4], agent_reward[4]))
        print('stone    step %5d reward %3d'%(steptime[5], agent_reward[5]))
        print('meta  Q %4.3f loss %4.3f epsilon %3.3f%%'%(meta_Q, meta_loss, meta.epsilon*100))
        print('craft Q %4.3f loss %4.3f epsilon %3.3f%%'%(wood_Q, wood_loss, agent2.epsilon*100))
        print('stone Q %4.3f loss %4.3f epsilon %3.3f%%'%(ston_Q, ston_loss, agent3.epsilon*100))

    env.close()

    meta.save_model('train/')
    agent2.save_model('train/')
    agent3.save_model('train/')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='NeurIPS 2019 MineRL Competition Train')

    # hyper-parameter

    parser.add_argument('--treechop', default=300, type=int, help='the number of episodes to train treechop model')
    parser.add_argument('--metacontroller', default=300, type=int, help='the number of episodes to train metacontroller and all subagents except treechop')

    args = parser.parse_args()

    treechop_train(args.treechop)
    main(args.metacontroller)


