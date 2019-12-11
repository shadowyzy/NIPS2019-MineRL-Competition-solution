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

from meta import Agent as metaagent
from treechop import Agent as Agent1
from craft import Agent as Agent2
from stone import Agent as Agent3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def np2torch(s):
    state = torch.from_numpy(s.copy())
    return state.to(dtype=torch.float, device=device)



def main(episodes):
    """
    This function will be called for training phase.
    """
    # Load trained model from train/ directory

    env = gym.make('MineRLObtainDiamond-v0')
    meta = metaagent()
    agent1 = Agent1()  # treechop
    agent2 = Agent2()  # craft woodpkaxe
    agent3 = Agent3()  # stone
    meta.load_model('train/')
    agent1.load_model('train/')
    agent2.load_model('train/')
    agent3.load_model('train/')
    meta.updata_device()
    agent1.updata_device()
    agent2.updata_device()
    agent3.updata_device()

    print("start test")
    step = 0
    all_frame = 0

    for i_episode in range(episodes):
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
        while not done:
            meta_time += 1
            subagent = meta.get_action(m_inv[-1], test=True)
            if subagent == 0:
                rew, done = agent1.step(300, env, m_obs, m_inv, test=True)
                steptime[0] += 300
            elif subagent == 1 or subagent == 6 or subagent == 7:
                rew, done = agent2.step1(5, env, m_obs, m_inv, subagent, test=True)
                steptime[subagent] += 5
            elif subagent == 2:
                rew, done = agent2.step2(20, env, m_obs, m_inv, 0, test=True)
                steptime[2] += 20
            elif subagent == 3:
                rew, done = agent2.step2(20, env, m_obs, m_inv, 1, test=True)
                steptime[3] += 20
            elif subagent == 4:
                rew, done = agent2.step2(20, env, m_obs, m_inv, 2, test=True)
                steptime[4] += 20
            elif subagent == 5:
                rew, done = agent3.step(300, env, m_obs, m_inv, test=True)
                steptime[5] += 300
            agent_reward[subagent] += rew
            preinv = m_inv[-1]
            inv.update(m_inv[-1])

        frame = np.sum(steptime)
        _reward = np.sum(agent_reward)
        all_frame += frame

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
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NeurIPS 2019 MineRL Competition Test')

    # hyper-parameter
    parser.add_argument('--episodes', default=100, type=int, help='the number of episodes to test')

    args = parser.parse_args()

    main(args.episodes)




