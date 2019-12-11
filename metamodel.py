# -*- coding: utf-8 -*-
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
from collections import namedtuple

class DQN(nn.Module):
    def __init__(self, inputs, outputs, atoms):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.atoms = atoms
        self.linear1 = nn.Linear(inputs, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, outputs * atoms)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        x = x.view(x.size(0), self.outputs, self.atoms)
        x = F.softmax(x, 2)
        return x
