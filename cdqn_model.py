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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, ba, outputs, atoms):
        super().__init__()
        self.atoms = atoms
        self.actions = outputs
        self.conv1 = nn.Conv2d(in_channels=ba, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=outputs*atoms, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.bn3 = nn.BatchNorm2d(num_features=256)
        self.bn4 = nn.BatchNorm2d(num_features=512)
        self.bn5 = nn.BatchNorm2d(num_features=512)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = x.view(x.size(0), self.actions, self.atoms)
        x = F.softmax(x, 2)
        return x

