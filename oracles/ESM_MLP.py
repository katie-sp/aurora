import torch, os, esm
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset

torch.manual_seed(67)

from configs.config import *
from oracles.base import BaseOracle

with open(WT_PATH, 'r') as file:
    wt = file.readline().strip()

class ESM_MLP_Oracle(BaseOracle):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(320, 320) # was 1280,320 for 650M model
        self.l2 = nn.Linear(320, 128)
        self.l3 = nn.Linear(128, 1)
    
    def forward(self, embeddings):
        # embeddings = 16 x 320 or similar batch size x esm embedding size
        x = self.l1(embeddings)
        x = F.relu(x).square()
        x = self.l2(x)
        x = F.relu(x).square()
        x = self.l3(x)
        return x.squeeze(-1)   # shape [16]

    
    
