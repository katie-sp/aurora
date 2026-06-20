## DELETE THIS FILE???
import numpy as np
import torch.nn.functional as F

from configs.config import ORACLE_NAME, DMS_MEAN, DMS_STD
from oracles import ORACLE_REGISTRY

def fitness(wt, mut, DMS):
    ''' Calculate fitness based on oracle'''

    DMS_score = ORACLE_REGISTRY[ORACLE_NAME][0]([mut]).item()

    DMS_normalized = (DMS_score - DMS_MEAN) / DMS_STD

    return DMS_normalized, None
