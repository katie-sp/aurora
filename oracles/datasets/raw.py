import torch
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset

torch.manual_seed(67)

from configs.config import *

class Raw_Dataset(Dataset):
    """
    Optimized PyTorch Dataset for DMS data with raw protein sequences.
    """
    def __init__(self, seq_col="mutated_sequence", score_col="DMS_score"):
        self.df = pd.read_csv(DMS_PATH).reset_index(drop=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sequences = self.df[seq_col].tolist()
        self.scores = torch.tensor(self.df[score_col].values, dtype=torch.float32, device=self.device)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.scores[idx]