"""AURORA PPO explorer."""

from functools import partial
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import flexs
from flexs import aurora_config
WT_PATH = aurora_config.WT_PATH
DMS_PATH = aurora_config.DMS_PATH
SURROGATE_PATH = aurora_config.SURROGATE_PATH
MODEL_PATH = aurora_config.PPO_PATH

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3 import PPO

import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
from gymnasium import spaces

FIRST_POS = 1  # NOT 0-indexed!
LAST_POS = 238


class AURORA(flexs.Explorer):
    """
    Explorer which uses PPO trained by AURORA.

    The algorithm is:
        for N experiment rounds
            collect samples with policy
            train policy on samples

    A simpler baseline than DyNAPPOMutative with similar performance.
    """

    def __init__(
        self,
        model,
        rounds: int,
        sequences_batch_size: int,
        model_queries_per_batch: int,
        starting_sequence,
        log_file: Optional[str] = None,
    ):
        """Create PPO explorer."""
        super().__init__(
            model,
            "PPO_Agent",
            rounds,
            sequences_batch_size,
            model_queries_per_batch,
            starting_sequence,
            log_file,
        )

        #  # Load wild-type sequence
        with open(WT_PATH, 'r') as file:
            wt = file.readline().strip()
        
        # Create a single environment for sampling
        def make_env():
            return ProteinEnv(wt, self.fitness_DMS, DMS_PATH)
        
        self.env = make_env()
        
        # Load the trained model
        print(f"Loading model from {MODEL_PATH}...")
        self.ppo = PPO.load(MODEL_PATH)

    def fitness_DMS(self, mut):
        ''' Calculate fitness based solely on DMS (for use of models without ESM/Pairformer embeddings)
        '''
        SURROGATE_PATH = '/om/user/kspiv/protein-evolution/models/avgfp_dms_surrogate_noembeddings.state_dict'
        model = SurrogateNoEmbeddings()
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(SURROGATE_PATH, weights_only=True))
        else:
            model.load_state_dict(torch.load(SURROGATE_PATH, weights_only=True, map_location=torch.device('cpu')))
        model.eval()
    

        DMS = pd.read_csv('/om/user/kspiv/protein-evolution/data/Somermeyer2022_avGFP_dms_filtered.csv')
        if len(DMS.loc[DMS.mutated_sequence == mut].DMS_score) > 0:   # exists in dataset
            DMS_score = DMS.loc[DMS.mutated_sequence == mut].loc[:,'DMS_score'].mean().item() #duplicates?
            # print('Used DMS')

        else:   # surrogate!
            DMS_score = model([mut]).item()
            # print('Used surrogate')

        return DMS_score

    def propose_sequences(
        self, measured_sequences: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Propose top `sequences_batch_size` sequences for evaluation."""
        num_parallel_environments = 1

        sequences = {}

        # We propose the top `self.sequences_batch_size` new sequences we have generated        
        for i in range(2 * self.sequences_batch_size):
            obs, _ = self.env.reset()
            truncated = False
            
            # Run episode until truncated (6 steps)
            while not truncated:
                action, _states = self.ppo.predict(obs, deterministic=False)  # Use deterministic=False for sampling
                obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Extract final sequence
            final_seq = self.env.idxs_to_letters(self.env.state)
            # import pdb;pdb.set_trace()
            sequences[final_seq] = self.fitness_DMS(final_seq)

        new_seqs = np.array(list(sequences.keys()))
        preds = np.array(list(sequences.values()))
        sorted_order = np.argsort(preds)[: -self.sequences_batch_size : -1]

        return new_seqs[sorted_order], preds[sorted_order]


class SurrogateNoEmbeddings(nn.Module):
    """
    Optimized MLP model for learning from raw amino acid sequences.
    """
    def __init__(self, seq_length=238, emb_dim=64, n_layers=1, n_heads=2, dropout=0.1):
        super().__init__()
        self.seq_length = seq_length
        self.emb_dim = emb_dim

        # Create mapping - use a fixed order
        aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 
                   'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
        self.aa_to_idx = {aa: i for i, aa in enumerate(aa_list)}
        self.unk_idx = self.aa_to_idx['X']

        self.aa_embedding = nn.Embedding(num_embeddings=len(self.aa_to_idx), embedding_dim=emb_dim)
        self.pos_embedding = nn.Embedding(seq_length, emb_dim)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, 
            dim_feedforward=emb_dim*2, batch_first=True, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.mlp_head = nn.Sequential(
            nn.Linear(seq_length * emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def encode_sequences(self, sequences):
        """
        Optimized batch encoding of sequences to indices.
        Uses vectorized operations instead of nested loops.
        """
        batch_size = len(sequences)
        device = next(self.parameters()).device
        
        # Pre-allocate output tensor
        idxs = torch.full((batch_size, self.seq_length), self.unk_idx, 
                         dtype=torch.long, device=device)
        
        # Process all sequences at once
        for i, seq in enumerate(sequences):
            seq_upper = seq.upper()
            seq_len = min(len(seq_upper), self.seq_length)
            for j in range(seq_len):
                idxs[i, j] = self.aa_to_idx.get(seq_upper[j], self.unk_idx)
        
        return idxs

    def forward(self, sequences):
        """
        Args:
            sequences (List[str]): list of protein sequences
        Returns:
            predictions: shape [B]
        """
        batch_size = len(sequences)
        device = next(self.parameters()).device

        # Encode sequences efficiently
        idxs = self.encode_sequences(sequences)

        # Embedding
        aa_emb = self.aa_embedding(idxs)  # [B, seq_length, emb_dim]
        positions = torch.arange(self.seq_length, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)  # [B, seq_length, emb_dim]

        x = aa_emb + pos_emb
        x = self.dropout(x)

        # Transformer encoding
        x = self.transformer_encoder(x)  # [B, seq_length, emb_dim]

        # Flatten and predict
        x = x.reshape(batch_size, -1)
        x = self.mlp_head(x).squeeze(-1)
        return x

class ProteinEnv(gym.Env):
    """
    State: amino acid sequence (string or int array)
    Action: mutate position i to amino acid j
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, seq, fitness_fn, DMS_path):
        ''' Requires the wild-type aa sequence (string), 
                fitness_fn (defined in fitness_functions.py),
            and DMS dataset (path to csv)
        '''
        super().__init__()
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.amino_acids)}
        self.idx_to_aa = {i: aa for aa, i in self.aa_to_idx.items()}
        self.actions = []
        self.step_count = 0

        self.L = len(seq)
        # self.fitness_fn = self.fitness_DMS #fitness_fn
        self.DMS = pd.read_csv(DMS_path)
        
        # convert sequence string → array of indices
        self.wt = np.array([self.aa_to_idx[a] for a in seq], dtype=np.int32)

        # action = choose a position to mutate, and choose an aa to mutate to
        self.action_space = spaces.MultiDiscrete([LAST_POS - FIRST_POS + 1, 20])

        # observation = vector of length L with values in [0,19]
        self.observation_space = spaces.MultiDiscrete([20] * self.L)

        self.state = None
        self.step_count = 0
    
    def idxs_to_letters(self, seq):
        ''' convert string of indexes to string of aa letters '''
        return ''.join([self.idx_to_aa[i] for i in seq])

    def _decode_action(self, action):
        pos, aa_idx = action
        pos += FIRST_POS - 1
        self.actions.append((pos, aa_idx))
        return pos, aa_idx

    def reset(self, *, seed=None, options=None):
        # print(f'Step count: {self.step_count}')
        super().reset(seed=seed)
        self.state = self.wt.copy()  # back to wild-type
        self.step_count = 0
        obs = self.state.copy()
        return obs, {}

    def fitness_DMS(self, mut):
        ''' Calculate fitness based solely on DMS (for use of models without ESM/Pairformer embeddings)
        '''
        SURROGATE_PATH = '/om/user/kspiv/protein-evolution/models/avgfp_dms_surrogate_noembeddings.state_dict'
        model = SurrogateNoEmbeddings()
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(SURROGATE_PATH, weights_only=True))
        else:
            model.load_state_dict(torch.load(SURROGATE_PATH, weights_only=True, map_location=torch.device('cpu')))
        model.eval()

        DMS = pd.read_csv('/om/user/kspiv/protein-evolution/data/Somermeyer2022_avGFP_dms_filtered.csv')
        if len(DMS.loc[DMS.mutated_sequence == mut].DMS_score) > 0:   # exists in dataset
            DMS_score = DMS.loc[DMS.mutated_sequence == mut].loc[:,'DMS_score'].mean().item() #duplicates?
            print('Used DMS')

        else:   # surrogate!
            DMS_score = model([mut]).item()
            print('Used surrogate')

        return DMS_score

    def step(self, action):
        pos, aa_idx = self._decode_action(action)

        # Apply mutation
        new_state = self.state.copy()
        new_state[pos] = aa_idx

        # Reward from fitness function
        # reward, dataset_used = self.fitness_fn(self.idxs_to_letters(new_state))
        reward, dataset_used = self.fitness_DMS(self.idxs_to_letters(new_state)), 'who cares'

        # You can choose episode termination rule:
        # e.g., fixed length episode of mutations
        terminated = False # there are no natural terminal states for the model; the protein is never "done" evolving
        
        # Increment step count before checking truncation
        self.step_count += 1
        self.state = new_state
        
        # Check truncation after incrementing (so we get exactly 6 steps before truncating)
        truncated = (self.step_count > 5) # truncate after 6 steps (step_count will be 7 after 6 steps)

        # Key distinction: truncated allows for future bootstrapping, meaning that the model could continue to receive reward in the future.
        info = {'dataset_used': dataset_used, 
        'mutation_count': (self.wt != new_state),
        'num_mutations_per_variant': (self.wt != new_state).sum()}

        return new_state.copy(), reward, terminated, truncated, info

    def render(self):
        seq_str = "".join(self.idx_to_aa[i] for i in self.state)
        print(seq_str)

