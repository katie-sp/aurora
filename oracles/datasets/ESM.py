import torch, os, esm
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset

torch.manual_seed(67)

from aurora.configs.config import *

# Load the pretrained ESM2 model
esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D() #esm.pretrained.esm2_t6_8M_UR50D()
esm_model.eval()

batch_converter = esm_alphabet.get_batch_converter()
mask_idx = esm_alphabet.mask_idx

class Embedder:
    def __init__(self, wt_seq, WT_NAME):
        self.wt_seq = wt_seq
        self.WT_NAME = WT_NAME
        self.wt_embedding = None
        self.position_embeddings = {}  # Cache per-position embeddings
    
    def set_wildtype(self):
        """Cache wildtype embedding and per-position contributions"""
        save_path = f'{self.WT_NAME}_wt_token_embeddings_650M.pkl'
        if os.path.exists(save_path):
            print(f"Found {save_path}. Loading token-level embeddings from pickle...")
            with open(save_path, 'rb') as f:
                self.wt_token_embeddings = pickle.load(f)
                self.wt_embedding = self.wt_token_embeddings.mean(axis=0)  # [320]
                return 
        else:
            print('Within Embedder set_wildtype: no token-level embeddings found. Doing it the long way. ')

        # Get full WT embedding
        data = [("wt", self.wt_seq)]
        _, _, batch_tokens = batch_converter(data)
        
        print('Within Embedder set_wildtype: about to embed wt seq.')
        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[6], return_contacts=False)
        
        token_embeddings = results["representations"][6][:, 1:-1, :]  # [1, L, 320]
        self.wt_token_embeddings = token_embeddings[0].cpu().numpy()  # [L, 320]
        self.wt_embedding = self.wt_token_embeddings.mean(axis=0)  # [320]

        print('Within Embedder set_wildtype: saving token-level embeddings.')
        with open(save_path, 'wb') as file:
            pickle.dump(self.wt_token_embeddings, file)
    
    def embed_mutant(self, seq):
        """ Compute ESM embedding (mean over all amino acids) of a protein """
        data = [("protein_id", seq)] 
        _, _, batch_tokens = batch_converter(data)

        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
        #results = esm_model(batch_tokens, repr_layers=[6], return_contacts=False)

        token_embeddings = results["representations"][33][:, 1:-1, :][0].cpu().numpy()  # [L, 1280]
        # remove start/end tokens with [:, 1:-1, :]
        #token_embeddings = results["representations"][6][:, 1:-1, :][0].cpu().numpy()  # [L, 320]

        sequence_embedding = token_embeddings.mean(axis=0)  # [320]
        return sequence_embedding # [320]

class ESM_Dataset(Dataset):
    """
    PyTorch Dataset for DMS data with pre-computed embeddings.
    """
    def __init__(self, precompute=True):
        self.df = pd.read_csv(DMS_PATH).reset_index(drop=True)
        self.sequences = self.df["mutated_sequence"].tolist()
        self.scores = torch.tensor(self.df["DMS_score"].values, dtype=torch.float32)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with open(WT_PATH, 'r') as file:
            wt = file.readline().strip()  # wt seq as string

        self.embedder = Embedder(wt, WT_NAME)
        
        if precompute:
            print("Pre-computing embeddings for all sequences...")
            self.embeddings = self._precompute_embeddings()
            print(f"Done! Cached {len(self.embeddings)} embeddings")
        else:
            self.embeddings = None
    
    def _precompute_embeddings(self):
        """Compute all embeddings once and cache them"""
        embeddings = []
        print('Within ESM_MLP_Dataset _precompute_embeddings: setting wildtype.')
        self.embedder.set_wildtype()

        save_path = f'{ORACLE_DIR}/ESM_650M_embeddings.pkl'

        if os.path.exists(save_path):
            print(f"Found {save_path}. Loading embeddings from pickle...")
            with open(save_path, 'rb') as f:
                return pickle.load(f)
        else:
            print('Within ESM_MLP_Dataset _precompute_embeddings: no embeddings found. Doing it the long way.')
        
        # Process in batches for efficiency
        batch_size = 32
        with torch.no_grad():
            for i in range(0, len(self.sequences), batch_size):
                batch_seqs = self.sequences[i:i+batch_size]
                
                # Batch embedding computation
                batch_embeds = []
                for seq in batch_seqs:
                    emb = self.embedder.embed_mutant(seq)  
                    batch_embeds.append(emb)
                
                # Stack and move to device
                batch_embeds = torch.tensor(np.array(batch_embeds), 
                                           dtype=torch.float32).to(self.device)
                embeddings.append(batch_embeds)
                
                if i % 64 == 0:
                    print(f"  Processed {i}/{len(self.sequences)} sequences")
        
        # Concatenate all batches
        with open(save_path, 'wb') as file:
            pickle.dump(torch.cat(embeddings, dim=0), file)
        return torch.cat(embeddings, dim=0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.embeddings is not None:
            # Use cached embeddings - instant!
            x = self.embeddings[idx]
        else:
            # Fallback to on-the-fly computation
            x = torch.tensor(self.embedder.embed_mutant(self.sequences[idx]), 
                           dtype=torch.float32).to(self.device)
        
        y = self.scores[idx].to(self.device)
        return x, y