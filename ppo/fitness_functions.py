import torch, pickle, os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from config import SURROGATE_PATH

def fitness_DMS(wt, mut, DMS, dms_mean=2.658, dms_std=1.058):
    ''' Calculate fitness based solely on DMS (for use of models without ESM/Pairformer embeddings)
    '''

    if len(DMS.loc[DMS.mutated_sequence == mut].DMS_score) > 0:   # exists in dataset
        DMS_score = DMS.loc[DMS.mutated_sequence == mut].loc[:,'DMS_score'].mean().item() #duplicates?
        dataset = 'DMS'
        print('Used DMS')

    else:   # surrogate!
        DMS_score = surrogate([mut]).item()
        print('Used surrogate')
        dataset = 'surrogate'

    DMS_normalized = (DMS_score - dms_mean) / dms_std

    return DMS_normalized, dataset

#### HELPERS

class SurrogateESM(nn.Module):   # surrogate MLP for ESM embeddings
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(320, 320)
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
        
Surrogate = SurrogateNoEmbeddings
surrogate = Surrogate()
if torch.cuda.is_available():
    surrogate.load_state_dict(torch.load(SURROGATE_PATH, weights_only=True))
else:
    surrogate.load_state_dict(torch.load(SURROGATE_PATH, weights_only=True, map_location=torch.device('cpu')))
surrogate.eval()
