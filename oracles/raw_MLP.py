import torch
from torch import nn

from config import *
from oracles.base import BaseOracle

with open(WT_PATH, 'r') as file:
    wt = file.readline().strip()

class Raw_MLP_Oracle(BaseOracle):
    """
    Optimized MLP model for learning from raw amino acid sequences.
    """
    def __init__(self, seq_length=LAST_POS - FIRST_POS + 1, emb_dim=64, n_layers=1, n_heads=2, dropout=0.1):
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

    def encode(self, sequences):
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
        idxs = self.encode(sequences)

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

def raw_MLP_fitness(embeddings):
    oracle = Raw_MLP_Oracle()
    if torch.cuda.is_available():
        oracle.load_state_dict(torch.load(ORACLE_DIR + '/oracle.state_dict', weights_only=True))
    else:
        oracle.load_state_dict(torch.load(ORACLE_DIR + '/oracle.state_dict', weights_only=True, map_location=torch.device('cpu')))
    oracle.eval()
    return oracle.forward([embeddings]).item()