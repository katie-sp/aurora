"""Defines the AURORA landscape., i.e. avGFP with MLP oracle for now (until krithik tries) """

import numpy as np
import pandas as pd
# import tape
import torch
import torch.nn as nn
import torch.nn.functional as F

import flexs
from flexs import aurora_config
SURROGATE_PATH = aurora_config.SURROGATE_PATH

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


class avGFPBrightness(flexs.Landscape):
    """
    Green fluorescent protein (GFP) brightness landscape.

    The oracle used in this lanscape is the transformer model
    from TAPE (https://github.com/songlab-cal/tape).

    To create the transformer model used here, run the command:

        ```tape-train transformer fluorescence --from_pretrained bert-base \
                                               --batch_size 128 \
                                               --gradient_accumulation_steps 10 \
                                               --data_dir .```

    Note that the output of this landscape is not normalized to be between 0 and 1.

    Attributes:
        gfp_wt_sequence (str): Wild-type sequence for jellyfish
            green fluorescence protein.
        starts (dict): A dictionary of starting sequences at different edit distances
            from wild-type with different difficulties of optimization.

    """

    gfp_wt_sequence = (
        'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'
    )

    starts = { 
        'wt' : 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'
    }

    def __init__(self):
        """
        Create GFP landscape.

        Downloads model into `./fluorescence-model` if not already cached there.
        If interrupted during download, may have to delete this folder and try again.
        """
        super().__init__(name="GFP")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SurrogateNoEmbeddings()
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(SURROGATE_PATH, weights_only=True))
        else:
            model.load_state_dict(torch.load(SURROGATE_PATH, weights_only=True, map_location=torch.device('cpu')))
        model.eval()
    
        self.model = model

    def _fitness_function(self, sequences):
        sequences = np.array(sequences)
        scores = []

        # # Score sequences in batches of size 32
        # for subset in np.array_split(sequences, max(1, len(sequences) // 32)):
        #     encoded_seqs = torch.tensor(
        #         [self.tokenizer.encode(seq) for seq in subset]
        #     ).to(self.device)

        #     scores.append(
        #         self.model(encoded_seqs)[0].detach().numpy().astype(float).reshape(-1)
        #     )
        DMS = pd.read_csv('/om/user/kspiv/protein-evolution/data/Somermeyer2022_avGFP_dms_filtered.csv')
        # import pdb;pdb.set_trace()
        for mut in sequences:
            if len(DMS.loc[DMS.mutated_sequence == mut].DMS_score) > 0:   # exists in dataset
                DMS_score = DMS.loc[DMS.mutated_sequence == mut].loc[:,'DMS_score'].mean().item() #duplicates?
                # print('Used DMS')

            else:   # surrogate!
                DMS_score = self.model([mut]).item()
                # print('Used surrogate')

            scores.append(DMS_score)

        return np.array(scores) #np.concatenate(scores)
