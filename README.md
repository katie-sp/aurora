# AURORA: Alignment-Guided Mutation Proposer and Oracle

## File organization

protein_engineering/
│
├── configs/
│   ├── oracle/
│   ├── ppo/
│   ├── experiments/
│
├── oracles/   **each file should be a self-contained oracle, i.e. both the BaseOracle-based oracle, as well as the PyTorch Dataset for efficient sample batching**
│   ├── esm_mlp.py/
│   ├── raw_mlp.py/
│
── ppo/
│   all the scripts defining environment, etc.
│   
├── training/
│   ├── train_oracle.py
│   ├── train_ppo.py
│   ├── train_end_to_end.py
│
├── generation/
│   ├── generate.py
│
├── utils/
│   ├── logging.py
│   ├── io.py
│   ├── metrics.py
│
├── train.py
└── evaluate.py

