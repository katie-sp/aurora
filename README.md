# AURORA: Alignment-Guided Mutation Proposer and Oracle

**Paper accepted at the 2026 Workshop on Generative and Agentic AI for Biology (ICML 2026)**: https://openreview.net/pdf?id=Pj84px7sne

## Getting started
You will need to install packages either via pip (`pip install -r requirements.txt`) or conda (`conda env create -f requirement.yml`). This includes ESM but not MSA Pairformer or BiMamba, which you must install separately if you'd like to use those models.

## Modifying the config
`config.py` is the main file you'll need to modify in order to train and run our built-in oracle types and PPO mutation proposer on your own protein sequence. Comments should adequately explain each variable.

## Training your own oracle
Run `training/train_oracle.py` as a module, e.g. from within `aurora` run `python -m training.train_oracle`. Trained model and metrics will be saved in `oracles` according to the protein name and embedding and model type, e.g. `oracles/avgfp_esm_mlp`.

To define your own custom oracle:
1. If a new type of sequence embedding is necessary, create a PyTorch Dataset within `oracles/datasets/`.
2. Define the new oracle in `oracles` by inheriting from BaseOracle, and within the same file, define a fitness function that queries from the oracle to predict the fitness of a mutant.
3. Modify `oracles/__init__.py` to add the oracle to the registry.

## Training your own mutation proposal policy
After training an oracle, run `training/train_ppo.py` as a module, e.g. from within `aurora` run `python -m training.train_ppo`. Trained PPO model and metrics will be saved in `ppo` according to the protein name, embedding and model type, and number of steps PPO was trained for, e.g. `ppo/avgfp_esm_mlp_10000steps`.

## Generating mutants
After training a PPO model, run `generation/sample_variants_from_policy.py` as a module, e.g. from within `aurora` run `python -m generation.sample_variants_from_policy`. By default, 10 variants will be sampled from the policy, and their sequences and predicted fitness (according to the oracle defined in `config.py`) will be saved in `generation` according to the protein name, embedding and model type, number of steps PPO was trained for, and date, e.g. `generation/avgfp_esm_mlp_10000steps/2026-06-22_15-18-22_sampled_variants.csv`.

## Analyzing mutants
Some sample analysis notebooks exist in `analysis` but are not well-commented, and you may benefit from creating your own custom analysis pipeline for your use case.
