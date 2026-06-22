# AURORA: Alignment-Guided Mutation Proposer and Oracle

## Modifying the config
`config.py` is the main file you'll need to modify in order to train and run our built-in oracle types and PPO mutation proposer. Comments should adequately explain each variable.

## Training your own oracle
Run `training/train_oracle.py` as a module, e.g. from within `aurora` run `python -m training.train_oracle`. Trained model and metrics will be saved in `oracles` according to the protein name and embedding and model type, e.g. `oracles/avgfp_esm_mlp`.

## Training your own mutation proposal policy
After training an oracle, run `training/train_ppo.py` as a module, e.g. from within `aurora` run `python -m training.train_ppo`. Trained PPO model and metrics will be saved in `ppo` according to the protein name, embedding and model type, and number of steps PPO was trained for, e.g. `ppo/avgfp_esm_mlp_10000steps`.

## Generating mutants
After training a PPO model, run `generation/sample_variants_from_policy.py` as a module, e.g. from within `aurora` run `python -m generation.sample_variants_from_policy`. By default, 10 variants will be sampled from the policy, and their sequences and predicted fitness (according to the oracle defined in `config.py`) will be saved in `generation` according to the protein name, embedding and model type, number of steps PPO was trained for, and date, e.g. `generation/avgfp_esm_mlp_10000steps/2026-06-22_15-18-22_sampled_variants.csv`.

## Analyzing mutants
Some sample analysis notebooks exist in `analysis` but are not well-commented, and you may benefit from creating your own custom analysis pipeline for your use case.