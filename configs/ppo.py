# set directory of project
ROOT = '/home/kspivakovsky/aurora'
WT_NAME = 'avgfp'

# set the total number of timesteps to train PPO for
TOTAL_TIMESTEPS = 10 #10000
NUM_ENVS = 64

# set path to PPO model (to save after training, and later queried to generate mutants)
PPO_DIR = ROOT + f'/ppo/{WT_NAME}_{TOTAL_TIMESTEPS}steps'