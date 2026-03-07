# set directory of project
ROOT = '/home/kspivakovsky/aurora'
WT_NAME = 'avgfp'

# set the total number of timesteps to train PPO for
TOTAL_TIMESTEPS = 10000

# set path to PPO model (for training, and later for querying)
PPO_DIR = ROOT + '/ppo/{WT_NAME}_{TOTAL_TIMESTEPS}steps'