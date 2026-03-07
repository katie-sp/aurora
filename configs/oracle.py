# set directory of project
ROOT = '/home/kspivakovsky/aurora'
WT_NAME = 'avgfp'

# set path to oracle model (for training, and later for querying)
ORACLE_DIR = ROOT + f'/oracles/{WT_NAME}_Raw_MLP'
ORACLE_NAME = 'raw_mlp'
# ORACLE_DIR = ROOT + '/oracles/{WT_NAME}_ESM_MLP'
# ORACLE_NAME = 'esm_mlp'

# set the path to the wild-type sequence as .txt 
WT_PATH = ROOT + '/data/avgfp_wt.txt'

# set the path to the DMS dataset as .csv
DMS_PATH = ROOT + '/data/Somermeyer2022_avGFP_dms_filtered.csv'

# set the first and last position of the protein subsequence that is being evolved
FIRST_POS = 1  # NOT 0-indexed!
LAST_POS = 238
