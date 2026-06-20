# set directory of project
ROOT = '/home/kspivakovsky/aurora'
WT_NAME = 'avgfp'  # will be used to name saved files, but is not used to find any data/files/etc.

# set path to oracle model (for training, and later for querying)
# ORACLE_NAME = 'esm_mlp'
ORACLE_NAME = 'raw_mlp'
ORACLE_DIR = ROOT + f'/oracles/{WT_NAME}_{ORACLE_NAME}'

# set the path to the wild-type sequence as .txt 
WT_PATH = ROOT + '/data/avgfp_wt.txt'

# set the path to the DMS dataset as .csv
DMS_PATH = ROOT + '/data/Somermeyer2022_avGFP_dms_filtered.csv'
DMS_MEAN = 2.658
DMS_STD = 1.058

# set the first and last position of the protein subsequence that is being evolved
FIRST_POS = 1  # NOT 0-indexed!
LAST_POS = 238
