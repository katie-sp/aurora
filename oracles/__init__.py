from oracles.ESM_MLP import ESM_MLP_Oracle
from oracles.raw_MLP import Raw_MLP_Oracle
from oracles.datasets.ESM import ESM_Dataset
from oracles.datasets.raw import Raw_Dataset

ORACLE_REGISTRY = {
    "esm_mlp": (ESM_MLP_Oracle, ESM_Dataset),
    "raw_mlp": (Raw_MLP_Oracle, Raw_Dataset),
}