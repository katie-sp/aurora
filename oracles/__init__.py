from oracles.ESM_MLP import ESM_MLP_Oracle, ESM_MLP_fitness
from oracles.raw_MLP import Raw_MLP_Oracle, Raw_MLP_fitness
from oracles.datasets.ESM import ESM_Dataset
from oracles.datasets.raw import Raw_Dataset

ORACLE_REGISTRY = {
    "esm_mlp": (ESM_MLP_Oracle, ESM_Dataset, ESM_MLP_fitness),
    "raw_mlp": (Raw_MLP_Oracle, Raw_Dataset, Raw_MLP_fitness),
}