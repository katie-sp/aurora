import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import sys, os

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

from datetime import datetime
from oracles import ORACLE_REGISTRY
from ppo.environments import ProteinEnv
from config import *

NUM_VARIANTS = 10  # vary this to sample more or fewer variants
MODEL_PATH = PPO_DIR + '/ppo_model' # zip file
MODEL_TYPE = f'{WT_NAME}_{ORACLE_NAME}_{TOTAL_TIMESTEPS}steps' # name to title .csv as

if __name__ == "__main__":
    
    # Load wild-type sequence
    with open(WT_PATH, 'r') as file:
        wt = file.readline().strip()
    
    # Load DMS dataset for fitness computation
    DMS = pd.read_csv(DMS_PATH)
    def fitness(mut):
        DMS_score = ORACLE_REGISTRY[ORACLE_NAME][2](mut)
        DMS_normalized = (DMS_score - DMS_MEAN) / DMS_STD
        return DMS_normalized
    
    # Create a single environment for sampling
    def make_env():
        return ProteinEnv(wt, fitness)
    
    env = make_env()
    
    # Load the trained model
    print(f"Loading model from {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH)
    print("Model loaded successfully!")
    
    # Sample variants
    variants = []
    print(f"\nSampling {NUM_VARIANTS} variants from the policy...")
    
    for i in range(NUM_VARIANTS):
        obs, _ = env.reset()
        done = False
        truncated = False
        
        # Run episode until truncated (6 steps)
        while not truncated:
            action, _states = model.predict(obs, deterministic=False)  # Use deterministic=False for sampling
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Extract final sequence
        final_seq = env.idxs_to_letters(env.state)
        variants.append(final_seq)
        print(f"Variant {i+1}/{NUM_VARIANTS}: {final_seq}")
    
    # Compute fitness for each variant
    print("\n" + "="*80)
    print("Computing fitness scores...")
    print("="*80)
    
    results = []
    for i, variant_seq in enumerate(variants):
        predicted_fitness = fitness(variant_seq)
        results.append({
            'variant_id': i + 1,
            'sequence': variant_seq,
            'predicted fitness (DMS normalized)': predicted_fitness
        })
        print(f"\nVariant {i+1}:")
        print(f"  Sequence: {variant_seq}")
        print(f"  Fitness: {predicted_fitness:.4f}")
    
    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Save results to CSV
    results_df = pd.DataFrame(results)
    os.makedirs(f'{ROOT}/generation/{MODEL_TYPE}', exist_ok=True)
    output_path = f'{ROOT}/generation/{MODEL_TYPE}/{formatted_date}_sampled_variants.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to {output_path}")
    print(f"{'='*80}")


