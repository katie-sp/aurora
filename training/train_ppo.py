from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3 import PPO

from ppo.callbacks import *
from ppo.environments import ProteinEnv
from config import *
from oracles import ORACLE_REGISTRY

if __name__ == "__main__":
    with open(WT_PATH, 'r') as file:
        wt = file.readline().strip()

    def fitness(mut):
        DMS_score = ORACLE_REGISTRY[ORACLE_NAME][2](mut)
        DMS_normalized = (DMS_score - DMS_MEAN) / DMS_STD
        return DMS_normalized

    def make_env():
        return ProteinEnv(wt, fitness)

    vec_env = DummyVecEnv([make_env for _ in range(NUM_ENVS)])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=6, 
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        clip_range=0.2,
        verbose=1,
        device="cpu"
    )

    tqdm_cb = TQDMCallback(total_timesteps=TOTAL_TIMESTEPS, algo='PPO')
    logger_cb = ProteinRLLogger(check_freq=1)
    callback = CallbackList([tqdm_cb, logger_cb])
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    model.save(PPO_DIR + '/ppo_model')