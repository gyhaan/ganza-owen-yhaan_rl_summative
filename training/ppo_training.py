# training/ppo_training.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from environment.custom_env import HeartHealthEnv
import time
import pandas as pd

os.makedirs("reports/ppo", exist_ok=True)
os.makedirs("models/pg", exist_ok=True)

hyperparams = [
    {"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64,   "gamma": 0.99},
    {"learning_rate": 1e-3, "n_steps": 1024, "batch_size": 128,  "gamma": 0.99},
    {"learning_rate": 5e-4, "n_steps": 2048, "batch_size": 128,  "gamma": 0.995},
    {"learning_rate": 2e-4, "n_steps": 4096, "batch_size": 256,  "gamma": 0.99},
    {"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64,   "gae_lambda": 0.95},
    {"learning_rate": 3e-4, "n_steps": 1024, "batch_size": 64,   "clip_range": 0.2},
    {"learning_rate": 5e-4, "n_steps": 2048, "batch_size": 128,  "ent_coef": 0.01},
    {"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 256,  "n_epochs": 10},
    {"learning_rate": 1e-4, "n_steps": 4096, "batch_size": 128,  "gamma": 0.999},
    {"learning_rate": 8e-4, "n_steps": 1024, "batch_size": 64,   "gamma": 0.99},
    {"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64,   "clip_range": 0.3},
    {"learning_rate": 2.5e-4,"n_steps": 2048, "batch_size": 128,  "gamma": 0.99},
    {"learning_rate": 3e-4, "n_steps": 3072, "batch_size": 96,   "gamma": 0.995},
    {"learning_rate": 4e-4, "n_steps": 2048, "batch_size": 64,   "ent_coef": 0.005},
    {"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 128,  "gae_lambda": 0.98},
]

results = []
for i, hp in enumerate(hyperparams):
    print(f"\n=== PPO Run {i+1}/{len(hyperparams)} ===")
    env = HeartHealthEnv()

    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log="./ppo_tensorboard/", seed=42, **hp)

    start = time.time()
    model.learn(total_timesteps=200_000, tb_log_name=f"run_{i}")
    train_time = (time.time() - start) / 60

    obs, _ = env.reset()
    env.risk_history = []
    for _ in range(520):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        env.risk_history.append(env.risk_score)
        if term or trunc: break
    final_risk = env.risk_score

    model_path = f"models/pg/ppo_heart_run_{i}"
    model.save(model_path)

    row = {**hp, "run": i, "final_risk": round(final_risk, 3), "train_time_minutes": round(train_time, 2), "model_path": model_path}
    results.append(row)
    print(f"PPO Run {i} → Final Risk: {final_risk:.2f}%")

df = pd.DataFrame(results)
csv_path = "reports/ppo/ppo_results.csv"
df.to_csv(csv_path, index=False)
print(f"\nPPO TRAINING COMPLETE → {csv_path} saved!")