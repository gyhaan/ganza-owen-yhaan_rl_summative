# training/dqn_training.py  ←←← FULLY FIXED VERSION ←←←
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from environment.custom_env import HeartHealthEnv
import time
import pandas as pd

os.makedirs("reports/dqn", exist_ok=True)
os.makedirs("models/dqn", exist_ok=True)

hyperparams = [
    {"learning_rate": 1e-3, "batch_size": 64,   "gamma": 0.99,  "buffer_size": 100_000},
    {"learning_rate": 5e-4, "batch_size": 128,  "gamma": 0.99,  "buffer_size": 100_000},
    {"learning_rate": 1e-3, "batch_size": 64,   "gamma": 0.995, "buffer_size": 500_000},
    {"learning_rate": 5e-4, "batch_size": 128,  "gamma": 0.995, "buffer_size": 500_000},
    {"learning_rate": 1e-4, "batch_size": 256,  "gamma": 0.99,  "buffer_size": 100_000},
    {"learning_rate": 1e-3, "batch_size": 128,  "gamma": 0.99,  "tau": 0.005},
    {"learning_rate": 5e-4, "batch_size": 64,   "gamma": 0.999, "train_freq": 8},
    {"learning_rate": 1e-3, "batch_size": 256,  "gamma": 0.99,  "exploration_fraction": 0.3},
    {"learning_rate": 3e-4, "batch_size": 128,  "gamma": 0.995, "buffer_size": 200_000},
    {"learning_rate": 8e-4, "batch_size": 64,   "gamma": 0.99,  "target_update_interval": 500},
    {"learning_rate": 1e-3, "batch_size": 128,  "gamma": 0.99,  "gradient_steps": 4},
    {"learning_rate": 5e-4, "batch_size": 256,  "gamma": 0.995},
]

results = []
for i, hp in enumerate(hyperparams):
    print(f"\n=== DQN Run {i+1}/{len(hyperparams)} | {hp} ===")
    
    raw_env = HeartHealthEnv()           # ← keep original reference
    env = Monitor(raw_env)                # ← wrapped for logging only

    model = DQN("MlpPolicy", env, verbose=0, tensorboard_log="./dqn_tensorboard/", seed=42,
                policy_kwargs={"net_arch": [128, 128]}, **hp)

    start = time.time()
    model.learn(total_timesteps=150_000, tb_log_name=f"run_{i}")
    train_time = (time.time() - start) / 60

    # ←←← NOW we use the original unwrapped env for evaluation
    obs, _ = raw_env.reset()
    raw_env.risk_history = []
    for _ in range(520):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = raw_env.step(action)
        raw_env.risk_history.append(raw_env.risk_score)
        if term or trunc: break
    final_risk = raw_env.risk_score

    model_path = f"models/dqn/dqn_heart_run_{i}"
    model.save(model_path)

    row = {**hp, "run": i, "final_risk": round(final_risk, 3), "train_time_minutes": round(train_time, 2), "model_path": model_path}
    results.append(row)
    print(f"DQN Run {i} → Final Risk: {final_risk:.2f}% | Time: {train_time:.1f} min")

df = pd.DataFrame(results)
csv_path = "reports/dqn/dqn_results.csv"
df.to_csv(csv_path, index=False)
print(f"\nDQN TRAINING COMPLETE → {csv_path} saved!")