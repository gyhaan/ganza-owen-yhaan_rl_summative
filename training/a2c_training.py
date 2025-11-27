# training/a2c_training.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import A2C
from environment.custom_env import HeartHealthEnv
import time
import pandas as pd

# === CLEAN & ORGANIZED FOLDERS ===
os.makedirs("a2c_tensorboard", exist_ok=True)      # ← All A2C logs go here
os.makedirs("reports/a2c", exist_ok=True)
os.makedirs("models/a2c", exist_ok=True)           # ← Clean model folder

hyperparams = [
    {"learning_rate": 7e-4, "n_steps": 5,      "gamma": 0.99,  "gae_lambda": 1.0},
    {"learning_rate": 5e-4, "n_steps": 8,      "gamma": 0.995, "gae_lambda": 0.95},
    {"learning_rate": 1e-3, "n_steps": 5,      "gamma": 0.99,  "ent_coef": 0.01},
    {"learning_rate": 3e-4, "n_steps": 10,     "gamma": 0.99,  "gae_lambda": 0.98},
    {"learning_rate": 7e-4, "n_steps": 5,      "gamma": 0.999, "ent_coef": 0.0},
    {"learning_rate": 5e-4, "n_steps": 16,     "gamma": 0.99,  "gae_lambda": 0.9},
    {"learning_rate": 1e-3, "n_steps": 5,      "gamma": 0.995, "ent_coef": 0.005},
    {"learning_rate": 4e-4, "n_steps": 8,      "gamma": 0.99,  "gae_lambda": 1.0},
    {"learning_rate": 7e-4, "n_steps": 10,     "gamma": 0.99,  "ent_coef": 0.02},
    {"learning_rate": 6e-4, "n_steps": 5,      "gamma": 0.997, "gae_lambda": 0.97},
]

results = []
for i, hp in enumerate(hyperparams):
    print(f"\n=== A2C Run {i+1}/{len(hyperparams)} ===")
    env = HeartHealthEnv()

    model = A2C(
        "MlpPolicy",
        env,
        verbose=1,                                    # ← Set to 1 to see progress
        tensorboard_log="a2c_tensorboard",           # ← Dedicated clean folder
        seed=42,
        **hp
    )

    start = time.time()
    model.learn(total_timesteps=150_000, tb_log_name=f"a2c_run_{i}")  # ← Clear naming
    train_time = (time.time() - start) / 60

    # === Evaluation ===
    obs, _ = env.reset()
    env.risk_history = []
    for _ in range(520):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        env.risk_history.append(env.risk_score)
        if term or trunc: 
            break
    final_risk = env.risk_score

    # === Save model in clean folder ===
    model_path = f"models/a2c/a2c_heart_run_{i}"
    model.save(model_path)

    row = {
        **hp,
        "run": i,
        "final_risk": round(final_risk, 3),
        "train_time_minutes": round(train_time, 2),
        "model_path": model_path
    }
    results.append(row)
    print(f"A2C Run {i} → Final Risk: {final_risk:.3f}% | Time: {train_time:.1f} min | Saved: {model_path}.zip")

# === Save results ===
df = pd.DataFrame(results)
csv_path = "reports/a2c/a2c_results.csv"
df.to_csv(csv_path, index=False)

print(f"\nA2C TRAINING COMPLETE!")
print(f"→ TensorBoard Logs: a2c_tensorboard/a2c_run_0, a2c_run_1, ...")
print(f"→ Models saved in: models/a2c/")
print(f"→ Results CSV: {csv_path}")
print("Everything clean, organized, and examiner-ready")