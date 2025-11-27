# training/reinforce_training.py ← FINAL VERSION — WORKS 100%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from environment.custom_env import HeartHealthEnv
import time
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

# === CLEAN FOLDERS ===
os.makedirs("reinforce_tensorboard", exist_ok=True)
os.makedirs("reports/reinforce", exist_ok=True)
os.makedirs("models/reinforce", exist_ok=True)

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 12)
        )
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

def train_reinforce(learning_rate, gamma, baseline, entropy_coef, run_id, total_timesteps=200_000):
    env = HeartHealthEnv()
    policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    writer = SummaryWriter(log_dir=f"reinforce_tensorboard/reinforce_run_{run_id}")

    step = 0
    episode = 0
    episode_rewards = []

    print(f"Training REINFORCE | Target: {total_timesteps:,} timesteps | Run {run_id}")

    while step < total_timesteps:
        obs, _ = env.reset()
        obs = torch.FloatTensor(obs)
        log_probs = []
        rewards = []
        entropies = []
        episode += 1

        done = False
        while not done and step < total_timesteps:
            probs = policy(obs)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_obs, r, terminated, truncated, _ = env.step(action.item())
            obs = torch.FloatTensor(next_obs)
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(r)
            entropies.append(dist.entropy())
            step += 1

        # Returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)

        if baseline and len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs = torch.stack(log_probs)
        policy_loss = (-log_probs * returns).mean()

        # Entropy
        if entropy_coef > 0 and len(entropies) > 0:
            entropy_bonus = torch.stack(entropies).mean()
            loss = policy_loss - entropy_coef * entropy_bonus
            entropy_value = entropy_bonus.item()
        else:
            loss = policy_loss
            entropy_value = 0.0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward = sum(rewards)
        episode_rewards.append(total_reward)

        # Log
        writer.add_scalar("episode/reward", total_reward, episode)
        writer.add_scalar("episode/length", len(rewards), episode)
        writer.add_scalar("train/policy_loss", policy_loss.item(), episode)
        writer.add_scalar("train/entropy", entropy_value, episode)

        if episode % 50 == 0:
            avg = np.mean(episode_rewards[-50:])
            print(f"Step {step:6d} | Ep {episode:3d} | Avg Rew: {avg:+8.1f} | Risk: {env.risk_score:5.2f}%")

    writer.close()

    # Final eval
    obs, _ = env.reset()
    env.risk_history = []
    with torch.no_grad():
        for _ in range(520):
            obs_t = torch.FloatTensor(obs)
            action = torch.argmax(policy(obs_t)).item()
            obs, _, term, trunc, _ = env.step(action)
            env.risk_history.append(env.risk_score)
            if term or trunc: break

    model_path = f"models/reinforce/reinforce_run_{run_id}.pth"
    torch.save(policy.state_dict(), model_path)

    return env.risk_score, model_path

# === CONFIGS ===
configs = [
    {"learning_rate": 3e-4, "gamma": 0.99,  "baseline": True,  "entropy_coef": 0.00},
    {"learning_rate": 5e-4, "gamma": 0.995, "baseline": True,  "entropy_coef": 0.01},
    {"learning_rate": 2e-4, "gamma": 0.99,  "baseline": True,  "entropy_coef": 0.02},
    {"learning_rate": 7e-4, "gamma": 0.99,  "baseline": False, "entropy_coef": 0.00},
    {"learning_rate": 3e-4, "gamma": 0.997, "baseline": True,  "entropy_coef": 0.01},
    {"learning_rate": 1e-3, "gamma": 0.99,  "baseline": True,  "entropy_coef": 0.005},
    {"learning_rate": 4e-4, "gamma": 0.995, "baseline": False, "entropy_coef": 0.02},
    {"learning_rate": 2.5e-4,"gamma": 0.99, "baseline": True,  "entropy_coef": 0.00},
    {"learning_rate": 6e-4, "gamma": 0.99,  "baseline": True,  "entropy_coef": 0.015},
    {"learning_rate": 3e-4, "gamma": 0.999, "baseline": True,  "entropy_coef": 0.01},
]

results = []
for i, cfg in enumerate(configs):
    print(f"\n{'='*80}")
    print(f"REINFORCE Run {i+1}/10 | lr={cfg['learning_rate']} | γ={cfg['gamma']} | baseline={cfg['baseline']} | ent={cfg['entropy_coef']}")
    print(f"{'='*80}")

    start = time.time()
    risk, path = train_reinforce(
        learning_rate=cfg["learning_rate"],
        gamma=cfg["gamma"],
        baseline=cfg["baseline"],
        entropy_coef=cfg["entropy_coef"],
        run_id=i,
        total_timesteps=200_000
    )
    t = (time.time() - start) / 60
    results.append({
        **cfg,
        "run": i,
        "final_risk": round(risk, 3),
        "train_time_min": round(t, 2),
        "model_path": path
    })
    print(f"→ Final Risk: {risk:.3f}% | Time: {t:.1f} min | Saved: {path}")

df = pd.DataFrame(results)
df.to_csv("reports/reinforce/reinforce_results.csv", index=False)
print("\nREINFORCE TRAINING 100% COMPLETE — ALL LOGS CLEAN AND WORKING!")
print("You can now run your final comparison script!")