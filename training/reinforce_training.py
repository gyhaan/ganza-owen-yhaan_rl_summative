# training/reinforce_training.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from environment.custom_env import HeartHealthEnv
import time
import csv
import pandas as pd

os.makedirs("reports/reinforce", exist_ok=True)
os.makedirs("models/pg", exist_ok=True)

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

def train_reinforce(lr=3e-4, gamma=0.99, baseline=True, run_id=0):
    env = HeartHealthEnv()
    policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    all_rewards = []

    for episode in range(800):
        obs, _ = env.reset()
        obs = torch.FloatTensor(obs)
        log_probs = []
        rewards = []

        for _ in range(520):
            probs = policy(obs)
            action = torch.multinomial(probs, 1).item()
            log_prob = torch.log(probs[action])
            obs, r, term, trunc, _ = env.step(action)
            obs = torch.FloatTensor(obs)

            log_probs.append(log_prob)
            rewards.append(r)
            if term or trunc: break

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)
        if baseline and len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs = torch.stack(log_probs)
        policy_loss = (-log_probs * returns).mean()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        all_rewards.append(sum(rewards))

        if episode % 100 == 0:
            print(f"Ep {episode} | Avg Reward: {np.mean(all_rewards[-100:]):.2f}")

    # Final evaluation
    obs, _ = env.reset()
    env.risk_history = []
    for _ in range(520):
        obs_t = torch.FloatTensor(obs)
        action = policy(obs_t).multinomial(1).item()
        obs, _, term, trunc, _ = env.step(action)
        env.risk_history.append(env.risk_score)
        if term or trunc: break

    torch.save(policy.state_dict(), f"models/pg/reinforce_run_{run_id}.pth")
    return env.risk_score

# Run 10 configurations
configs = [
    (3e-4, 0.99, True), (1e-3, 0.99, True), (3e-4, 0.995, True),
    (5e-4, 0.99, False), (3e-4, 0.99, False), (1e-3, 0.995, True),
    (2e-4, 0.99, True), (3e-4, 0.997, True), (7e-4, 0.99, True),
    (3e-4, 0.99, True),
]

results = []
for i, (lr, gamma, baseline) in enumerate(configs):
    print(f"\n=== REINFORCE Run {i+1}/10 | lr={lr} gamma={gamma} baseline={baseline} ===")
    start = time.time()
    final_risk = train_reinforce(lr, gamma, baseline, i)
    time_min = (time.time() - start) / 60
    results.append({
        "run": i, "learning_rate": lr, "gamma": gamma, "baseline": baseline,
        "final_risk": round(final_risk, 3), "train_time_minutes": round(time_min, 2)
    })
    print(f"REINFORCE Run {i} → Final Risk: {final_risk:.2f}%")

df = pd.DataFrame(results)
df.to_csv("reports/reinforce/reinforce_results.csv", index=False)
print("\nREINFORCE COMPLETE → reports/reinforce/reinforce_results.csv saved")