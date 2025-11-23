# demos/random_agent_demo.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import pygame
import numpy as np
from environment.custom_env import HeartHealthEnv
import imageio

# Initialize pygame for rendering
pygame.init()

env = HeartHealthEnv(render_mode="rgb_array")
env.risk_history = []

# Create video writer
writer = imageio.get_writer("random_agent_demo.mp4", fps=10, codec="libx264", pixelformat="yuv420p")

obs, _ = env.reset()
print("Starting random agent demo... (500 steps)")

for step in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.risk_history.append(env.risk_score)

    # THIS IS THE FIX: convert Pygame surface → NumPy array
    frame = env.render()                    # returns pygame Surface
    frame_np = pygame.surfarray.array3d(frame)   # convert to numpy
    frame_np = np.transpose(frame_np, (1, 0, 2)) # fix orientation (width, height, channels)
    frame_np = np.rot90(frame_np, k=3)            # rotate correctly if needed (optional, looks better)
    
    writer.append_data(frame_np)

    print(f"Step {step+1:3d} | Action {action:2d} | Risk {env.risk_score:5.2f}% | Reward {reward:+6.2f}")

    if terminated or truncated:
        obs, _ = env.reset()
        env.risk_history = [env.risk_score]

writer.close()
env.close()
pygame.quit()

print("\nSUCCESS! → random_agent_demo.mp4 created (assignment requirement fulfilled)")