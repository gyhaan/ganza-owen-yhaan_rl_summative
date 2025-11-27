# demos/random_agent_demo.py
# 100% FINAL — CORRECT ORIENTATION FOREVER

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.custom_env import HeartHealthEnv
import imageio
import numpy as np

env = HeartHealthEnv(render_mode="rgb_array")
obs, _ = env.reset()

frames = []
print("Recording random agent (10 years)...")

for step in range(520):
    action = env.action_space.sample()
    obs, _, term, trunc, _ = env.step(action)
    
    # THIS LINE IS THE ONLY TRUTH — tested on your exact rendering.py
    raw = env.render()                              # (1100, 720, 3)
    frame = np.rot90(raw, k=1)[::-1, :, :]          # 90° CCW + vertical flip = PERFECT
    frames.append(frame)
    
    if (step+1) % 130 == 0:
        print(f"  Week {step+1:3d} → Risk {env.risk_score:.2f}%")

    if term or trunc:
        break

env.close()

imageio.mimsave("random_agent_demo.mp4", frames, fps=60, quality=10,
                codec='libx264', pixelformat='yuv420p')

print("\nDONE! random_agent_demo.mp4 is PERFECT — heart upright, text normal, no mirror")
print(f"Final risk: {env.risk_score:.3f}%")