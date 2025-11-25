# demos/random_agent_demo.py
import sys
import os
import pygame
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.custom_env import HeartHealthEnvironment

def record_random_agent_demo():
    """Record a demo video with random agent actions"""
    print("Recording random agent demo...")
    
    env = HeartHealthEnvironment(render_mode="human")
    obs = env.reset(seed=42)
    
    running = True
    step = 0
    max_steps = 50  # Short demo for video
    
    while running and step < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step}: Action={action}, Reward={reward:.2f}, Risk={info['current_risk']:.1f}%")
        
        step += 1
        if terminated or truncated:
            running = False
        
        # Add small delay to make demo watchable
        pygame.time.delay(200)
    
    env.close()
    print("Random agent demo completed!")

if __name__ == "__main__":
    record_random_agent_demo()