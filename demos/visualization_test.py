# demos/visualization_test.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.custom_env import HeartHealthEnvironment

def test_visualization():
    """Test the environment with visualization"""
    print("Testing environment visualization...")
    
    env = HeartHealthEnvironment(render_mode="human")
    obs = env.reset(seed=123)
    
    print("Environment started with visualization.")
    print("Close the Pygame window to continue...")
    
    # Run a few steps to see the visualization
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step}: Reward = {reward:.2f}, Risk = {info['current_risk']:.1f}%")
        
        if terminated or truncated:
            break
    
    env.close()
    print("Visualization test completed!")

if __name__ == "__main__":
    test_visualization()