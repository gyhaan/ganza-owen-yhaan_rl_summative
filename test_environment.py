import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.custom_env import HeartHealthEnvironment

def test_environment():
    """Test the basic environment functionality"""
    env = HeartHealthEnvironment(render_mode=None)
    
    print("Testing Heart Health Environment...")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    
    # Test reset (gymnasium returns (obs, info))
    obs, info = env.reset(seed=42)
    print(f"Initial observation: {obs}")
    print(f"Initial risk score: {obs[7]:.1f}%")
    
    # Test a few steps (gymnasium returns (obs, reward, terminated, truncated, info))
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nStep {step + 1}:")
        print(f"Action: {action}")
        print(f"Reward: {reward:.2f}")
        print(f"Risk Score: {info['current_risk']:.1f}%")
        print(f"Risk Reduction: {info['risk_reduction']:.1f}%")
        print(f"Terminated: {terminated}")
        
        if terminated or truncated:
            break
    
    env.close()
    print("\nEnvironment test completed successfully!")

if __name__ == "__main__":
    test_environment()