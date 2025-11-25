import argparse
import sys
import os
import glob
from training.dqn_training import test_trained_dqn
from training.reinforce_training import test_trained_reinforce
from training.a2c_training import test_trained_a2c
from training.ppo_training import test_trained_ppo

def main():
    parser = argparse.ArgumentParser(description="Heart Health RL Environment")
    parser.add_argument("--mode", choices=["test", "visualize", "train", "run_model"], 
                       default="test", help="Run mode")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--model", type=str, help="Path to model file to run")
    parser.add_argument("--algorithm", choices=["dqn", "reinforce", "a2c", "ppo"], 
                       help="Algorithm type for model")
    parser.add_argument("--episodes", type=int, default=5, help="Number of test episodes")
    parser.add_argument("--list-models", action="store_true", help="List all available models")
    
    args = parser.parse_args()
    
    if args.mode == "test":
        test_environment()
    elif args.mode == "visualize":
        visualize_environment()
    elif args.mode == "run_model":
        if args.list_models:
            list_available_models()
        elif args.model:
            run_model(args.model, args.algorithm, args.episodes, args.render)
        else:
            print("❌ Please specify --model or use --list-models to see available models")
    else:
        print("Training mode not implemented yet - use training scripts directly")

def test_environment():
    """Test basic environment functionality"""
    from environment.custom_env import HeartHealthEnvironment
    
    env = HeartHealthEnvironment(render_mode=None)
    
    print("🧪 Testing Heart Health Environment...")
    obs, info = env.reset(seed=42)
    print(f"✅ Initial risk score: {obs[7]:.1f}%")
    
    # Test a few steps
    for step in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step}: Reward={reward:.2f}, Risk={info['current_risk']:.1f}%")
    
    env.close()
    print("✅ Environment test passed!")

def visualize_environment():
    """Run environment with visualization"""
    from environment.custom_env import HeartHealthEnvironment
    
    env = HeartHealthEnvironment(render_mode="human")
    obs, info = env.reset(seed=123)
    
    print("🎮 Starting visualization... Close window to stop.")
    
    step = 0
    running = True
    while running and step < 20:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        
        if terminated or truncated:
            running = False
    
    env.close()

def list_available_models():
    """List all trained models available"""
    print("📁 Available Trained Models:")
    print("="*60)
    
    algorithms = ["dqn", "reinforce", "a2c", "ppo"]
    
    for algo in algorithms:
        model_files = glob.glob(f"models/{algo}/**/*.pth", recursive=True)
        model_files.extend(glob.glob(f"models/{algo}/**/*.zip", recursive=True))
        
        if model_files:
            print(f"\n🔧 {algo.upper()} Models:")
            for model_file in sorted(model_files):
                config_name = os.path.basename(os.path.dirname(model_file))
                model_name = os.path.basename(model_file)
                print(f"   • {config_name:.<25} {model_name}")

def run_model(model_path, algorithm=None, episodes=5, render=True):
    """Run a specific trained model"""
    print(f"🚀 Running model: {model_path}")
    
    # Auto-detect algorithm if not specified
    if algorithm is None:
        if "dqn" in model_path.lower() or model_path.endswith(".pth"):
            algorithm = "dqn"
        elif "reinforce" in model_path.lower():
            algorithm = "reinforce" 
        elif "a2c" in model_path.lower():
            algorithm = "a2c"
        elif "ppo" in model_path.lower() or model_path.endswith(".zip"):
            algorithm = "ppo"
        else:
            print("❓ Could not auto-detect algorithm. Please specify --algorithm")
            return
    
    print(f"🔧 Algorithm: {algorithm}")
    print(f"📊 Episodes: {episodes}")
    print(f"👀 Rendering: {'Yes' if render else 'No'}")
    print("-" * 50)
    
    try:
        if algorithm == "dqn":
            rewards, risks = test_trained_dqn(
                model_path=model_path,
                episodes=episodes,
                render=render,
                deterministic=True
            )
        elif algorithm == "reinforce":
            rewards, risks = test_trained_reinforce(
                model_path=model_path,
                episodes=episodes,
                render=render
            )
        elif algorithm == "a2c":
            rewards, risks = test_trained_a2c(
                model_path=model_path,
                episodes=episodes,
                render=render
            )
        elif algorithm == "ppo":
            rewards, risks, lengths = test_trained_ppo(
                model_path=model_path,
                episodes=episodes,
                render=render,
                deterministic=True
            )
        else:
            print(f"❌ Unsupported algorithm: {algorithm}")
            return
        
        # Show results
        print(f"\n📈 Test Results:")
        print(f"   Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"   Average Final Risk: {np.mean(risks):.1f}% ± {np.std(risks):.1f}%")
        print(f"   Best Reward: {np.max(rewards):.2f}")
        print(f"   Worst Reward: {np.min(rewards):.2f}")
        print(f"   Success Rate (Risk < 20%): {np.mean(np.array(risks) < 20) * 100:.1f}%")
        
    except Exception as e:
        print(f"❌ Error running model: {e}")
        print("💡 Make sure the model file exists and the algorithm is correct")

if __name__ == "__main__":
    # Add numpy import at the top if needed for results
    import numpy as np
    main()