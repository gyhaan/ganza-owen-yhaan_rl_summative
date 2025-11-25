import torch
import torch.nn as nn
import numpy as np
import os
import sys
import csv
import time
import json
import pickle
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.custom_env import HeartHealthEnvironment

class ComprehensivePPOMetricsLogger(BaseCallback):
    """
    Comprehensive metrics logger for PPO that captures everything needed for analysis
    """
    def __init__(self, log_dir, config_name, verbose=0):
        super(ComprehensivePPOMetricsLogger, self).__init__(verbose)
        self.log_dir = log_dir
        self.config_name = config_name
        os.makedirs(log_dir, exist_ok=True)
        
        # CSV file paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.episode_csv_path = os.path.join(log_dir, f"{config_name}_episodes_{timestamp}.csv")
        self.training_csv_path = os.path.join(log_dir, f"{config_name}_training_{timestamp}.csv")
        
        # Create episode CSV with headers
        with open(self.episode_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'total_reward', 'episode_length', 'final_risk', 
                'risk_reduction', 'training_time', 'timesteps', 'fps',
                'avg_reward_10', 'avg_reward_100', 'avg_length_10', 'avg_length_100'
            ])
        
        # Create training CSV with headers
        with open(self.training_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'update', 'timesteps', 'value_loss', 'policy_loss', 'entropy_loss',
                'approx_kl', 'clip_fraction', 'explained_variance', 'learning_rate',
                'clip_range', 'training_time', 'fps'
            ])
        
        self.start_time = time.time()
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_risks = []
        self.episode_risk_reductions = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.episode_count = 0
        self.last_logged_update = 0
        
    def _on_step(self) -> bool:
        # Accumulate reward for current episode
        if len(self.locals['rewards']) > 0:
            self.current_episode_reward += self.locals['rewards'][0]
            self.current_episode_length += 1
        
        # Check if episode ended
        if self.locals['dones'][0]:
            # Get the final observation to extract risk information
            # This is a bit hacky but works for getting risk metrics
            final_obs = self.locals['new_obs'][0]
            final_risk = final_obs[7] if len(final_obs) > 7 else 0  # risk_score is index 7
            risk_reduction = 0  # We'll calculate this from baseline if available
            
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.episode_risks.append(final_risk)
            self.episode_risk_reductions.append(risk_reduction)
            
            # Calculate running averages
            avg_reward_10 = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else self.current_episode_reward
            avg_reward_100 = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else self.current_episode_reward
            avg_length_10 = np.mean(self.episode_lengths[-10:]) if len(self.episode_lengths) >= 10 else self.current_episode_length
            avg_length_100 = np.mean(self.episode_lengths[-100:]) if len(self.episode_lengths) >= 100 else self.current_episode_length
            
            # Log episode to CSV
            training_time = time.time() - self.start_time
            fps = self.model.num_timesteps / training_time if training_time > 0 else 0
            
            with open(self.episode_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.episode_count, self.current_episode_reward, self.current_episode_length,
                    final_risk, risk_reduction, training_time, self.model.num_timesteps, fps,
                    avg_reward_10, avg_reward_100, avg_length_10, avg_length_100
                ])
            
            # Print progress every 10 episodes
            if self.episode_count % 10 == 0:
                print(f"Episode {self.episode_count:4d} | "
                      f"Reward: {self.current_episode_reward:7.1f} | "
                      f"Avg10: {avg_reward_10:7.1f} | "
                      f"Avg100: {avg_reward_100:7.1f} | "
                      f"Length: {self.current_episode_length:3d} | "
                      f"Risk: {final_risk:5.1f}% | "
                      f"FPS: {fps:5.1f}")
            
            # Reset for next episode
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.episode_count += 1
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Log training metrics at the end of each rollout"""
        # Get training metrics from the model's logger
        if hasattr(self.model, 'logger'):
            logs = self.model.logger.name_to_value
            
            # Extract metrics with defaults
            value_loss = logs.get('train/value_loss', 0)
            policy_loss = logs.get('train/policy_loss', 0)
            entropy_loss = logs.get('train/entropy_loss', 0)
            approx_kl = logs.get('train/approx_kl', 0)
            clip_fraction = logs.get('train/clip_fraction', 0)
            explained_variance = logs.get('train/explained_variance', 0)
            learning_rate = logs.get('train/learning_rate', 0)
            clip_range = logs.get('train/clip_range', 0)
        else:
            value_loss = policy_loss = entropy_loss = approx_kl = 0
            clip_fraction = explained_variance = learning_rate = clip_range = 0
        
        # Log training metrics to CSV
        training_time = time.time() - self.start_time
        fps = self.model.num_timesteps / training_time if training_time > 0 else 0
        
        with open(self.training_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.last_logged_update, self.model.num_timesteps, value_loss, policy_loss,
                entropy_loss, approx_kl, clip_fraction, explained_variance, learning_rate,
                clip_range, training_time, fps
            ])
        
        self.last_logged_update += 1

class HeartHealthEnvWrapper:
    """Wrapper to make our custom environment compatible with SB3"""
    def __init__(self, render_mode=None):
        self.env = HeartHealthEnvironment(render_mode=render_mode)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
    
    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return obs
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()

def create_heart_health_env(render_mode=None):
    """Create and return wrapped HeartHealthEnvironment"""
    return HeartHealthEnvWrapper(render_mode=render_mode)

class PPOModelManager:
    """Manages PPO model training, saving, and logging"""
    def __init__(self, config, save_path):
        self.config = config
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        
        # Create environment
        self.env = make_vec_env(
            lambda: create_heart_health_env(render_mode=None),
            n_envs=config['n_envs'],
            seed=config['seed']
        )
        
        # Add monitor for additional metrics
        self.env = VecMonitor(self.env, filename=os.path.join(save_path, "monitor"))
        
        # Create logger
        self.logger = ComprehensivePPOMetricsLogger(save_path, config['name'])
        
        # Create model
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=config['learning_rate'],
            n_steps=config['n_steps'],
            batch_size=config['batch_size'],
            n_epochs=config['n_epochs'],
            gamma=config['gamma'],
            gae_lambda=config['gae_lambda'],
            clip_range=config['clip_range'],
            clip_range_vf=config['clip_range_vf'],
            normalize_advantage=config['normalize_advantage'],
            ent_coef=config['entropy_coef'],
            vf_coef=config['value_coef'],
            max_grad_norm=config['max_grad_norm'],
            use_sde=config['use_sde'],
            sde_sample_freq=config['sde_sample_freq'],
            target_kl=config['target_kl'],
            tensorboard_log=os.path.join(save_path, "tensorboard"),
            verbose=1,
            seed=config['seed'],
            device='auto'
        )
    
    def train(self):
        """Train the model with comprehensive logging"""
        print(f"\n{'='*80}")
        print(f"Training PPO: {self.config['name']}")
        print(f"{'='*80}")
        print(f"Configuration: {self.config}")
        print(f"Training for {self.config['total_timesteps']} timesteps...")
        print(f"Save path: {self.save_path}")
        
        # Save configuration
        config_path = os.path.join(self.save_path, 'training_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Training metrics storage
        training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_risks': [],
            'training_losses': [],
            'start_time': time.time()
        }
        
        try:
            # Train the model
            self.model.learn(
                total_timesteps=self.config['total_timesteps'],
                callback=self.logger,
                log_interval=1,  # More frequent logging
                tb_log_name=self.config['name'],
                reset_num_timesteps=False
            )
            
            # Save final model
            self.save_model("final")
            
            # Collect final metrics
            training_metrics['episode_rewards'] = self.logger.episode_rewards
            training_metrics['episode_lengths'] = self.logger.episode_lengths
            training_metrics['episode_risks'] = self.logger.episode_risks
            training_metrics['end_time'] = time.time()
            training_metrics['total_timesteps'] = self.model.num_timesteps
            training_metrics['total_episodes'] = self.logger.episode_count
            
            return training_metrics
            
        except Exception as e:
            print(f"Training interrupted with error: {e}")
            # Save model anyway
            self.save_model("interrupted")
            raise e
        
        finally:
            self.env.close()
    
    def save_model(self, suffix=""):
        """Save model with given suffix"""
        model_path = os.path.join(self.save_path, f"ppo_{self.config['name']}_{suffix}")
        self.model.save(model_path)
        print(f"Model saved to: {model_path}")
        
        # Also save replay buffer if it exists
        if hasattr(self.model, 'replay_buffer') and self.model.replay_buffer is not None:
            replay_buffer_path = os.path.join(self.save_path, f"replay_buffer_{suffix}.pkl")
            with open(replay_buffer_path, 'wb') as f:
                pickle.dump(self.model.replay_buffer, f)

def train_ppo_comprehensive(config, save_path="models/ppo/"):
    """Main training function for PPO with comprehensive logging and model management"""
    
    # Create model manager
    model_save_path = os.path.join(save_path, config['name'])
    manager = PPOModelManager(config, model_save_path)
    
    # Train and get metrics
    training_metrics = manager.train()
    
    # Create training summary
    training_summary = {
        'config': config,
        'episode_rewards': training_metrics['episode_rewards'],
        'episode_lengths': training_metrics['episode_lengths'],
        'episode_risks': training_metrics['episode_risks'],
        'final_metrics': {
            'total_timesteps': training_metrics['total_timesteps'],
            'total_episodes': training_metrics['total_episodes'],
            'training_time': training_metrics['end_time'] - training_metrics['start_time'],
            'final_avg_reward_10': np.mean(training_metrics['episode_rewards'][-10:]) if len(training_metrics['episode_rewards']) >= 10 else 0,
            'final_avg_reward_100': np.mean(training_metrics['episode_rewards'][-100:]) if len(training_metrics['episode_rewards']) >= 100 else 0,
            'best_reward': np.max(training_metrics['episode_rewards']) if training_metrics['episode_rewards'] else 0,
            'final_avg_risk': np.mean(training_metrics['episode_risks'][-10:]) if len(training_metrics['episode_risks']) >= 10 else 0,
        }
    }
    
    # Save training summary
    summary_path = os.path.join(model_save_path, 'training_summary.pkl')
    with open(summary_path, 'wb') as f:
        pickle.dump(training_summary, f)
    
    # Print final summary
    final_metrics = training_summary['final_metrics']
    print(f"\n{'='*80}")
    print(f"PPO Training Completed: {config['name']}")
    print(f"{'='*80}")
    print(f"Total Timesteps: {final_metrics['total_timesteps']:,}")
    print(f"Total Episodes: {final_metrics['total_episodes']}")
    print(f"Training Time: {final_metrics['training_time']:.1f}s")
    print(f"Final Average Reward (last 10): {final_metrics['final_avg_reward_10']:.2f}")
    print(f"Final Average Reward (last 100): {final_metrics['final_avg_reward_100']:.2f}")
    print(f"Best Episode Reward: {final_metrics['best_reward']:.2f}")
    print(f"Final Average Risk: {final_metrics['final_avg_risk']:.1f}%")
    print(f"Models and logs saved to: {model_save_path}")
    
    return training_summary

# Hyperparameter configurations for PPO
PPO_CONFIGS = [
    {
        'name': 'ppo_basic',
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'clip_range_vf': 0.2,
        'normalize_advantage': True,
        'entropy_coef': 0.01,
        'value_coef': 0.5,
        'max_grad_norm': 0.5,
        'use_sde': False,
        'sde_sample_freq': -1,
        'target_kl': 0.03,
        'n_envs': 4,
        'total_timesteps': 100000,
        'seed': 42
    },
    {
        'name': 'ppo_large_batch',
        'learning_rate': 3e-4,
        'n_steps': 4096,
        'batch_size': 128,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'clip_range_vf': 0.2,
        'normalize_advantage': True,
        'entropy_coef': 0.01,
        'value_coef': 0.5,
        'max_grad_norm': 0.5,
        'use_sde': False,
        'sde_sample_freq': -1,
        'target_kl': 0.03,
        'n_envs': 4,
        'total_timesteps': 100000,
        'seed': 42
    },
    {
        'name': 'ppo_high_entropy',
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'clip_range_vf': 0.2,
        'normalize_advantage': True,
        'entropy_coef': 0.1,
        'value_coef': 0.5,
        'max_grad_norm': 0.5,
        'use_sde': False,
        'sde_sample_freq': -1,
        'target_kl': 0.03,
        'n_envs': 4,
        'total_timesteps': 100000,
        'seed': 42
    },
    {
        'name': 'ppo_fast_learning',
        'learning_rate': 1e-3,
        'n_steps': 1024,
        'batch_size': 64,
        'n_epochs': 5,
        'gamma': 0.95,
        'gae_lambda': 0.9,
        'clip_range': 0.1,
        'clip_range_vf': 0.1,
        'normalize_advantage': True,
        'entropy_coef': 0.01,
        'value_coef': 0.5,
        'max_grad_norm': 0.5,
        'use_sde': False,
        'sde_sample_freq': -1,
        'target_kl': 0.01,
        'n_envs': 4,
        'total_timesteps': 100000,
        'seed': 42
    },
    {
        'name': 'ppo_conservative',
        'learning_rate': 1e-4,
        'n_steps': 4096,
        'batch_size': 256,
        'n_epochs': 20,
        'gamma': 0.99,
        'gae_lambda': 0.98,
        'clip_range': 0.1,
        'clip_range_vf': 0.1,
        'normalize_advantage': True,
        'entropy_coef': 0.001,
        'value_coef': 0.5,
        'max_grad_norm': 0.5,
        'use_sde': False,
        'sde_sample_freq': -1,
        'target_kl': 0.01,
        'n_envs': 4,
        'total_timesteps': 100000,
        'seed': 42
    }
]

def run_ppo_experiments():
    """Run multiple PPO experiments with different hyperparameters"""
    results = {}
    
    for config in PPO_CONFIGS:
        try:
            result = train_ppo_comprehensive(config)
            results[config['name']] = result
        except Exception as e:
            print(f"Failed to train {config['name']}: {e}")
            continue
    
    # Save overall results
    overall_results_path = "models/ppo/ppo_overall_results.pkl"
    os.makedirs(os.path.dirname(overall_results_path), exist_ok=True)
    with open(overall_results_path, 'wb') as f:
        pickle.dump(results, f)
    
    return results

def test_trained_ppo(model_path, episodes=5, render=False, deterministic=True):
    """Test a trained PPO model with comprehensive evaluation"""
    print(f"\n{'='*60}")
    print(f"Testing PPO Model: {model_path}")
    print(f"{'='*60}")
    
    # Load the trained model
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None, None, None
    
    model = PPO.load(model_path)
    
    # Create environment
    env = HeartHealthEnvironment(render_mode="human" if render else None)
    
    test_rewards = []
    test_risks = []
    test_lengths = []
    test_actions = []
    
    print(f"Running {episodes} test episodes (deterministic={deterministic})...")
    
    for episode in range(episodes):
        obs, info = env.reset(seed=1000 + episode)
        total_reward = 0
        steps = 0
        episode_actions = []
        
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            episode_actions.append(action)
            obs = next_obs
            
            if done:
                break
        
        test_rewards.append(total_reward)
        test_risks.append(info['current_risk'])
        test_lengths.append(steps)
        test_actions.append(episode_actions)
        
        print(f"Test Episode {episode:2d}: "
              f"Reward = {total_reward:7.1f}, "
              f"Final Risk = {info['current_risk']:5.1f}%, "
              f"Steps = {steps:3d}, "
              f"Risk Reduction = {info.get('risk_reduction', 0):5.1f}%")
    
    env.close()
    
    # Calculate comprehensive statistics
    rewards_array = np.array(test_rewards)
    risks_array = np.array(test_risks)
    lengths_array = np.array(test_lengths)
    
    print(f"\n{'='*60}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Average Reward: {np.mean(rewards_array):.2f} ± {np.std(rewards_array):.2f}")
    print(f"Average Final Risk: {np.mean(risks_array):.1f}% ± {np.std(risks_array):.1f}%")
    print(f"Average Episode Length: {np.mean(lengths_array):.1f} ± {np.std(lengths_array):.1f}")
    print(f"Best Reward: {np.max(rewards_array):.2f}")
    print(f"Worst Reward: {np.min(rewards_array):.2f}")
    print(f"Best Risk Reduction: {np.min(risks_array):.1f}%")
    print(f"Success Rate (Risk < 20%): {np.mean(risks_array < 20) * 100:.1f}%")
    
    # Analyze action patterns
    all_actions = np.concatenate(test_actions)
    if len(all_actions) > 0:
        print(f"\nAction Analysis:")
        action_names = ['Exercise', 'Diet', 'Medication', 'Sleep', 'Stress Mgmt']
        for i, name in enumerate(action_names):
            unique, counts = np.unique(all_actions[:, i], return_counts=True)
            print(f"  {name}: {dict(zip(unique, counts))}")
    
    return test_rewards, test_risks, test_lengths

def compare_ppo_models(model_paths, episodes=10):
    """Compare multiple PPO models"""
    print(f"\n{'='*80}")
    print("PPO MODELS COMPARISON")
    print(f"{'='*80}")
    
    results = {}
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            model_name = os.path.basename(os.path.dirname(model_path))
            print(f"\nTesting {model_name}...")
            
            rewards, risks, lengths = test_trained_ppo(
                model_path, episodes=episodes, render=False, deterministic=True
            )
            
            if rewards is not None:
                results[model_name] = {
                    'rewards': rewards,
                    'risks': risks,
                    'lengths': lengths,
                    'avg_reward': np.mean(rewards),
                    'avg_risk': np.mean(risks),
                    'std_reward': np.std(rewards),
                    'success_rate': np.mean(np.array(risks) < 20) * 100
                }
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Avg Reward':>12} {'Std Reward':>12} {'Avg Risk':>10} {'Success Rate':>12}")
    print(f"{'-'*80}")
    
    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['avg_reward']:>12.2f} {metrics['std_reward']:>12.2f} "
              f"{metrics['avg_risk']:>10.1f}% {metrics['success_rate']:>12.1f}%")
    
    return results

if __name__ == "__main__":
    # Run all PPO experiments
    print("Starting PPO Hyperparameter Search...")
    results = run_ppo_experiments()
    
    print(f"\n{'='*80}")
    print("PPO TRAINING COMPLETED!")
    print(f"{'='*80}")
    print("All models, logs, and metrics saved to 'models/ppo/' directory")
    print("\nFolder structure:")
    print("models/ppo/")
    print("├── ppo_basic/")
    print("│   ├── ppo_basic_final.zip")
    print("│   ├── ppo_basic_episodes_YYYYMMDD_HHMMSS.csv")
    print("│   ├── ppo_basic_training_YYYYMMDD_HHMMSS.csv")
    print("│   ├── training_config.json")
    print("│   └── training_summary.pkl")
    print("├── ppo_large_batch/")
    print("└── ...")
    
    print(f"\nTo test a trained model, run:")
    print("test_trained_ppo('models/ppo/ppo_basic/ppo_basic_final.zip', episodes=5, render=True)")
    
    print(f"\nTo compare multiple models, run:")
    print("model_paths = [")
    print("    'models/ppo/ppo_basic/ppo_basic_final.zip',")
    print("    'models/ppo/ppo_large_batch/ppo_large_batch_final.zip',")
    print("    'models/ppo/ppo_high_entropy/ppo_high_entropy_final.zip'")
    print("]")
    print("compare_ppo_models(model_paths, episodes=10)")