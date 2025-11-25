import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import os
import sys
import csv
import time
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.custom_env import HeartHealthEnvironment

class PolicyNetwork(nn.Module):
    """
    Policy Network for REINFORCE algorithm with multi-discrete actions
    """
    def __init__(self, state_size, action_dims, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.action_dims = action_dims
        
        # Shared feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Separate heads for each action dimension
        self.exercise_head = nn.Linear(hidden_size, action_dims[0])
        self.diet_head = nn.Linear(hidden_size, action_dims[1])
        self.medication_head = nn.Linear(hidden_size, action_dims[2])
        self.sleep_head = nn.Linear(hidden_size, action_dims[3])
        self.stress_head = nn.Linear(hidden_size, action_dims[4])
    
    def forward(self, x):
        features = self.shared_net(x)
        
        exercise_logits = self.exercise_head(features)
        diet_logits = self.diet_head(features)
        medication_logits = self.medication_head(features)
        sleep_logits = self.sleep_head(features)
        stress_logits = self.stress_head(features)
        
        return [exercise_logits, diet_logits, medication_logits, sleep_logits, stress_logits]

class ValueNetwork(nn.Module):
    """
    Value Network as baseline for REINFORCE
    """
    def __init__(self, state_size, hidden_size=128):
        super(ValueNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.network(x)

class REINFORCEAgent:
    """
    REINFORCE Agent with baseline for multi-discrete actions
    """
    def __init__(self, state_size, action_dims, config):
        self.state_size = state_size
        self.action_dims = action_dims
        self.config = config
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Networks
        self.policy_net = PolicyNetwork(state_size, action_dims, config['hidden_size']).to(self.device)
        self.value_net = ValueNetwork(state_size, config['hidden_size']).to(self.device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=config['policy_lr'])
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=config['value_lr'])
        
        # Training buffers
        self.saved_states = []
        self.saved_actions = []
        self.saved_log_probs = []
        self.saved_rewards = []
        self.saved_entropies = []
    
    def select_action(self, state):
        """Sample action from policy and return action, log_prob, entropy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action logits
        action_logits = self.policy_net(state_tensor)
        
        # Sample actions for each dimension
        actions = []
        log_probs = []
        entropies = []
        
        for i, logits in enumerate(action_logits):
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            
            actions.append(action.item())
            log_probs.append(log_prob)
            entropies.append(entropy)
        
        # Store for training
        self.saved_states.append(state)
        self.saved_actions.append(actions)
        self.saved_log_probs.append(log_probs)
        self.saved_entropies.append(entropies)
        
        return np.array(actions), torch.stack(log_probs).sum(), torch.stack(entropies).mean()
    
    def update_policy(self):
        """Update policy using REINFORCE with baseline"""
        if not self.saved_rewards:
            return 0, 0, 0
        
        returns = []
        R = 0
        
        # Calculate discounted returns
        for r in self.saved_rewards[::-1]:
            R = r + self.config['gamma'] * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize
        
        # Calculate baseline values
        states = torch.FloatTensor(np.array(self.saved_states)).to(self.device)
        baseline_values = self.value_net(states).squeeze()
        
        # Calculate advantages
        advantages = returns - baseline_values.detach()
        
        # Calculate policy loss
        policy_loss = 0
        total_entropy = 0
        
        for i, (log_probs, entropy) in enumerate(zip(self.saved_log_probs, self.saved_entropies)):
            # Sum log probs across all action dimensions
            total_log_prob = torch.stack(log_probs).sum()
            policy_loss -= total_log_prob * advantages[i]
            total_entropy += entropy
        
        # Add entropy regularization
        policy_loss -= self.config['entropy_coef'] * total_entropy
        
        # Value loss (mean squared error)
        value_loss = nn.MSELoss()(baseline_values, returns)
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.policy_optimizer.step()
        
        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
        self.value_optimizer.step()
        
        # Store metrics
        policy_loss_value = policy_loss.item()
        value_loss_value = value_loss.item()
        entropy_value = total_entropy.item() / len(self.saved_log_probs)
        
        # Clear buffers
        self.saved_states = []
        self.saved_actions = []
        self.saved_log_probs = []
        self.saved_rewards = []
        self.saved_entropies = []
        
        return policy_loss_value, value_loss_value, entropy_value
    
    def store_reward(self, reward):
        """Store reward for the current step"""
        self.saved_rewards.append(reward)
    
    def save(self, path):
        """Save model weights and optimizer states"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'config': self.config
        }, path)
    
    def load(self, path):
        """Load model weights and optimizer states"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])

class ReinforceMetricsLogger:
    """Comprehensive metrics logging for REINFORCE training"""
    def __init__(self, log_dir, config_name):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(log_dir, f"{config_name}_{timestamp}.csv")
        
        # Create CSV with headers
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'total_reward', 'episode_length', 'policy_loss',
                'value_loss', 'entropy', 'final_risk', 'risk_reduction', 
                'training_time', 'avg_reward_10', 'avg_reward_100'
            ])
        
        self.start_time = time.time()
        self.episode_times = []
    
    def log_episode(self, episode, total_reward, episode_length, policy_loss,
                   value_loss, entropy, final_risk, risk_reduction):
        """Log episode metrics to CSV"""
        current_time = time.time()
        training_time = current_time - self.start_time
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, total_reward, episode_length, policy_loss,
                value_loss, entropy, final_risk, risk_reduction, training_time,
                0, 0  # avg rewards will be calculated during analysis
            ])

def train_reinforce(config, save_path="models/reinforce/"):
    """Main training function for REINFORCE with comprehensive logging"""
    
    # Create environment
    env = HeartHealthEnvironment(render_mode=None)
    state_size = env.observation_space.shape[0]
    action_dims = [3, 3, 2, 3, 3]  # Dimensions for each action type
    
    # Create agent
    agent = REINFORCEAgent(state_size, action_dims, config)
    
    # Create save directory and logger
    model_save_path = os.path.join(save_path, config['name'])
    os.makedirs(model_save_path, exist_ok=True)
    
    logger = ReinforceMetricsLogger(model_save_path, config['name'])
    
    # Training metrics
    episodes_rewards = []
    episodes_lengths = []
    policy_losses = []
    value_losses = []
    entropies = []
    
    print(f"Starting REINFORCE Training: {config['name']}")
    print(f"Configuration: {config}")
    
    for episode in range(config['episodes']):
        episode_start_time = time.time()
        state, _ = env.reset(seed=config['seed'] + episode)
        total_reward = 0
        steps = 0
        episode_entropy = 0
        
        while True:
            # Select and execute action
            action, log_prob, entropy = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store reward and update metrics
            agent.store_reward(reward)
            total_reward += reward
            steps += 1
            episode_entropy += entropy.item()
            
            state = next_state
            
            if done:
                break
        
        # Update policy after each episode
        policy_loss, value_loss, avg_entropy = agent.update_policy()
        
        # Store metrics
        episodes_rewards.append(total_reward)
        episodes_lengths.append(steps)
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)
        entropies.append(avg_entropy)
        
        # Calculate metrics for logging
        risk_reduction = info.get('risk_reduction', 0)
        final_risk = info.get('current_risk', 0)
        avg_episode_entropy = episode_entropy / steps if steps > 0 else 0
        
        # Log episode to CSV
        logger.log_episode(
            episode=episode,
            total_reward=total_reward,
            episode_length=steps,
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy=avg_episode_entropy,
            final_risk=final_risk,
            risk_reduction=risk_reduction
        )
        
        # Save model periodically
        if episode % 100 == 0 or episode == config['episodes'] - 1:
            model_filename = f"reinforce_{config['name']}_episode_{episode}.pth"
            agent.save(os.path.join(model_save_path, model_filename))
        
        # Logging to console
        if episode % 10 == 0:
            avg_reward_10 = np.mean(episodes_rewards[-10:]) if len(episodes_rewards) >= 10 else total_reward
            avg_reward_100 = np.mean(episodes_rewards[-100:]) if len(episodes_rewards) >= 100 else total_reward
            
            print(f"Episode {episode:4d} | "
                  f"Reward: {total_reward:6.1f} | "
                  f"Avg10: {avg_reward_10:6.1f} | "
                  f"Avg100: {avg_reward_100:6.1f} | "
                  f"Steps: {steps:3d} | "
                  f"Policy Loss: {policy_loss:8.4f} | "
                  f"Value Loss: {value_loss:8.4f} | "
                  f"Entropy: {avg_episode_entropy:6.4f} | "
                  f"Risk: {final_risk:5.1f}%")
    
    # Save final model
    final_model_path = os.path.join(model_save_path, f"reinforce_{config['name']}_final.pth")
    agent.save(final_model_path)
    
    # Save training summary
    training_summary = {
        'config': config,
        'episodes_rewards': episodes_rewards,
        'episodes_lengths': episodes_lengths,
        'policy_losses': policy_losses,
        'value_losses': value_losses,
        'entropies': entropies,
        'final_metrics': {
            'final_risk': final_risk,
            'best_reward': np.max(episodes_rewards),
            'avg_reward_last_10': np.mean(episodes_rewards[-10:]),
            'avg_reward_last_100': np.mean(episodes_rewards[-100:]),
            'training_time': time.time() - logger.start_time,
            'total_episodes': len(episodes_rewards)
        }
    }
    
    # Save summary as pickle
    import pickle
    summary_path = os.path.join(model_save_path, 'training_summary.pkl')
    with open(summary_path, 'wb') as f:
        pickle.dump(training_summary, f)
    
    env.close()
    
    return training_summary

# Hyperparameter configurations for REINFORCE
REINFORCE_CONFIGS = [
    {
        'name': 'reinforce_basic',
        'policy_lr': 1e-4,
        'value_lr': 1e-3,
        'gamma': 0.99,
        'entropy_coef': 0.01,
        'hidden_size': 128,
        'episodes': 1000,
        'seed': 42
    },
    {
        'name': 'reinforce_high_lr',
        'policy_lr': 1e-3,
        'value_lr': 1e-2,
        'gamma': 0.95,
        'entropy_coef': 0.01,
        'hidden_size': 128,
        'episodes': 1000,
        'seed': 42
    },
    {
        'name': 'reinforce_high_entropy',
        'policy_lr': 5e-4,
        'value_lr': 5e-3,
        'gamma': 0.99,
        'entropy_coef': 0.1,
        'hidden_size': 128,
        'episodes': 1000,
        'seed': 42
    },
    {
        'name': 'reinforce_deep',
        'policy_lr': 1e-4,
        'value_lr': 1e-3,
        'gamma': 0.99,
        'entropy_coef': 0.01,
        'hidden_size': 256,
        'episodes': 1000,
        'seed': 42
    },
    {
        'name': 'reinforce_low_gamma',
        'policy_lr': 1e-4,
        'value_lr': 1e-3,
        'gamma': 0.9,
        'entropy_coef': 0.01,
        'hidden_size': 128,
        'episodes': 1000,
        'seed': 42
    }
]

def run_reinforce_experiments():
    """Run multiple REINFORCE experiments with different hyperparameters"""
    results = {}
    
    for config in REINFORCE_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Training REINFORCE: {config['name']}")
        print(f"{'='*60}")
        
        result = train_reinforce(config)
        results[config['name']] = result
        
        # Print summary
        final_metrics = result['final_metrics']
        print(f"\n{config['name']} Summary:")
        print(f"Final Average Reward (last 10): {final_metrics['avg_reward_last_10']:.2f}")
        print(f"Final Average Reward (last 100): {final_metrics['avg_reward_last_100']:.2f}")
        print(f"Best Episode Reward: {final_metrics['best_reward']:.2f}")
        print(f"Final Risk Score: {final_metrics['final_risk']:.1f}%")
        print(f"Total Training Time: {final_metrics['training_time']:.1f}s")
        print(f"Total Episodes: {final_metrics['total_episodes']}")
    
    return results

def test_trained_reinforce(model_path, episodes=5, render=False):
    """Test a trained REINFORCE model"""
    print(f"\nTesting trained REINFORCE model: {model_path}")
    
    # Load environment
    env = HeartHealthEnvironment(render_mode="human" if render else None)
    state_size = env.observation_space.shape[0]
    action_dims = [3, 3, 2, 3, 3]
    
    # Load agent config from saved model
    checkpoint = torch.load(model_path)
    config = checkpoint['config']
    
    # Create agent and load weights
    agent = REINFORCEAgent(state_size, action_dims, config)
    agent.load(model_path)
    
    test_rewards = []
    test_risks = []
    
    for episode in range(episodes):
        state, _ = env.reset(seed=1000 + episode)
        total_reward = 0
        steps = 0
        
        while True:
            # Use greedy action selection (no exploration during testing)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            action_logits = agent.policy_net(state_tensor)
            
            # Select greedy actions
            action = []
            for logits in action_logits:
                probs = torch.softmax(logits, dim=-1)
                action.append(torch.argmax(probs).item())
            
            action = np.array(action)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        test_rewards.append(total_reward)
        test_risks.append(info['current_risk'])
        
        print(f"Test Episode {episode}: Reward = {total_reward:.1f}, "
              f"Final Risk = {info['current_risk']:.1f}%, "
              f"Steps = {steps}")
    
    env.close()
    
    print(f"\nTest Results (over {episodes} episodes):")
    print(f"Average Reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"Average Final Risk: {np.mean(test_risks):.1f}% ± {np.std(test_risks):.1f}%")
    print(f"Best Reward: {np.max(test_rewards):.2f}")
    print(f"Worst Reward: {np.min(test_rewards):.2f}")
    
    return test_rewards, test_risks

if __name__ == "__main__":
    # Run all REINFORCE experiments
    results = run_reinforce_experiments()
    
    # Save overall results
    import pickle
    overall_results_path = "models/reinforce/reinforce_overall_results.pkl"
    os.makedirs(os.path.dirname(overall_results_path), exist_ok=True)
    with open(overall_results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n{'='*60}")
    print("REINFORCE training completed!")
    print(f"{'='*60}")
    print("All models and metrics saved to 'models/reinforce/' directory")
    print("\nFolder structure:")
    print("models/reinforce/")
    print("├── reinforce_basic/")
    print("│   ├── reinforce_basic_episode_100.pth")
    print("│   ├── reinforce_basic_final.pth")
    print("│   ├── reinforce_basic_YYYYMMDD_HHMMSS.csv")
    print("│   └── training_summary.pkl")
    print("├── reinforce_high_lr/")
    print("└── ...")
    print(f"\nTo test a trained model, run:")
    print("test_trained_reinforce('models/reinforce/reinforce_basic/reinforce_basic_final.pth')")