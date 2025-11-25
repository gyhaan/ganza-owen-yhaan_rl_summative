import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
import sys
import csv
import time
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.custom_env import HeartHealthEnvironment

class DQN(nn.Module):
    """Deep Q-Network for Heart Health Intervention Agent"""
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    """Experience Replay Buffer for DQN"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN Agent with target network and experience replay"""
    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Networks
        self.policy_net = DQN(state_size, self._get_total_actions(), config['hidden_size']).to(self.device)
        self.target_net = DQN(state_size, self._get_total_actions(), config['hidden_size']).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config['learning_rate'])
        
        # Replay buffer
        self.memory = ReplayBuffer(config['replay_buffer_size'])
        
        # Training state
        self.steps_done = 0
        
    def _get_total_actions(self):
        """Calculate total number of discrete actions (3*3*2*3*3 = 162)"""
        return 3 * 3 * 2 * 3 * 3
    
    def _action_to_index(self, action):
        """Convert multi-discrete action to single index"""
        return (action[0] * 54 + action[1] * 18 + action[2] * 9 + 
                action[3] * 3 + action[4])
    
    def _index_to_action(self, index):
        """Convert single index back to multi-discrete action"""
        action = [0, 0, 0, 0, 0]
        action[4] = index % 3
        index //= 3
        action[3] = index % 3
        index //= 3
        action[2] = index % 2
        index //= 2
        action[1] = index % 3
        index //= 3
        action[0] = index % 3
        return np.array(action)
    
    def select_action(self, state, epsilon):
        """Epsilon-greedy action selection"""
        if random.random() < epsilon:
            return np.array([random.randint(0, 2), random.randint(0, 2), 
                           random.randint(0, 1), random.randint(0, 2), 
                           random.randint(0, 2)])
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action_index = q_values.max(1)[1].item()
                return self._index_to_action(action_index)
    
    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < self.config['batch_size']:
            return 0
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.config['batch_size'])
        
        states = torch.FloatTensor(states).to(self.device)
        actions_indices = torch.LongTensor([self._action_to_index(a) for a in actions]).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions_indices.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (self.config['gamma'] * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path):
        """Save model weights"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'steps_done': self.steps_done
        }, path)
    
    def load(self, path):
        """Load model weights"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

class MetricsLogger:
    """Comprehensive metrics logging for training"""
    def __init__(self, log_dir, config_name):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # CSV file for episode metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(log_dir, f"{config_name}_{timestamp}.csv")
        
        # Create CSV with headers
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'total_reward', 'episode_length', 'epsilon', 
                'avg_loss', 'final_risk', 'risk_reduction', 'training_time',
                'steps_per_second', 'avg_reward_10', 'avg_reward_100'
            ])
        
        self.start_time = time.time()
        self.episode_times = []
    
    def log_episode(self, episode, total_reward, episode_length, epsilon, 
                   avg_loss, final_risk, risk_reduction):
        """Log episode metrics to CSV"""
        current_time = time.time()
        training_time = current_time - self.start_time
        steps_per_second = episode_length / (training_time - sum(self.episode_times)) if episode > 0 else 0
        
        self.episode_times.append(training_time - sum(self.episode_times))
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, total_reward, episode_length, epsilon,
                avg_loss, final_risk, risk_reduction, training_time,
                steps_per_second, 0, 0  # avg rewards will be calculated later
            ])
    
    def update_average_rewards(self, episodes_rewards):
        """Update CSV with running average rewards"""
        # This would require reading and rewriting the CSV, 
        # so we'll calculate averages during analysis instead

def train_dqn(config, save_path="models/dqn/"):
    """Main training function for DQN with comprehensive logging"""
    
    # Create environment
    env = HeartHealthEnvironment(render_mode=None)
    state_size = env.observation_space.shape[0]
    action_size = 162
    
    # Create agent
    agent = DQNAgent(state_size, action_size, config)
    
    # Create save directory and logger
    model_save_path = os.path.join(save_path, config['name'])
    os.makedirs(model_save_path, exist_ok=True)
    
    logger = MetricsLogger(model_save_path, config['name'])
    
    # Training metrics
    episodes_rewards = []
    episodes_lengths = []
    losses = []
    epsilons = []
    
    print(f"Starting DQN Training: {config['name']}")
    print(f"Configuration: {config}")
    
    for episode in range(config['episodes']):
        episode_start_time = time.time()
        state, _ = env.reset(seed=config['seed'] + episode)
        total_reward = 0
        steps = 0
        episode_losses = []
        
        # Epsilon decay
        epsilon = max(config['epsilon_min'], 
                     config['epsilon_start'] * (config['epsilon_decay'] ** episode))
        epsilons.append(epsilon)
        
        while True:
            # Select and execute action
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store experience
            agent.memory.push(state, action, reward, next_state, done)
            
            # Train
            loss = agent.train_step()
            if loss > 0:
                losses.append(loss)
                episode_losses.append(loss)
            
            state = next_state
            total_reward += reward
            steps += 1
            agent.steps_done += 1
            
            # Update target network
            if agent.steps_done % config['target_update'] == 0:
                agent.update_target_network()
            
            if done:
                break
        
        episodes_rewards.append(total_reward)
        episodes_lengths.append(steps)
        
        # Calculate metrics for logging
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        risk_reduction = info.get('risk_reduction', 0)
        final_risk = info.get('current_risk', 0)
        
        # Log episode to CSV
        logger.log_episode(
            episode=episode,
            total_reward=total_reward,
            episode_length=steps,
            epsilon=epsilon,
            avg_loss=avg_loss,
            final_risk=final_risk,
            risk_reduction=risk_reduction
        )
        
        # Save model periodically
        if episode % 100 == 0 or episode == config['episodes'] - 1:
            model_filename = f"dqn_{config['name']}_episode_{episode}.pth"
            agent.save(os.path.join(model_save_path, model_filename))
        
        # Logging
        if episode % 10 == 0:
            avg_reward_10 = np.mean(episodes_rewards[-10:]) if len(episodes_rewards) >= 10 else total_reward
            avg_reward_100 = np.mean(episodes_rewards[-100:]) if len(episodes_rewards) >= 100 else total_reward
            avg_loss_100 = np.mean(losses[-100:]) if losses else 0
            
            print(f"Episode {episode:4d} | "
                  f"Reward: {total_reward:6.1f} | "
                  f"Avg10: {avg_reward_10:6.1f} | "
                  f"Avg100: {avg_reward_100:6.1f} | "
                  f"Steps: {steps:3d} | "
                  f"Epsilon: {epsilon:.3f} | "
                  f"Loss: {avg_loss_100:.4f} | "
                  f"Risk: {final_risk:.1f}%")
    
    # Save final model and training summary
    agent.save(os.path.join(model_save_path, f"dqn_{config['name']}_final.pth"))
    
    # Save training summary
    training_summary = {
        'config': config,
        'episodes_rewards': episodes_rewards,
        'episodes_lengths': episodes_lengths,
        'losses': losses,
        'epsilons': epsilons,
        'final_metrics': {
            'final_risk': final_risk,
            'best_reward': np.max(episodes_rewards),
            'avg_reward_last_100': np.mean(episodes_rewards[-100:]),
            'training_time': time.time() - logger.start_time
        }
    }
    
    # Save summary as pickle
    import pickle
    with open(os.path.join(model_save_path, 'training_summary.pkl'), 'wb') as f:
        pickle.dump(training_summary, f)
    
    env.close()
    
    return training_summary

# Hyperparameter grid for DQN
DQN_CONFIGS = [
    {
        'name': 'dqn_basic',
        'learning_rate': 1e-4,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'replay_buffer_size': 10000,
        'batch_size': 32,
        'target_update': 100,
        'hidden_size': 128,
        'episodes': 1000,
        'seed': 42
    },
    {
        'name': 'dqn_large_buffer',
        'learning_rate': 1e-4,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'replay_buffer_size': 50000,
        'batch_size': 64,
        'target_update': 100,
        'hidden_size': 128,
        'episodes': 1000,
        'seed': 42
    },
    {
        'name': 'dqn_deep',
        'learning_rate': 5e-4,
        'gamma': 0.95,
        'epsilon_start': 1.0,
        'epsilon_min': 0.05,
        'epsilon_decay': 0.99,
        'replay_buffer_size': 20000,
        'batch_size': 128,
        'target_update': 50,
        'hidden_size': 256,
        'episodes': 1000,
        'seed': 42
    },
    {
        'name': 'dqn_fast_decay',
        'learning_rate': 1e-3,
        'gamma': 0.9,
        'epsilon_start': 1.0,
        'epsilon_min': 0.1,
        'epsilon_decay': 0.98,
        'replay_buffer_size': 15000,
        'batch_size': 64,
        'target_update': 200,
        'hidden_size': 64,
        'episodes': 1000,
        'seed': 42
    }
]

def run_dqn_experiments():
    """Run multiple DQN experiments with different hyperparameters"""
    results = {}
    
    for config in DQN_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Training {config['name']}")
        print(f"{'='*60}")
        
        result = train_dqn(config)
        results[config['name']] = result
        
        # Print summary
        final_metrics = result['final_metrics']
        print(f"\n{config['name']} Summary:")
        print(f"Final Average Reward (last 100): {final_metrics['avg_reward_last_100']:.2f}")
        print(f"Best Episode Reward: {final_metrics['best_reward']:.2f}")
        print(f"Final Risk Score: {final_metrics['final_risk']:.1f}%")
        print(f"Total Training Time: {final_metrics['training_time']:.1f}s")
    
    return results

def test_trained_dqn(model_path, episodes=5, render=False, deterministic=True):
    """Test a trained DQN model"""
    print(f"\nTesting trained DQN model: {model_path}")
    
    # Load environment
    env = HeartHealthEnvironment(render_mode="human" if render else None)
    state_size = env.observation_space.shape[0]
    action_size = 162  # 3*3*2*3*3
    
    # Load agent config from saved model
    checkpoint = torch.load(model_path)
    config = checkpoint['config']
    
    # Create agent and load weights
    agent = DQNAgent(state_size, action_size, config)
    agent.load(model_path)
    
    test_rewards = []
    test_risks = []
    
    for episode in range(episodes):
        state, _ = env.reset(seed=1000 + episode)
        total_reward = 0
        steps = 0
        
        while True:
            # Use greedy action selection during testing
            if deterministic:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                q_values = agent.policy_net(state_tensor)
                action_index = q_values.max(1)[1].item()
                action = agent._index_to_action(action_index)
            else:
                # Use epsilon-greedy with very small epsilon
                action = agent.select_action(state, epsilon=0.01)
            
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
    # Run all DQN experiments
    results = run_dqn_experiments()
    
    print("\nDQN training completed! All models and metrics saved.")
    print("Check the 'models/dqn/' directory for:")
    print("- Model weights (.pth files)")
    print("- Training metrics (CSV files)")
    print("- Training summaries (pickle files)")