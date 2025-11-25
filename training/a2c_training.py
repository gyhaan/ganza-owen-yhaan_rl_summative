import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import csv
import time
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.custom_env import HeartHealthEnvironment

class ActorNetwork(nn.Module):
    """
    Actor Network for A2C - Policy for multi-discrete actions
    """
    def __init__(self, state_size, action_dims, hidden_size=128):
        super(ActorNetwork, self).__init__()
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

class CriticNetwork(nn.Module):
    """
    Critic Network for A2C - Value function
    """
    def __init__(self, state_size, hidden_size=128):
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.network(x)

class A2CAgent:
    """
    Advantage Actor-Critic (A2C) Agent
    """
    def __init__(self, state_size, action_dims, config):
        self.state_size = state_size
        self.action_dims = action_dims
        self.config = config
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Networks
        self.actor = ActorNetwork(state_size, action_dims, config['hidden_size']).to(self.device)
        self.critic = CriticNetwork(state_size, config['hidden_size']).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['actor_lr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config['critic_lr'])
        
        # Training buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.entropies = []
    
    def select_action(self, state):
        """Sample action from policy and return action, log_prob, value, entropy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action logits and value
        action_logits = self.actor(state_tensor)
        value = self.critic(state_tensor)
        
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
        self.states.append(state)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.entropies.append(entropies)
        self.values.append(value.squeeze())
        
        return np.array(actions), torch.stack(log_probs).sum(), value.item(), torch.stack(entropies).mean()
    
    def store_transition(self, reward, next_state, done):
        """Store transition for n-step learning"""
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def update(self):
        """Update networks using n-step returns"""
        if len(self.rewards) < self.config['n_steps']:
            return 0, 0, 0
        
        # Calculate n-step returns
        returns = []
        R = 0
        
        # Bootstrap with value of last state if not done
        if not self.dones[-1]:
            last_state_tensor = torch.FloatTensor(self.next_states[-1]).unsqueeze(0).to(self.device)
            R = self.critic(last_state_tensor).squeeze().item()
        
        # Calculate returns backwards
        for r, done in zip(reversed(self.rewards), reversed(self.dones)):
            R = r + self.config['gamma'] * R * (not done)
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns).to(self.device)
        values = torch.stack(self.values)
        
        # Calculate advantages
        advantages = returns - values.detach()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Calculate losses
        policy_loss = 0
        value_loss = 0
        entropy_loss = 0
        
        for i, (log_probs, entropy) in enumerate(zip(self.log_probs, self.entropies)):
            total_log_prob = torch.stack(log_probs).sum()
            policy_loss -= total_log_prob * advantages[i]
            entropy_loss += entropy
        
        # Add entropy regularization
        policy_loss -= self.config['entropy_coef'] * entropy_loss
        
        # Value loss
        value_loss = nn.MSELoss()(values, returns)
        
        # Total loss
        total_loss = policy_loss + self.config['value_coef'] * value_loss
        
        # Update actor
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        # Store metrics
        policy_loss_value = policy_loss.item()
        value_loss_value = value_loss.item()
        entropy_value = entropy_loss.item() / len(self.log_probs)
        
        # Clear buffers
        self._clear_buffers()
        
        return policy_loss_value, value_loss_value, entropy_value
    
    def _clear_buffers(self):
        """Clear training buffers"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.entropies = []
    
    def save(self, path):
        """Save model weights and optimizer states"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config
        }, path)
    
    def load(self, path):
        """Load model weights and optimizer states"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

class A2CMetricsLogger:
    """Comprehensive metrics logging for A2C training"""
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
                'training_time', 'avg_reward_10', 'avg_reward_100',
                'n_step_updates'
            ])
        
        self.start_time = time.time()
    
    def log_episode(self, episode, total_reward, episode_length, policy_loss,
                   value_loss, entropy, final_risk, risk_reduction, n_updates):
        """Log episode metrics to CSV"""
        training_time = time.time() - self.start_time
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, total_reward, episode_length, policy_loss,
                value_loss, entropy, final_risk, risk_reduction, training_time,
                0, 0, n_updates  # avg rewards calculated during analysis
            ])

def train_a2c(config, save_path="models/a2c/"):
    """Main training function for A2C with comprehensive logging"""
    
    # Create environment
    env = HeartHealthEnvironment(render_mode=None)
    state_size = env.observation_space.shape[0]
    action_dims = [3, 3, 2, 3, 3]
    
    # Create agent
    agent = A2CAgent(state_size, action_dims, config)
    
    # Create save directory and logger
    model_save_path = os.path.join(save_path, config['name'])
    os.makedirs(model_save_path, exist_ok=True)
    
    logger = A2CMetricsLogger(model_save_path, config['name'])
    
    # Training metrics
    episodes_rewards = []
    episodes_lengths = []
    policy_losses = []
    value_losses = []
    entropies = []
    n_step_updates = []
    
    print(f"Starting A2C Training: {config['name']}")
    print(f"Configuration: {config}")
    
    for episode in range(config['episodes']):
        state, _ = env.reset(seed=config['seed'] + episode)
        total_reward = 0
        steps = 0
        episode_policy_loss = 0
        episode_value_loss = 0
        episode_entropy = 0
        updates_count = 0
        
        while True:
            # Select and execute action
            action, log_prob, value, entropy = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(reward, next_state, done)
            
            total_reward += reward
            steps += 1
            episode_entropy += entropy.item()
            
            # Update every n_steps or when episode ends
            if steps % config['n_steps'] == 0 or done:
                policy_loss, value_loss, avg_entropy = agent.update()
                if policy_loss != 0:  # Only count if update happened
                    episode_policy_loss += policy_loss
                    episode_value_loss += value_loss
                    updates_count += 1
            
            state = next_state
            
            if done:
                break
        
        # Store episode metrics
        episodes_rewards.append(total_reward)
        episodes_lengths.append(steps)
        
        if updates_count > 0:
            policy_losses.append(episode_policy_loss / updates_count)
            value_losses.append(episode_value_loss / updates_count)
            entropies.append(episode_entropy / steps)
        else:
            policy_losses.append(0)
            value_losses.append(0)
            entropies.append(0)
        
        n_step_updates.append(updates_count)
        
        # Calculate metrics for logging
        risk_reduction = info.get('risk_reduction', 0)
        final_risk = info.get('current_risk', 0)
        avg_policy_loss = episode_policy_loss / max(updates_count, 1)
        avg_value_loss = episode_value_loss / max(updates_count, 1)
        avg_entropy = episode_entropy / steps if steps > 0 else 0
        
        # Log episode to CSV
        logger.log_episode(
            episode=episode,
            total_reward=total_reward,
            episode_length=steps,
            policy_loss=avg_policy_loss,
            value_loss=avg_value_loss,
            entropy=avg_entropy,
            final_risk=final_risk,
            risk_reduction=risk_reduction,
            n_updates=updates_count
        )
        
        # Save model periodically
        if episode % 100 == 0 or episode == config['episodes'] - 1:
            model_filename = f"a2c_{config['name']}_episode_{episode}.pth"
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
                  f"Policy Loss: {avg_policy_loss:8.4f} | "
                  f"Value Loss: {avg_value_loss:8.4f} | "
                  f"Entropy: {avg_entropy:6.4f} | "
                  f"Risk: {final_risk:5.1f}% | "
                  f"Updates: {updates_count:2d}")
    
    # Save final model
    final_model_path = os.path.join(model_save_path, f"a2c_{config['name']}_final.pth")
    agent.save(final_model_path)
    
    # Save training summary
    training_summary = {
        'config': config,
        'episodes_rewards': episodes_rewards,
        'episodes_lengths': episodes_lengths,
        'policy_losses': policy_losses,
        'value_losses': value_losses,
        'entropies': entropies,
        'n_step_updates': n_step_updates,
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

# Hyperparameter configurations for A2C
A2C_CONFIGS = [
    {
        'name': 'a2c_basic',
        'actor_lr': 1e-4,
        'critic_lr': 1e-3,
        'gamma': 0.99,
        'n_steps': 5,
        'entropy_coef': 0.01,
        'value_coef': 0.5,
        'hidden_size': 128,
        'episodes': 1000,
        'seed': 42
    },
    {
        'name': 'a2c_fast',
        'actor_lr': 1e-3,
        'critic_lr': 1e-2,
        'gamma': 0.95,
        'n_steps': 3,
        'entropy_coef': 0.01,
        'value_coef': 0.5,
        'hidden_size': 128,
        'episodes': 1000,
        'seed': 42
    },
    {
        'name': 'a2c_high_entropy',
        'actor_lr': 5e-4,
        'critic_lr': 5e-3,
        'gamma': 0.99,
        'n_steps': 8,
        'entropy_coef': 0.1,
        'value_coef': 0.5,
        'hidden_size': 128,
        'episodes': 1000,
        'seed': 42
    },
    {
        'name': 'a2c_deep',
        'actor_lr': 1e-4,
        'critic_lr': 1e-3,
        'gamma': 0.99,
        'n_steps': 5,
        'entropy_coef': 0.01,
        'value_coef': 0.5,
        'hidden_size': 256,
        'episodes': 1000,
        'seed': 42
    },
    {
        'name': 'a2c_long_nstep',
        'actor_lr': 1e-4,
        'critic_lr': 1e-3,
        'gamma': 0.99,
        'n_steps': 10,
        'entropy_coef': 0.01,
        'value_coef': 0.5,
        'hidden_size': 128,
        'episodes': 1000,
        'seed': 42
    }
]

def run_a2c_experiments():
    """Run multiple A2C experiments with different hyperparameters"""
    results = {}
    
    for config in A2C_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Training A2C: {config['name']}")
        print(f"{'='*60}")
        
        result = train_a2c(config)
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

def test_trained_a2c(model_path, episodes=5, render=False):
    """Test a trained A2C model"""
    print(f"\nTesting trained A2C model: {model_path}")
    
    # Load environment
    env = HeartHealthEnvironment(render_mode="human" if render else None)
    state_size = env.observation_space.shape[0]
    action_dims = [3, 3, 2, 3, 3]
    
    # Load agent config from saved model
    checkpoint = torch.load(model_path)
    config = checkpoint['config']
    
    # Create agent and load weights
    agent = A2CAgent(state_size, action_dims, config)
    agent.load(model_path)
    
    test_rewards = []
    test_risks = []
    
    for episode in range(episodes):
        state, _ = env.reset(seed=1000 + episode)
        total_reward = 0
        steps = 0
        
        while True:
            # Use greedy action selection
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            action_logits = agent.actor(state_tensor)
            
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
    # Run all A2C experiments
    results = run_a2c_experiments()
    
    # Save overall results
    import pickle
    overall_results_path = "models/a2c/a2c_overall_results.pkl"
    os.makedirs(os.path.dirname(overall_results_path), exist_ok=True)
    with open(overall_results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n{'='*60}")
    print("A2C training completed!")
    print(f"{'='*60}")
    print("All models and metrics saved to 'models/a2c/' directory")
    print(f"\nTo test a trained model, run:")
    print("test_trained_a2c('models/a2c/a2c_basic/a2c_basic_final.pth')")