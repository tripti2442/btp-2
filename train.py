"""
Main Training Script for DRL-TCOA Algorithm
Implements Algorithm 1 from the paper
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os

from config import get_config
from environment import UAVAssistedVECEnv
from td3_agent import TD3Agent


def train_drl_tcoa(config, save_dir='results'):
    """
    Train the DRL-TCOA algorithm
    Implements Algorithm 1 from the paper
    
    Args:
        config: Configuration dictionary
        save_dir: Directory to save results
        
    Returns:
        Training metrics
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize environment
    env = UAVAssistedVECEnv(config)
    
    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Max action: {max_action}")
    
    # Initialize TD3 agent (Lines 1-3 of Algorithm 1)
    agent = TD3Agent(state_dim, action_dim, max_action, config['td3'])
    
    # Training parameters
    max_episodes = config['td3']['max_episodes']
    max_steps = config['td3']['max_steps']
    batch_size = config['td3']['batch_size']
    
    # Metrics storage
    episode_rewards = []
    episode_costs = []
    episode_penalties = []
    critic_losses = []
    actor_losses = []
    
    # Training loop (Lines 4-21 of Algorithm 1)
    for episode in tqdm(range(max_episodes), desc="Training"):
        # Reset environment (Line 5)
        state = env.reset()
        episode_reward = 0
        episode_cost = 0
        episode_penalty = 0
        
        # Episode loop (Lines 6-20)
        for step in range(max_steps):
            # Select action with exploration noise (Line 7)
            action = agent.select_action(state, add_noise=True, noise_scale=0.1)
            
            # Execute action in environment (Line 8)
            next_state, reward, done, info = env.step(action)

            reward = reward / 1e10
            
            # Store transition in replay buffer (Line 9)
            agent.replay_buffer.add(state, action, next_state, reward, done)
            
            # Update metrics
            episode_reward += reward
            episode_cost += info['total_cost']
            episode_penalty += info['penalty']
            
            # Train agent if buffer is full (Lines 10-19)
            if agent.replay_buffer.size >= batch_size:
                metrics = agent.train()
                
                if metrics['critic_loss'] is not None:
                    critic_losses.append(metrics['critic_loss'])
                if metrics['actor_loss'] is not None:
                    actor_losses.append(metrics['actor_loss'])
            
            state = next_state
            
            if done:
                break
        
        # Store episode metrics
        episode_rewards.append(episode_reward)
        episode_costs.append(episode_cost)
        episode_penalties.append(episode_penalty)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_cost = np.mean(episode_costs[-10:])
            print(f"\nEpisode {episode + 1}/{max_episodes}")
            print(f"  Avg Reward (last 10): {avg_reward:.2f}")
            print(f"  Avg Cost (last 10): {avg_cost:.2f}")
            if len(critic_losses) > 0:
                print(f"  Avg Critic Loss: {np.mean(critic_losses[-100:]):.4f}")
            if len(actor_losses) > 0:
                print(f"  Avg Actor Loss: {np.mean(actor_losses[-100:]):.4f}")
    
    # Save model
    model_path = os.path.join(save_dir, 'td3_model.pth')
    agent.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save metrics
    metrics = {
        'episode_rewards': episode_rewards,
        'episode_costs': episode_costs,
        'episode_penalties': episode_penalties,
        'critic_losses': critic_losses,
        'actor_losses': actor_losses,
    }
    
    metrics_path = os.path.join(save_dir, 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        # Convert to lists for JSON serialization
        json_metrics = {k: [float(v) for v in vals] for k, vals in metrics.items()}
        json.dump(json_metrics, f, indent=2)
    
    print(f"Metrics saved to {metrics_path}")
    
    return metrics, agent


def evaluate_agent(agent, config, num_episodes=10):
    """
    Evaluate trained agent
    
    Args:
        agent: Trained TD3 agent
        config: Configuration dictionary
        num_episodes: Number of evaluation episodes
        
    Returns:
        Evaluation metrics
    """
    env = UAVAssistedVECEnv(config)
    
    episode_rewards = []
    episode_costs = []
    episode_penalties = []
    
    for episode in range(num_episodes):
        # state = env.reset()
        state = env.reset()
        episode_reward = 0
        episode_cost = 0
        episode_penalty = 0
        
        for step in range(config['td3']['max_steps']):
            # Select action without exploration noise
            action = agent.select_action(state, add_noise=False)
            
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_cost += info['total_cost']
            episode_penalty += info['penalty']
            
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_costs.append(episode_cost)
        episode_penalties.append(episode_penalty)
    
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_cost': np.mean(episode_costs),
        'std_cost': np.std(episode_costs),
        'mean_penalty': np.mean(episode_penalties),
        'std_penalty': np.std(episode_penalties),
    }
    
    return results


def main():
    """Main training function"""
    # Get configuration for normal scenario with Map 1 and 10 vehicles
    config = get_config(scenario='normal', map_name='map1', num_vehicles=10)
    
    print("=" * 80)
    print("DRL-TCOA Training - UAV-Assisted VEC Network")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Scenario: normal")
    print(f"  Map: map1")
    print(f"  Number of vehicles: {config['num_vehicles']}")
    print(f"  Max episodes: {config['td3']['max_episodes']}")
    print(f"  Max steps per episode: {config['td3']['max_steps']}")
    print(f"  Batch size: {config['td3']['batch_size']}")
    print(f"  Actor learning rate: {config['td3']['actor_lr']}")
    print(f"  Critic learning rate: {config['td3']['critic_lr']}")
    print("=" * 80)
    
    # Train agent
    print("\nStarting training...")
    metrics, agent = train_drl_tcoa(config, save_dir='results')
    
    # Evaluate agent
    print("\nEvaluating trained agent...")
    eval_results = evaluate_agent(agent, config, num_episodes=10)
    
    print("\n" + "=" * 80)
    print("Evaluation Results:")
    print("=" * 80)
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    print("=" * 80)
    
    # Save evaluation results
    eval_path = 'results/evaluation_results.json'
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nEvaluation results saved to {eval_path}")


if __name__ == "__main__":
    main()
