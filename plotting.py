"""
Plotting Script for Paper Figures
Reproduces all figures from the paper
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from matplotlib import rcParams

# Set publication-quality plot settings
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 12
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['figure.titlesize'] = 12


def plot_convergence(metrics_list, labels, save_path='plots/fig5_convergence.png'):
    """
    Plot Figure 5: Convergence performance
    
    Args:
        metrics_list: List of metrics dictionaries
        labels: List of labels for each metric
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # (a) Episode reward for different N
    ax = axes[0]
    for metrics, label in zip(metrics_list, labels):
        rewards = metrics['episode_rewards']
        # Smooth with moving average
        window = 10
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label=label, linewidth=1.5)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.set_title('(a) Episode reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (b) Loss with different N
    ax = axes[1]
    for metrics, label in zip(metrics_list, labels):
        losses = metrics['critic_losses']
        # Smooth
        window = 50
        if len(losses) > window:
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, label=label, linewidth=1.5)
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Critic Loss')
    ax.set_title('(b) Loss with different N')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # (c) Episode reward versus learning rate
    ax = axes[2]
    # This would need to be run with different learning rates
    # Placeholder for demonstration
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.set_title('(c) Episode reward versus learning rate')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    plt.close()


def plot_cost_comparison(results_dict, save_path='plots/fig6_cost_comparison.png'):
    """
    Plot Figure 6: Cost with different strategies and maps
    
    Args:
        results_dict: Dictionary with strategy names as keys and costs as values
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    strategies = list(results_dict.keys())
    maps = ['Map1', 'Map2', 'Map3']
    
    x = np.arange(len(strategies))
    width = 0.25
    
    for i, map_name in enumerate(maps):
        costs = [results_dict[strategy][map_name] for strategy in strategies]
        ax.bar(x + i*width, costs, width, label=map_name)
    
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Total System Cost')
    ax.set_title('Cost with different strategies and maps')
    ax.set_xticks(x + width)
    ax.set_xticklabels(strategies, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    plt.close()


def plot_scenario_comparison(results_dict, save_path='plots/fig7_scenario_comparison.png'):
    """
    Plot Figure 7: Cost with different strategies and tasks (scenarios)
    
    Args:
        results_dict: Dictionary with scenario names as keys
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    scenarios = ['Normal', 'Time-sensitive', 'Energy-deficiency', 'Fund-poor', 'Energy-rich']
    strategies = ['DRL-TCOA', 'DDPG-TCOA', 'DQN-BOA', 'LE', 'FO', 'RO']
    
    x = np.arange(len(scenarios))
    width = 0.13
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
    
    for i, strategy in enumerate(strategies):
        costs = [results_dict[scenario][strategy] for scenario in scenarios]
        ax.bar(x + i*width, costs, width, label=strategy, color=colors[i])
    
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Total System Cost')
    ax.set_title('Cost with different strategies and tasks')
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(scenarios, rotation=15, ha='right')
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    plt.close()


def plot_cost_over_time(results_dict, save_path='plots/fig8a_cost_over_time.png'):
    """
    Plot Figure 8(a): Cost in different slot number
    
    Args:
        results_dict: Dictionary with strategy names and their costs over time
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    strategies = ['DRL-TCOA', 'DDPG-TCOA', 'DQN-BOA', 'LE', 'FO', 'RO']
    
    for strategy in strategies:
        costs = results_dict[strategy]
        time_slots = range(10, len(costs) * 10 + 1, 10)
        ax.plot(time_slots, costs, marker='o', label=strategy, linewidth=2, markersize=4)
    
    ax.set_xlabel('Time Slot')
    ax.set_ylabel('Total System Cost')
    ax.set_title('(a) Cost in different slot number')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    plt.close()


def plot_cost_components(results_dict, save_path='plots/fig8b_cost_components.png'):
    """
    Plot Figure 8(b): Percentage of different cost component
    
    Args:
        results_dict: Dictionary with strategy names and cost components
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    strategies = list(results_dict.keys())
    aoi_costs = [results_dict[s]['aoi'] for s in strategies]
    energy_costs = [results_dict[s]['energy'] for s in strategies]
    rental_costs = [results_dict[s]['rental'] for s in strategies]
    
    x = np.arange(len(strategies))
    width = 0.6
    
    p1 = ax.bar(x, aoi_costs, width, label='AoI Cost', color='#4472C4')
    p2 = ax.bar(x, energy_costs, width, bottom=aoi_costs, label='Energy Cost', color='#ED7D31')
    p3 = ax.bar(x, rental_costs, width, 
                bottom=np.array(aoi_costs) + np.array(energy_costs),
                label='Rental Price', color='#A5A5A5')
    
    ax.set_ylabel('Cost')
    ax.set_title('(b) Percentage of different cost component')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    plt.close()


def plot_weight_impact(results_dict, save_path='plots/fig9_weight_impact.png'):
    """
    Plot Figure 9: AoI weight impact on cost
    
    Args:
        results_dict: Dictionary with weight values and costs
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # (a) Tradeoff among AoI cost, energy consumption, and rental price
    ax = axes[0]
    weights = results_dict['weights']
    aoi_costs = results_dict['aoi_costs']
    energy_costs = results_dict['energy_costs']
    rental_costs = results_dict['rental_costs']
    
    ax.plot(weights, aoi_costs, marker='o', label='AoI Cost', linewidth=2)
    ax.plot(weights, energy_costs, marker='s', label='Energy Consumption', linewidth=2)
    ax.plot(weights, rental_costs, marker='^', label='Rental Price', linewidth=2)
    
    ax.set_xlabel('Weight on AoI (λ_A)')
    ax.set_ylabel('Cost Component')
    ax.set_title('(a) Tradeoff among AoI cost, energy consumption, and rental price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (b) Cost under different strategies and weight on AoI
    ax = axes[1]
    strategies = ['DRL-TCOA', 'DDPG-TCOA', 'DQN-BOA', 'LE', 'FO', 'RO']
    
    for strategy in strategies:
        costs = results_dict['strategy_costs'][strategy]
        ax.plot(weights, costs, marker='o', label=strategy, linewidth=2)
    
    ax.set_xlabel('Weight on AoI (λ_A)')
    ax.set_ylabel('Total System Cost')
    ax.set_title('(b) Cost under different strategies and weight on AoI')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    plt.close()


def plot_uav_energy_impact(results_dict, save_path='plots/fig10_uav_energy.png'):
    """
    Plot Figure 10: Impact of UAV energy on cost
    
    Args:
        results_dict: Dictionary with UAV energy values and corresponding costs
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    uav_energies = results_dict['uav_energies']
    
    ax.plot(uav_energies, results_dict['total_cost'], 
            marker='o', label='Total Cost', linewidth=2, markersize=6)
    ax.plot(uav_energies, results_dict['aoi_cost'], 
            marker='s', label='AoI Cost', linewidth=2, markersize=6)
    ax.plot(uav_energies, results_dict['rental_cost'], 
            marker='^', label='Rental Price', linewidth=2, markersize=6)
    ax.plot(uav_energies, results_dict['penalty'], 
            marker='d', label='Penalty', linewidth=2, markersize=6)
    
    ax.set_xlabel('UAV Maximum Available Energy (J)')
    ax.set_ylabel('Cost')
    ax.set_title('Impact of UAV energy on cost')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    plt.close()


def plot_vehicle_number_impact(results_dict, save_path='plots/fig11_vehicle_number.png'):
    """
    Plot Figure 11: Impact of vehicle numbers on cost
    
    Args:
        results_dict: Dictionary with vehicle numbers and costs
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    vehicle_numbers = results_dict['vehicle_numbers']
    strategies = ['DRL-TCOA', 'DDPG-TCOA', 'DQN-BOA', 'LE', 'FO', 'RO']
    
    for strategy in strategies:
        if strategy in results_dict:
            costs = results_dict[strategy]
            ax.plot(vehicle_numbers, costs, marker='o', 
                   label=strategy, linewidth=2, markersize=6)
    
    ax.set_xlabel('Vehicle Number (N)')
    ax.set_ylabel('Total System Cost')
    ax.set_title('Impact of vehicle number N on cost')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    plt.close()


def plot_task_arrival_impact(results_dict, save_path='plots/fig12_task_arrival.png'):
    """
    Plot Figure 12: Impact of task arrival rate on cost
    
    Args:
        results_dict: Dictionary with arrival rates and costs
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    arrival_rates = results_dict['arrival_rates']
    strategies = ['DRL-TCOA', 'DDPG-TCOA', 'DQN-BOA', 'LE', 'FO', 'RO']
    
    # (a) Total cost
    ax = axes[0]
    for strategy in strategies:
        if strategy in results_dict:
            costs = results_dict[strategy]['total']
            ax.plot(arrival_rates, costs, marker='o', 
                   label=strategy, linewidth=2, markersize=6)
    
    ax.set_xlabel('Task Arrival Rate (λ_u)')
    ax.set_ylabel('Total System Cost')
    ax.set_title('(a) Cost under different strategies with λ_u')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (b) AoI cost
    ax = axes[1]
    for strategy in strategies:
        if strategy in results_dict:
            costs = results_dict[strategy]['aoi']
            ax.plot(arrival_rates, costs, marker='o', 
                   label=strategy, linewidth=2, markersize=6)
    
    ax.set_xlabel('Task Arrival Rate (λ_u)')
    ax.set_ylabel('AoI Cost')
    ax.set_title('(b) AoI cost under different strategies with λ_u')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    plt.close()


def plot_data_size_impact(results_dict, save_path='plots/fig13_data_size.png'):
    """
    Plot Figure 13: Impact of average data size on cost
    
    Args:
        results_dict: Dictionary with data sizes and costs
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    data_sizes = results_dict['data_sizes']
    strategies = ['DRL-TCOA', 'DDPG-TCOA', 'DQN-BOA', 'LE', 'FO', 'RO']
    
    # (a) Total cost
    ax = axes[0]
    for strategy in strategies:
        if strategy in results_dict:
            costs = results_dict[strategy]['total']
            ax.plot(data_sizes, costs, marker='o', 
                   label=strategy, linewidth=2, markersize=6)
    
    ax.set_xlabel('Average Data Size (kB)')
    ax.set_ylabel('Total System Cost')
    ax.set_title('(a) Cost under different strategies')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (b) AoI cost
    ax = axes[1]
    for strategy in strategies:
        if strategy in results_dict:
            costs = results_dict[strategy]['aoi']
            ax.plot(data_sizes, costs, marker='o', 
                   label=strategy, linewidth=2, markersize=6)
    
    ax.set_xlabel('Average Data Size (kB)')
    ax.set_ylabel('AoI Cost')
    ax.set_title('(b) AoI cost under different strategies')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    plt.close()


def load_and_plot_training_results(results_dir='results'):
    """
    Load training results and create basic plots
    
    Args:
        results_dir: Directory containing training results
    """
    # Load metrics
    metrics_path = os.path.join(results_dir, 'training_metrics.json')
    
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found: {metrics_path}")
        return
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Plot episode rewards
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Episode rewards
    ax = axes[0, 0]
    rewards = metrics['episode_rewards']
    ax.plot(rewards, alpha=0.3, color='blue')
    # Moving average
    window = 10
    if len(rewards) > window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards)), smoothed, color='blue', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Episode Rewards')
    ax.grid(True, alpha=0.3)
    
    # Episode costs
    ax = axes[0, 1]
    costs = metrics['episode_costs']
    ax.plot(costs, alpha=0.3, color='red')
    if len(costs) > window:
        smoothed = np.convolve(costs, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(costs)), smoothed, color='red', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Cost')
    ax.set_title('Episode Costs')
    ax.grid(True, alpha=0.3)
    
    # Critic loss
    ax = axes[1, 0]
    if len(metrics['critic_losses']) > 0:
        losses = metrics['critic_losses']
        ax.plot(losses, alpha=0.3, color='green')
        window = 50
        if len(losses) > window:
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(losses)), smoothed, color='green', linewidth=2)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Critic Loss')
    ax.set_title('Critic Loss')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Actor loss
    ax = axes[1, 1]
    if len(metrics['actor_losses']) > 0:
        losses = [l for l in metrics['actor_losses'] if l is not None]
        if len(losses) > 0:
            ax.plot(losses, alpha=0.3, color='purple')
            window = 50
            if len(losses) > window:
                smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(losses)), smoothed, color='purple', linewidth=2)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Actor Loss')
    ax.set_title('Actor Loss')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'training_progress.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved training progress plot to {save_path}")
    plt.close()


if __name__ == "__main__":
    print("Plotting training results...")
    load_and_plot_training_results('results')
    print("Done!")
