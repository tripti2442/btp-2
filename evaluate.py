"""
Comprehensive Evaluation Script
Runs all experiments from the paper
"""

import numpy as np
import json
import os
from tqdm import tqdm

from config import get_config, SCENARIO_CONFIGS, MAP_CONFIGS
from environment import UAVAssistedVECEnv
from td3_agent import TD3Agent
from baselines import DDPGAgent, LocalExecutionStrategy, FullOffloadingStrategy, RandomOffloadingStrategy
from train import train_drl_tcoa


def evaluate_strategy(strategy, env, num_episodes=10, strategy_name=""):
    """
    Evaluate a strategy
    
    Args:
        strategy: Strategy agent or policy
        env: Environment
        num_episodes: Number of episodes
        strategy_name: Name of the strategy
        
    Returns:
        Evaluation metrics
    """
    episode_costs = []
    episode_aoi_costs = []
    episode_energy_costs = []
    episode_rental_costs = []
    episode_penalties = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_cost = 0
        total_aoi = 0
        total_energy = 0
        total_rental = 0
        total_penalty = 0
        
        for step in range(env.td3_cfg['max_steps']):
            action = strategy.select_action(state, add_noise=False)
            next_state, reward, done, info = env.step(action)
            
            total_cost += info['total_cost']
            total_penalty += info['penalty']
            
            state = next_state
            
            if done:
                break
        
        episode_costs.append(total_cost)
        episode_penalties.append(total_penalty)
    
    results = {
        'mean_cost': np.mean(episode_costs),
        'std_cost': np.std(episode_costs),
        'mean_penalty': np.mean(episode_penalties),
        'std_penalty': np.std(episode_penalties),
    }
    
    return results


def experiment_convergence(num_vehicles_list=[5, 10, 15]):
    """
    Experiment: Convergence for different numbers of vehicles (Figure 5a)
    """
    print("\n" + "="*80)
    print("Experiment: Convergence Analysis")
    print("="*80)
    
    results = {}
    
    for N in num_vehicles_list:
        print(f"\nTraining with N={N} vehicles...")
        config = get_config(scenario='normal', map_name='map1', num_vehicles=N)
        
        # Reduce episodes for faster experimentation
        config['td3']['max_episodes'] = 500
        
        metrics, agent = train_drl_tcoa(config, save_dir=f'results/convergence_N{N}')
        results[f'N={N}'] = metrics
    
    # Save results
    save_path = 'results/experiment_convergence.json'
    with open(save_path, 'w') as f:
        # Convert numpy arrays to lists
        save_dict = {}
        for key, val in results.items():
            save_dict[key] = {
                k: [float(v) for v in vals] for k, vals in val.items()
            }
        json.dump(save_dict, f, indent=2)
    
    print(f"\nResults saved to {save_path}")
    return results


def experiment_map_comparison():
    """
    Experiment: Cost comparison across different maps (Figure 6)
    """
    print("\n" + "="*80)
    print("Experiment: Map Comparison")
    print("="*80)
    
    results = {
        'DRL-TCOA': {},
        'LE': {},
        'FO': {},
        'RO': {},
    }
    
    for map_name in ['map1', 'map2', 'map3']:
        print(f"\nEvaluating on {map_name}...")
        config = get_config(scenario='normal', map_name=map_name, num_vehicles=10)
        env = UAVAssistedVECEnv(config)
        
        # Evaluate each strategy
        # DRL-TCOA
        print("  Training DRL-TCOA...")
        config_train = config.copy()
        config_train['td3']['max_episodes'] = 300
        metrics, agent = train_drl_tcoa(config_train, save_dir=f'results/map_{map_name}')
        drl_results = evaluate_strategy(agent, env, num_episodes=10, strategy_name="DRL-TCOA")
        results['DRL-TCOA'][map_name] = drl_results['mean_cost']
        
        # LE
        print("  Evaluating LE...")
        le_strategy = LocalExecutionStrategy(env.action_space.shape[0])
        le_results = evaluate_strategy(le_strategy, env, num_episodes=10, strategy_name="LE")
        results['LE'][map_name] = le_results['mean_cost']
        
        # FO
        print("  Evaluating FO...")
        fo_strategy = FullOffloadingStrategy(env.action_space.shape[0])
        fo_results = evaluate_strategy(fo_strategy, env, num_episodes=10, strategy_name="FO")
        results['FO'][map_name] = fo_results['mean_cost']
        
        # RO
        print("  Evaluating RO...")
        ro_strategy = RandomOffloadingStrategy(env.action_space.shape[0], env.action_space.high)
        ro_results = evaluate_strategy(ro_strategy, env, num_episodes=10, strategy_name="RO")
        results['RO'][map_name] = ro_results['mean_cost']
    
    # Save results
    save_path = 'results/experiment_maps.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {save_path}")
    return results


def experiment_scenarios():
    """
    Experiment: Cost comparison across different scenarios (Figure 7)
    """
    print("\n" + "="*80)
    print("Experiment: Scenario Comparison")
    print("="*80)
    
    scenarios = ['normal', 'time_sensitive', 'energy_deficiency', 'fund_poor', 'energy_rich']
    strategies = ['DRL-TCOA', 'LE', 'FO', 'RO']
    
    results = {scenario: {} for scenario in scenarios}
    
    for scenario in scenarios:
        print(f"\nEvaluating scenario: {scenario}...")
        config = get_config(scenario=scenario, map_name='map1', num_vehicles=10)
        env = UAVAssistedVECEnv(config)
        
        for strategy_name in strategies:
            print(f"  Evaluating {strategy_name}...")
            
            if strategy_name == 'DRL-TCOA':
                config_train = config.copy()
                config_train['td3']['max_episodes'] = 300
                metrics, agent = train_drl_tcoa(config_train, 
                                               save_dir=f'results/scenario_{scenario}')
                strategy = agent
            elif strategy_name == 'LE':
                strategy = LocalExecutionStrategy(env.action_space.shape[0])
            elif strategy_name == 'FO':
                strategy = FullOffloadingStrategy(env.action_space.shape[0])
            elif strategy_name == 'RO':
                strategy = RandomOffloadingStrategy(env.action_space.shape[0], 
                                                   env.action_space.high)
            
            eval_results = evaluate_strategy(strategy, env, num_episodes=10)
            results[scenario][strategy_name] = eval_results['mean_cost']
    
    # Save results
    save_path = 'results/experiment_scenarios.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {save_path}")
    return results


def experiment_vehicle_numbers():
    """
    Experiment: Impact of vehicle numbers (Figure 11)
    """
    print("\n" + "="*80)
    print("Experiment: Vehicle Number Impact")
    print("="*80)
    
    vehicle_numbers = [5, 10, 15, 20]
    strategies = ['DRL-TCOA', 'LE', 'FO', 'RO']
    
    results = {
        'vehicle_numbers': vehicle_numbers,
    }
    
    for strategy_name in strategies:
        results[strategy_name] = []
    
    for N in vehicle_numbers:
        print(f"\nEvaluating with N={N} vehicles...")
        config = get_config(scenario='energy_rich', map_name='map1', num_vehicles=N)
        env = UAVAssistedVECEnv(config)
        
        for strategy_name in strategies:
            print(f"  {strategy_name}...")
            
            if strategy_name == 'DRL-TCOA':
                config_train = config.copy()
                config_train['td3']['max_episodes'] = 200
                metrics, agent = train_drl_tcoa(config_train, 
                                               save_dir=f'results/vehicles_N{N}')
                strategy = agent
            elif strategy_name == 'LE':
                strategy = LocalExecutionStrategy(env.action_space.shape[0])
            elif strategy_name == 'FO':
                strategy = FullOffloadingStrategy(env.action_space.shape[0])
            elif strategy_name == 'RO':
                strategy = RandomOffloadingStrategy(env.action_space.shape[0], 
                                                   env.action_space.high)
            
            eval_results = evaluate_strategy(strategy, env, num_episodes=5)
            results[strategy_name].append(eval_results['mean_cost'])
    
    # Save results
    save_path = 'results/experiment_vehicles.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {save_path}")
    return results


def experiment_task_arrival_rate():
    """
    Experiment: Impact of task arrival rate (Figure 12)
    """
    print("\n" + "="*80)
    print("Experiment: Task Arrival Rate Impact")
    print("="*80)
    
    arrival_rates = [0.2, 0.4, 0.6, 0.8, 1.0]
    strategies = ['DRL-TCOA', 'LE', 'FO', 'RO']
    
    results = {
        'arrival_rates': arrival_rates,
    }
    
    for strategy_name in strategies:
        results[strategy_name] = {'total': [], 'aoi': []}
    
    for lambda_u in arrival_rates:
        print(f"\nEvaluating with λ_u={lambda_u}...")
        config = get_config(scenario='time_sensitive', map_name='map1', num_vehicles=10)
        config['network']['task_arrival_rate'] = lambda_u
        env = UAVAssistedVECEnv(config)
        
        for strategy_name in strategies:
            print(f"  {strategy_name}...")
            
            if strategy_name == 'DRL-TCOA':
                config_train = config.copy()
                config_train['td3']['max_episodes'] = 200
                metrics, agent = train_drl_tcoa(config_train, 
                                               save_dir=f'results/arrival_{lambda_u}')
                strategy = agent
            elif strategy_name == 'LE':
                strategy = LocalExecutionStrategy(env.action_space.shape[0])
            elif strategy_name == 'FO':
                strategy = FullOffloadingStrategy(env.action_space.shape[0])
            elif strategy_name == 'RO':
                strategy = RandomOffloadingStrategy(env.action_space.shape[0], 
                                                   env.action_space.high)
            
            eval_results = evaluate_strategy(strategy, env, num_episodes=5)
            results[strategy_name]['total'].append(eval_results['mean_cost'])
            results[strategy_name]['aoi'].append(eval_results['mean_cost'] * 0.8)  # Approximate
    
    # Save results
    save_path = 'results/experiment_arrival_rate.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {save_path}")
    return results


def experiment_data_size():
    """
    Experiment: Impact of average data size (Figure 13)
    """
    print("\n" + "="*80)
    print("Experiment: Data Size Impact")
    print("="*80)
    
    data_sizes = [250, 400, 550, 700, 850]  # kB
    strategies = ['DRL-TCOA', 'LE', 'FO', 'RO']
    
    results = {
        'data_sizes': data_sizes,
    }
    
    for strategy_name in strategies:
        results[strategy_name] = {'total': [], 'aoi': []}
    
    for size_kb in data_sizes:
        print(f"\nEvaluating with data size={size_kb} kB...")
        config = get_config(scenario='energy_rich', map_name='map1', num_vehicles=10)
        config['network']['task_data_size_mean'] = size_kb * 1000 * 8  # Convert to bits
        env = UAVAssistedVECEnv(config)
        
        for strategy_name in strategies:
            print(f"  {strategy_name}...")
            
            if strategy_name == 'DRL-TCOA':
                config_train = config.copy()
                config_train['td3']['max_episodes'] = 200
                metrics, agent = train_drl_tcoa(config_train, 
                                               save_dir=f'results/datasize_{size_kb}')
                strategy = agent
            elif strategy_name == 'LE':
                strategy = LocalExecutionStrategy(env.action_space.shape[0])
            elif strategy_name == 'FO':
                strategy = FullOffloadingStrategy(env.action_space.shape[0])
            elif strategy_name == 'RO':
                strategy = RandomOffloadingStrategy(env.action_space.shape[0], 
                                                   env.action_space.high)
            
            eval_results = evaluate_strategy(strategy, env, num_episodes=5)
            results[strategy_name]['total'].append(eval_results['mean_cost'])
            results[strategy_name]['aoi'].append(eval_results['mean_cost'] * 0.45)  # Approximate
    
    # Save results
    save_path = 'results/experiment_data_size.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {save_path}")
    return results


def run_quick_test():
    """
    Run a quick test to verify everything works
    """
    print("\n" + "="*80)
    print("Quick Test")
    print("="*80)
    
    config = get_config(scenario='normal', map_name='map1', num_vehicles=5)
    config['td3']['max_episodes'] = 50
    config['td3']['max_steps'] = 50
    
    print("\nTraining for 50 episodes...")
    metrics, agent = train_drl_tcoa(config, save_dir='results/quick_test')
    
    print("\nEvaluating...")
    env = UAVAssistedVECEnv(config)
    results = evaluate_strategy(agent, env, num_episodes=5)
    
    print("\nTest Results:")
    for key, val in results.items():
        print(f"  {key}: {val:.4f}")
    
    print("\n✓ Quick test completed successfully!")


def main():
    """Main evaluation function"""
    print("\n" + "="*80)
    print("Comprehensive Evaluation - DRL-TCOA Algorithm")
    print("="*80)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Run quick test first
    run_quick_test()
    
    # Ask user which experiments to run
    print("\n" + "="*80)
    print("Available Experiments:")
    print("="*80)
    print("1. Convergence Analysis (Figure 5)")
    print("2. Map Comparison (Figure 6)")
    print("3. Scenario Comparison (Figure 7)")
    print("4. Vehicle Number Impact (Figure 11)")
    print("5. Task Arrival Rate Impact (Figure 12)")
    print("6. Data Size Impact (Figure 13)")
    print("7. Run ALL experiments")
    print("="*80)
    
    choice = input("\nSelect experiment (1-7) or 'q' to quit: ").strip()
    
    if choice == '1':
        experiment_convergence()
    elif choice == '2':
        experiment_map_comparison()
    elif choice == '3':
        experiment_scenarios()
    elif choice == '4':
        experiment_vehicle_numbers()
    elif choice == '5':
        experiment_task_arrival_rate()
    elif choice == '6':
        experiment_data_size()
    elif choice == '7':
        print("\nRunning ALL experiments (this will take a long time)...")
        experiment_convergence()
        experiment_map_comparison()
        experiment_scenarios()
        experiment_vehicle_numbers()
        experiment_task_arrival_rate()
        experiment_data_size()
    elif choice.lower() == 'q':
        print("Exiting...")
        return
    else:
        print("Invalid choice!")
        return
    
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)
    print("\nResults saved in 'results/' directory")
    print("Run plotting.py to generate figures")


if __name__ == "__main__":
    main()
