"""
Configuration file for UAV-Assisted VEC DRL-TCOA Algorithm
All parameters are taken directly from Tables II and III of the paper
"""

import numpy as np

# ====================================================================================
# Table II: TD3 Hyperparameters
# ====================================================================================
TD3_CONFIG = {
    'actor_lr': 5e-3,              # Learning rate for actor
    'critic_lr': 5e-3,             # Learning rate for critic
    'discount_factor': 0.999,        # Discount factor ζ
    'tau': 0.005,                   # Soft update rate ψ
    'policy_noise': 0.1,            # Policy noise for target smoothing
    'noise_clip': 0.5,              # Noise clip value c
    'policy_delay': 2,              # Delayed policy updates k
    'batch_size': 32,              # Mini-batch size
    'buffer_size': 1000000,         # Replay buffer size
    'hidden_layers': [600, 400],    # Hidden layer neurons
    'max_episodes': 1000,           # Maximum training episodes
    'max_steps': 100,               # Maximum steps per episode
}

# ====================================================================================
# Table III: VEC Network Parameters
# ====================================================================================
NETWORK_CONFIG = {
    # Simulation parameters
    'time_slot_duration': 0.05,     # τ = 0.05 seconds
    'carrier_frequency': 1.95e9,       # f = 1.95 GHz
    'speed_of_light': 3e8,          # C = 3 × 10^8 m/s
    
    # UAV parameters
    'uav_height': 100,              # H = 100 m
    'uav_max_velocity': 50,         # v_max = 50 m/s
    'uav_max_energy': 3000,         # E_max_U = 3000 J
    'uav_compute_capacity': 10e9,   # F_U = 10 GHz
    'uav_bandwidth': 5e6,           # W_U = 5 MHz
    'uav_max_associations': 5,      # N_max_U = 5
    'uav_rental_price': 0.5,        # P_U = 0.5 pence/CPU cycle
    
    # RSU parameters
    'rsu_compute_capacity': 10e9,   # F_R = 10 GHz
    'rsu_bandwidth': 5e6,           # W_R = 5 MHz
    'rsu_max_associations': 5,      # N_max_M = 5
    'rsu_rental_price': 0.3,        # P_R = 0.3 pence/CPU cycle
    'rsu_coverage_radius': 100,     # 100 m
    
    # Vehicle parameters
    'vehicle_compute_capacity': 1e9,    # F_n = 1 GHz
    'vehicle_transmit_power': 0.5,      # p_n = 0.5 W
    'vehicle_velocity_range': [10, 20], # 10-20 m/s
    
    # Channel parameters
    'awgn_power_density': 3.98e-21,     # N_0 = -90 dBm/Hz
    'eta_los': 1,                   # LoS path loss (dB)
    'eta_nlos': 20,                 # NLoS path loss (dB)
    'eta_rayleigh': 3,              # Rayleigh fading coefficient
    'a_param': 9.61,                # Environment parameter a
    'b_param': 0.16,                # Environment parameter b
    
    # Computation parameters
    'cpu_cycles_per_bit': 1000,     # X_c = 1000 cycles/bit
    'kappa_vehicle': 1e-28,         # k_n = 10^-28 (effective capacitance)
    'kappa_uav': 1e-28,             # k_u = 10^-28
    
    # Task parameters
    'task_data_size_mean': 500e3,   # Mean task size: 500 kB (in bits)
    'task_data_size_std': 100e3,    # Std of task size: 100 kB
    'task_max_latency': 0.5,        # T_max = 1 second
    'task_arrival_rate': 0.6,       # λ_u = 0.6 (Poisson parameter)
    
    # UAV energy model parameters (from equation 20)
    'P0': 79.8563,                  # Blade profile power (W)
    'Pi': 88.6279,                  # Induced power in hovering (W)
    'Utip': 120,                    # Tip speed of rotor blade (m/s)
    'v0': 4.03,                     # Mean rotor induced velocity (m/s)
    'd0': 0.6,                      # Fuselage drag ratio
    'rho0': 1.225,                  # Air density (kg/m^3)
    's0': 0.05,                     # Rotor solidity
    'A0': 0.503,                    # Rotor disc area (m^2)
    
    # Cost weights (normal scenario as default)
    'gamma_A': 0.33,                # Weight for AoI cost
    'gamma_E': 0.33,                # Weight for energy consumption
    'gamma_P': 0.33,                # Weight for rental price
    
    # Penalty constants
    'c1': 1000,                     # Penalty for association constraint violation
    'c2': 1000,                     # Penalty for latency constraint violation
    'c3': 1000,                     # Penalty for energy constraint violation
}

# ====================================================================================
# Map Configurations (from Figure 4)
# ====================================================================================
MAP_CONFIGS = {
    'map1': {
        'description': '600m two-way lane with 3 RSUs',
        'area': (600, 200),
        'rsu_positions': [
            (100, 200),
            (300, 200),
            (500, 200),
        ],
        'num_rsus': 3,
    },
    'map2': {
        'description': '400m × 400m crossroad with 2 RSUs',
        'area': (400, 400),
        'rsu_positions': [
            (100, 100),
            (300, 300),
        ],
        'num_rsus': 2,
    },
    'map3': {
        'description': '600m × 400m with 3 RSUs',
        'area': (600, 400),
        'rsu_positions': [
            (100, 100),
            (300, 300),
            (500, 100),
        ],
        'num_rsus': 3,
    }
}

# ====================================================================================
# Scenario Configurations (from Figure 7)
# ====================================================================================
SCENARIO_CONFIGS = {
    'normal': {
        'gamma_A': 0.33,
        'gamma_E': 0.33,
        'gamma_P': 0.33,
    },
    'time_sensitive': {
        'gamma_A': 0.8,
        'gamma_E': 0.1,
        'gamma_P': 0.1,
    },
    'energy_deficiency': {
        'gamma_A': 0.1,
        'gamma_E': 0.8,
        'gamma_P': 0.1,
    },
    'fund_poor': {
        'gamma_A': 0.1,
        'gamma_E': 0.1,
        'gamma_P': 0.8,
    },
    'energy_rich': {
        'gamma_A': 0.45,
        'gamma_E': 0.1,
        'gamma_P': 0.45,
    }
}

def get_config(scenario='normal', map_name='map1', num_vehicles=10):
    """
    Get complete configuration for a specific scenario
    
    Args:
        scenario: Scenario name from SCENARIO_CONFIGS
        map_name: Map name from MAP_CONFIGS
        num_vehicles: Number of vehicles in the simulation
        
    Returns:
        Complete configuration dictionary
    """
    config = {
        'td3': TD3_CONFIG.copy(),
        'network': NETWORK_CONFIG.copy(),
        'map': MAP_CONFIGS[map_name].copy(),
        'num_vehicles': num_vehicles,
    }
    
    # Update cost weights based on scenario
    if scenario in SCENARIO_CONFIGS:
        config['network'].update(SCENARIO_CONFIGS[scenario])
    
    return config
