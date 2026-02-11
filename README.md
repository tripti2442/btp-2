# DRL-TCOA: Deep Reinforcement Learning-Based Computation Offloading in UAV-Assisted VEC Networks

This is a complete implementation of the paper:
**"Deep-Reinforcement-Learning-Based Computation Offloading in UAV-Assisted Vehicular Edge Computing Networks"**
by Junjie Yan, Xiaohui Zhao, and Zan Li (IEEE Internet of Things Journal, 2024)

## Overview

This implementation provides:
- Complete UAV-Assisted VEC environment with all physical models
- TD3-based DRL-TCOA algorithm (Algorithm 1 from paper)
- All baseline strategies (DDPG-TCOA, DQN-BOA, LE, FO, RO)
- Reproduction of all experiments and figures from the paper
- Exact parameters from Tables II and III

## Project Structure

```
.
├── config.py              # All configuration parameters (Tables II & III)
├── environment.py         # UAV-Assisted VEC environment (equations 1-26)
├── networks.py           # Neural network architectures
├── td3_agent.py          # TD3 agent implementation
├── baselines.py          # Baseline strategies for comparison
├── train.py              # Main training script (Algorithm 1)
├── evaluate.py           # Comprehensive evaluation script
├── plotting.py           # Plotting functions for all figures
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

1. Install Python 3.8 or higher

2. Install PyTorch:
   ```bash
   # For CPU
   pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
   
   # For CUDA (GPU)
   pip install torch==2.0.1
   ```

3. Install other dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Basic Training

Train the DRL-TCOA algorithm with default settings:

```bash
python train.py
```

This will:
- Train for 1000 episodes with 100 steps each
- Use Map1 with 10 vehicles
- Save model to `results/td3_model.pth`
- Save training metrics to `results/training_metrics.json`

### 2. View Training Progress

After training, plot the results:

```bash
python plotting.py
```

This generates `results/training_progress.png` showing:
- Episode rewards over time
- Episode costs over time
- Critic and actor losses

### 3. Run Experiments

Run comprehensive experiments to reproduce paper figures:

```bash
python evaluate.py
```

Choose from:
1. Convergence Analysis (Figure 5)
2. Map Comparison (Figure 6)
3. Scenario Comparison (Figure 7)
4. Vehicle Number Impact (Figure 11)
5. Task Arrival Rate Impact (Figure 12)
6. Data Size Impact (Figure 13)
7. Run ALL experiments

## Configuration

### Network Parameters (Table III)

All parameters match the paper exactly. Edit `config.py` to modify:

**UAV Parameters:**
- Height: 100 m
- Max velocity: 50 m/s
- Max energy: 3000 J
- Compute capacity: 10 GHz
- Bandwidth: 5 MHz

**RSU Parameters:**
- Compute capacity: 10 GHz
- Bandwidth: 5 MHz
- Coverage radius: 100 m

**Task Parameters:**
- Mean data size: 500 kB
- Arrival rate: 0.6 (Poisson)
- Max latency: 1.0 s

### TD3 Hyperparameters (Table II)

- Actor learning rate: 0.005
- Critic learning rate: 0.005
- Discount factor: 0.99
- Hidden layers: [600, 400]
- Batch size: 256
- Replay buffer: 1M

### Scenarios

Five scenarios from the paper (Figure 7):

1. **Normal**: γ_A=0.33, γ_E=0.33, γ_P=0.33
2. **Time-sensitive**: γ_A=0.8, γ_E=0.1, γ_P=0.1
3. **Energy-deficiency**: γ_A=0.1, γ_E=0.8, γ_P=0.1
4. **Fund-poor**: γ_A=0.1, γ_E=0.1, γ_P=0.8
5. **Energy-rich**: γ_A=0.45, γ_E=0.1, γ_P=0.45

### Maps

Three map configurations (Figure 4):

1. **Map1**: 600m two-way lane with 3 RSUs
2. **Map2**: 400m × 400m crossroad with 2 RSUs
3. **Map3**: 600m × 400m with 3 RSUs

## Usage Examples

### Train with Custom Configuration

```python
from config import get_config
from train import train_drl_tcoa

# Get configuration
config = get_config(
    scenario='time_sensitive',
    map_name='map1',
    num_vehicles=10
)

# Modify if needed
config['td3']['max_episodes'] = 500
config['network']['task_arrival_rate'] = 0.8

# Train
metrics, agent = train_drl_tcoa(config, save_dir='my_results')
```

### Evaluate Trained Agent

```python
from environment import UAVAssistedVECEnv
from td3_agent import TD3Agent
import torch

# Load configuration
config = get_config(scenario='normal', map_name='map1', num_vehicles=10)

# Create environment
env = UAVAssistedVECEnv(config)

# Load trained agent
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high

agent = TD3Agent(state_dim, action_dim, max_action, config['td3'])
agent.load('results/td3_model.pth')

# Evaluate
state = env.reset()
total_reward = 0

for step in range(100):
    action = agent.select_action(state, add_noise=False)
    next_state, reward, done, info = env.step(action)
    total_reward += reward
    state = next_state
    if done:
        break

print(f"Total reward: {total_reward}")
print(f"Total cost: {info['total_cost']}")
```

### Compare Strategies

```python
from evaluate import evaluate_strategy
from baselines import LocalExecutionStrategy, FullOffloadingStrategy

# Create environment
env = UAVAssistedVECEnv(config)

# Load DRL-TCOA agent
agent = TD3Agent(...)
agent.load('results/td3_model.pth')

# Create baseline strategies
le_strategy = LocalExecutionStrategy(env.action_space.shape[0])
fo_strategy = FullOffloadingStrategy(env.action_space.shape[0])

# Evaluate
drl_results = evaluate_strategy(agent, env, num_episodes=10)
le_results = evaluate_strategy(le_strategy, env, num_episodes=10)
fo_results = evaluate_strategy(fo_strategy, env, num_episodes=10)

print("DRL-TCOA:", drl_results['mean_cost'])
print("LE:", le_results['mean_cost'])
print("FO:", fo_results['mean_cost'])
```

## Implementation Details

### Environment (environment.py)

Implements complete system model:
- **Communication Model**: A2G and G2G path loss (equations 3-7)
- **Computation Model**: Local and offloading computation (equations 8-18)
- **UAV Energy Model**: Propulsion energy (equation 20)
- **AoI Model**: Peak AoI calculation (equations 21-24)
- **Cost Function**: Weighted sum of AoI, energy, and rental price (equation 27)

### TD3 Agent (td3_agent.py)

Implements Algorithm 1 exactly:
- Twin delayed critic networks
- Target policy smoothing
- Delayed policy updates
- Clipped double Q-learning

### Network Architecture (networks.py)

Exact architecture from Table II:
- Actor: Input → [600, 400] → Output (Tanh)
- Critic: Input → [600, 400] → Q-value (Twin networks)
- ReLU activation for hidden layers

## Troubleshooting

### No Convergence

If the algorithm doesn't converge:

1. **Check learning rates**: Try reducing to 0.001
2. **Increase buffer size**: More diverse samples help
3. **Adjust reward scaling**: Normalize costs
4. **Check constraints**: Ensure penalties are appropriate

### High Loss Values

If critic loss keeps increasing:

1. **Gradient clipping**: Add in td3_agent.py
2. **Batch normalization**: Add to networks
3. **Target network update rate**: Reduce tau
4. **Replay buffer**: Ensure proper sampling

### Memory Issues

If running out of memory:

1. **Reduce batch size**: Try 128 or 64
2. **Reduce buffer size**: Try 500K
3. **Use CPU**: Set device to 'cpu'
4. **Reduce vehicles**: Start with N=5

## Expected Results

Based on the paper (approximate values):

**Figure 5 - Convergence:**
- DRL-TCOA converges in ~200 episodes
- DDPG-TCOA converges in ~400 episodes
- Stable rewards after convergence

**Figure 6 - Map Comparison:**
- DRL-TCOA: Lowest cost across all maps
- 56.6% cost reduction vs LE
- 12.4% cost reduction vs FO

**Figure 7 - Scenarios:**
- Time-sensitive: 62% reduction vs LE
- Energy-deficiency: 87.2% reduction vs LE
- Energy-rich: 33% reduction vs LE

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{yan2024deep,
  title={Deep-Reinforcement-Learning-Based Computation Offloading in UAV-Assisted Vehicular Edge Computing Networks},
  author={Yan, Junjie and Zhao, Xiaohui and Li, Zan},
  journal={IEEE Internet of Things Journal},
  volume={11},
  number={11},
  pages={19882--19897},
  year={2024},
  publisher={IEEE}
}
```

## License

This implementation is for research and educational purposes.

## Contact

For questions or issues:
1. Check the paper for algorithm details
2. Review the code comments for implementation details
3. Open an issue on GitHub

## Notes

- Training takes 2-6 hours depending on hardware
- GPU significantly speeds up training
- Results may vary slightly due to randomness
- Use fixed random seeds for reproducibility

## Acknowledgments

This implementation is based on the paper by Yan et al. (2024) published in IEEE IoT Journal.
All parameters, equations, and algorithms follow the paper exactly.
