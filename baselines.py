"""
Baseline Strategies for Comparison
Implements: DDPG-TCOA, DQN-BOA, LE, FO, RO
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import ReplayBuffer


class DDPGActor(nn.Module):
    """DDPG Actor Network"""
    def __init__(self, state_dim, action_dim, max_action):
        super(DDPGActor, self).__init__()
        self.max_action = max_action
        
        self.fc1 = nn.Linear(state_dim, 600)
        self.fc2 = nn.Linear(600, 400)
        self.fc3 = nn.Linear(400, action_dim)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x * self.max_action


class DDPGCritic(nn.Module):
    """DDPG Critic Network"""
    def __init__(self, state_dim, action_dim):
        super(DDPGCritic, self).__init__()
        
        self.fc1 = nn.Linear(state_dim + action_dim, 600)
        self.fc2 = nn.Linear(600, 400)
        self.fc3 = nn.Linear(400, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DDPGAgent:
    """DDPG Agent for comparison"""
    def __init__(self, state_dim, action_dim, max_action, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = DDPGActor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = DDPGActor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['actor_lr'])
        
        self.critic = DDPGCritic(state_dim, action_dim).to(self.device)
        self.critic_target = DDPGCritic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['critic_lr'])
        
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, config['buffer_size'])
        
        self.discount = config['discount_factor']
        self.tau = config['tau']
        self.batch_size = config['batch_size']
        self.max_action = max_action
        self.action_dim = action_dim
    
    def select_action(self, state, add_noise=True, noise_scale=0.1):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        
        if add_noise:
            noise = np.random.normal(0, noise_scale * self.max_action, size=self.action_dim)
            action = action + noise
            action = np.clip(action, 0, self.max_action)
        
        return action
    
    def train(self):
        if self.replay_buffer.size < self.batch_size:
            return {'critic_loss': None, 'actor_loss': None}
        
        state, action, next_state, reward, done = self.replay_buffer.sample(self.batch_size)
        
        # Critic update
        with torch.no_grad():
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (1 - done) * self.discount * target_Q
        
        current_Q = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Target update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }


class DQNNetwork(nn.Module):
    """DQN Network"""
    def __init__(self, state_dim, action_dim):
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, 600)
        self.fc2 = nn.Linear(600, 400)
        self.fc3 = nn.Linear(400, action_dim)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """
    DQN Agent with discrete actions
    Action space: {-1, 0, 1} for UAV direction × {0, 1, 2}^N for vehicle offloading
    """
    def __init__(self, state_dim, num_vehicles, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_vehicles = num_vehicles
        
        # Discrete action space: 3 UAV directions × 3^N vehicle decisions
        # For practical reasons, limit N or use approximation
        self.num_uav_actions = 3  # -1, 0, 1 (left, stay, right)
        self.num_vehicle_actions = 3  # 0: local, 1: RSU, 2: UAV
        
        # Simplified: Joint action index
        total_actions = self.num_uav_actions * (self.num_vehicle_actions ** min(num_vehicles, 5))
        
        self.q_network = DQNNetwork(state_dim, total_actions).to(self.device)
        self.target_network = DQNNetwork(state_dim, total_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=config['critic_lr'])
        
        self.discount = config['discount_factor']
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.tau = config['tau']
        self.total_actions = total_actions
    
    def select_action(self, state, add_noise=True):
        if add_noise and np.random.random() < self.epsilon:
            action_idx = np.random.randint(0, self.total_actions)
        else:
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state)
                action_idx = q_values.argmax().item()
        
        # Decode action
        return self._decode_action(action_idx)
    
    def _decode_action(self, action_idx):
        """Decode discrete action index to continuous action"""
        # Simplified decoding
        # This is a placeholder - full implementation would need careful design
        uav_action_idx = action_idx % self.num_uav_actions
        vehicle_action = action_idx // self.num_uav_actions
        
        # UAV: map {0,1,2} to direction angle
        uav_angle = (uav_action_idx - 1) * np.pi / 4 + np.pi
        uav_velocity = 25.0  # Fixed velocity
        
        # Vehicles: random assignment for simplification
        vehicle_actions = np.random.rand(self.num_vehicles)
        
        return np.array([uav_velocity, uav_angle] + list(vehicle_actions))


class LocalExecutionStrategy:
    """Local Execution (LE) baseline - all tasks computed locally"""
    def __init__(self, action_dim):
        self.action_dim = action_dim
    
    def select_action(self, state, add_noise=True):
        # UAV hovers (velocity = 0)
        # All alpha_n = 0 (local execution)
        return np.array([0.0, 0.0] + [0.0] * (self.action_dim - 2))


class FullOffloadingStrategy:
    """Full Offloading (FO) baseline - all tasks offloaded to RSUs"""
    def __init__(self, action_dim):
        self.action_dim = action_dim
    
    def select_action(self, state, add_noise=True):
        # UAV hovers
        # All alpha_n = 0.5 (offload to RSU)
        return np.array([0.0, 0.0] + [0.5] * (self.action_dim - 2))


class RandomOffloadingStrategy:
    """Random Offloading (RO) baseline - random decisions"""
    def __init__(self, action_dim, max_action):
        self.action_dim = action_dim
        self.max_action = max_action
    
    def select_action(self, state, add_noise=True):
        # Random UAV movement
        v_u = np.random.uniform(0, self.max_action[0])
        theta_u = np.random.uniform(0, 2 * np.pi)
        
        # Random offloading decisions
        alphas = np.random.rand(self.action_dim - 2)
        
        return np.array([v_u, theta_u] + list(alphas))
