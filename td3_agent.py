"""
TD3 Agent Implementation
Implements Algorithm 1 from the paper exactly
"""

import torch
import torch.nn.functional as F
import numpy as np
from networks import Actor, Critic, ReplayBuffer


class TD3Agent:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent
    Implements Algorithm 1 from the paper
    """
    
    def __init__(self, state_dim, action_dim, max_action, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        # self.max_action = max_action
        self.max_action = torch.tensor(max_action, dtype=torch.float32, device=self.device)

        
        # Hyperparameters from Table II
        self.actor_lr = config['actor_lr']
        self.critic_lr = config['critic_lr']
        self.discount = config['discount_factor']
        self.tau = config['tau']
        self.policy_noise = config['policy_noise']
        self.noise_clip = config['noise_clip']
        self.policy_delay = config['policy_delay']
        self.batch_size = config['batch_size']
        
        # Initialize actor networks
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        
        # Initialize critic networks
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, config['buffer_size'])
        
        # Training step counter
        self.total_it = 0
    
    def select_action(self, state, add_noise=True, noise_scale=0.1):
        """
        Select action using the actor network
        
        Args:
            state: Current state
            add_noise: Whether to add exploration noise
            noise_scale: Scale of the exploration noise
            
        Returns:
            Action array
        """

        #TEMP
        # print("STATE RANGE:", state.min(), state.max())

        # Normalize state (simple robust scaling)
        state = state / (np.abs(state).max() + 1e-6)
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        # action = self.actor(state).cpu().data.numpy().flatten()
        action = self.actor(state).detach().cpu().numpy().flatten()

        #TEMP
        # print("ACTION STD:", np.std(action))

        
        if add_noise:
            # Add Gaussian noise for exploration (Line 7 in Algorithm 1)
            # noise = np.random.normal(0, noise_scale * self.max_action, size=self.action_dim)
            # action = action + noise
            # action = np.clip(action, 0, self.max_action)
            noise = np.random.normal(
                0,
                noise_scale * self.max_action.cpu().numpy(),
                size=self.action_dim
            )

            action = action + noise
            action = np.clip(action, 0, self.max_action.cpu().numpy())
        
        return action
    
    def train(self):
        """
        Train the agent for one step
        Implements Lines 10-18 of Algorithm 1
        
        Returns:
            Dictionary containing training metrics
        """
        if self.replay_buffer.size < self.batch_size:
          return {'critic_loss': None, 'actor_loss': None}
        self.total_it += 1
        
        # Sample mini-batch from replay buffer (Line 11)
        state, action, next_state, reward, done = self.replay_buffer.sample(self.batch_size)
        state = state / (state.abs().max(dim=1, keepdim=True)[0] + 1e-6)
        next_state = next_state / (next_state.abs().max(dim=1, keepdim=True)[0] + 1e-6)

        
        with torch.no_grad():
            # Select action with target policy and add clipped noise (Line 12)
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            
            # next_action = (self.actor_target(next_state) + noise).clamp(0, self.max_action)
            next_action = (self.actor_target(next_state) + noise).clamp(
                min=torch.zeros_like(self.max_action),
                max=self.max_action
            )

            
            # Compute target Q-value (Line 13)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.discount * target_Q
        
        # Get current Q estimates (Line 14)
        current_Q1, current_Q2 = self.critic(state, action)
        
        # Compute critic loss (Line 14)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # Optimize the critic (Line 14)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Delayed policy updates (Line 15)
        actor_loss = None
        if self.total_it % self.policy_delay == 0:
            # Compute actor loss (Line 16)
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # Optimize the actor (Line 16)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update target networks (Lines 17)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        metrics = {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item() if actor_loss is not None else None,
        }
        
        return metrics
    
    def save(self, filename):
        """
        Save model parameters
        
        Args:
            filename: Path to save the model
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filename)
    
    def load(self, filename):
        """
        Load model parameters
        
        Args:
            filename: Path to load the model from
        """
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
