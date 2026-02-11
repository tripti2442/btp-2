"""
TD3 Networks: Actor and Critic networks
Implements the neural network architecture from Table II
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    Actor network for TD3
    Architecture: Input -> [600, 400] -> Output
    Activation: ReLU for hidden layers, Tanh for output
    """
    
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        # self.max_action = max_action
        self.register_buffer(
            "max_action",
            torch.tensor(max_action, dtype=torch.float32)
        )

        
        # Hidden layers with [600, 400] neurons as specified in Table II
        self.fc1 = nn.Linear(state_dim, 600)
        self.fc2 = nn.Linear(600, 400)
        self.fc3 = nn.Linear(400, action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
    
    def forward(self, state):
        """
        Forward pass
        
        Args:
            state: State tensor
            
        Returns:
            Action tensor with Tanh activation
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        
        # Scale action to appropriate range
        # Output is in [-1, 1], need to scale to [0, max_action] for each dimension
        return x * self.max_action


class Critic(nn.Module):
    """
    Critic network for TD3
    Architecture: Input (state+action) -> [600, 400] -> Output (Q-value)
    Activation: ReLU for hidden layers
    """
    
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.fc1 = nn.Linear(state_dim + action_dim, 600)
        self.fc2 = nn.Linear(600, 400)
        self.fc3 = nn.Linear(400, 1)
        
        # Q2 architecture (for twin critics)
        self.fc4 = nn.Linear(state_dim + action_dim, 600)
        self.fc5 = nn.Linear(600, 400)
        self.fc6 = nn.Linear(400, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.xavier_uniform_(self.fc5.weight)
        nn.init.xavier_uniform_(self.fc6.weight)
    
    def forward(self, state, action):
        """
        Forward pass for both Q networks
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Q1 and Q2 values
        """
        sa = torch.cat([state, action], dim=1)
        
        # Q1 forward pass
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        
        # Q2 forward pass
        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        
        return q1, q2
    
    def Q1(self, state, action):
        """
        Return only Q1 value (used for actor update)
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Q1 value
        """
        sa = torch.cat([state, action], dim=1)
        
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        
        return q1


class ReplayBuffer:
    """
    Replay buffer for storing and sampling transitions
    """
    
    def __init__(self, state_dim, action_dim, max_size=1000000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.state = torch.zeros((max_size, state_dim))
        self.action = torch.zeros((max_size, action_dim))
        self.next_state = torch.zeros((max_size, state_dim))
        self.reward = torch.zeros((max_size, 1))
        self.done = torch.zeros((max_size, 1))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def add(self, state, action, next_state, reward, done):
        """
        Add a transition to the buffer
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            reward: Reward received
            done: Done flag
        """
        self.state[self.ptr] = torch.FloatTensor(state)
        self.action[self.ptr] = torch.FloatTensor(action)
        self.next_state[self.ptr] = torch.FloatTensor(next_state)
        self.reward[self.ptr] = torch.FloatTensor([reward])
        self.done[self.ptr] = torch.FloatTensor([done])
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions
        
        Args:
            batch_size: Size of the batch
            
        Returns:
            Batch of (state, action, next_state, reward, done)
        """
        ind = torch.randint(0, self.size, size=(batch_size,))
        
        return (
            self.state[ind].to(self.device),
            self.action[ind].to(self.device),
            self.next_state[ind].to(self.device),
            self.reward[ind].to(self.device),
            self.done[ind].to(self.device)
        )
