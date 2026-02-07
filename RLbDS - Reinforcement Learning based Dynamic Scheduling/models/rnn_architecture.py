"""
Enhanced RNN architecture for DRL model
Based on the paper's Figure 3 architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUWithSkipConnections(nn.Module):
    """GRU layer with skip connections for better gradient flow"""
    
    def __init__(self, input_size, hidden_size):
        super(GRUWithSkipConnections, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.skip_projection = nn.Linear(input_size, hidden_size)
        
    def forward(self, x, hidden=None):
        """Forward pass with skip connection"""
        gru_out, hidden = self.gru(x, hidden)
        # Skip connection
        skip = self.skip_projection(x)
        output = gru_out + skip
        return output, hidden


class EnhancedRNN(nn.Module):
    """
    Enhanced RNN architecture as described in the paper
    Features:
    - Two fully connected layers (fc1, fc2)
    - Three recurrent layers (r1, r2, r3) with skip connections
    - Two output fully connected layers (fc3, fc4)
    - Dual outputs: action probabilities and cumulative loss
    """
    
    def __init__(self, input_size, hidden_size, num_tasks=100, num_hosts=100):
        super(EnhancedRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_tasks = num_tasks
        self.num_hosts = num_hosts
        
        # Initial fully connected layers
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        
        # Recurrent layers with skip connections
        self.r1 = GRUWithSkipConnections(256, hidden_size)
        self.r2 = GRUWithSkipConnections(hidden_size, hidden_size)
        self.r3 = GRUWithSkipConnections(hidden_size, hidden_size)
        
        # Output fully connected layers
        self.fc3 = nn.Linear(hidden_size, 128)
        
        # Action output (probability distribution)
        self.fc4_action = nn.Linear(128, num_tasks * num_hosts)
        
        # Value output (cumulative loss prediction)
        self.fc4_value = nn.Linear(128, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """
        Forward pass through the network
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
        Returns:
            action_probs: Probability distribution over task-host assignments
            value: Predicted cumulative loss
        """
        batch_size = x.size(0)
        
        # Flatten if needed (for 2D inputs)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # Initial FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Recurrent layers with skip connections
        x, _ = self.r1(x)
        x = self.dropout(x)
        
        x, _ = self.r2(x)
        x = self.dropout(x)
        
        x, _ = self.r3(x)
        
        # Take the last timestep output
        x = x[:, -1, :]
        
        # Output FC layer
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        
        # Action probabilities (task-host assignment)
        action_logits = self.fc4_action(x)
        action_logits = action_logits.view(batch_size, self.num_tasks, self.num_hosts)
        action_probs = F.softmax(action_logits, dim=2)  # Softmax over hosts
        
        # Value (cumulative loss prediction)
        value = self.fc4_value(x)
        
        return action_probs, value
    
    def select_action(self, state, epsilon=0.1):
        """
        Select action using epsilon-greedy strategy
        Args:
            state: Current state
            epsilon: Exploration rate
        Returns:
            action: Selected task-host assignments
            log_prob: Log probability of the action
            value: Predicted cumulative loss
        """
        action_probs, value = self.forward(state)
        
        # Epsilon-greedy exploration
        if torch.rand(1).item() < epsilon:
            # Random action
            action = torch.randint(0, self.num_hosts, (self.num_tasks,))
        else:
            # Greedy action
            action = torch.argmax(action_probs, dim=2).squeeze()
        
        # Calculate log probability
        log_prob = torch.log(action_probs.squeeze()[torch.arange(self.num_tasks), action] + 1e-10)
        
        return action, log_prob.sum(), value


class DRLAgent:
    """Deep Reinforcement Learning Agent wrapper"""
    
    def __init__(self, model, optimizer, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)
        
    def update(self, loss):
        """Update model parameters"""
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
    def save(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        
    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])