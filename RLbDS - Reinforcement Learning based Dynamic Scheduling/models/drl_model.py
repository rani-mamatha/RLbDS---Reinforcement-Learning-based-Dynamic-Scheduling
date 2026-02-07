"""
DRL Model implementation with preprocessing and training logic
"""


import torch
import torch.nn.functional as F
import numpy as np
from models.rnn_architecture import EnhancedRNN, DRLAgent


class DRLModel:
    """Main DRL model for task scheduling"""
    
    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device
        
        # Initialize the neural network
        self.model = EnhancedRNN(
            input_size=config['model']['input_size'],
            hidden_size=config['model']['hidden_size'],
            num_tasks=config['environment']['num_tasks'],
            num_hosts=config['environment']['num_hosts']
        )
        
        # Optimizer with adaptive learning rate
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['model']['learning_rate']
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=5
        )
        
        # Create agent
        self.agent = DRLAgent(self.model, self.optimizer, device)
        
        # Statistics for normalization
        self.feature_stats = {
            'cpu_max': 32.0,
            'ram_max': 128.0,
            'bandwidth_max': 1000.0,
            'priority_max': 5.0,
            'deadline_max': 10000.0
        }
        
    def preprocess_state(self, state):
        """
        Preprocess state as described in the paper
        Args:
            state: Dictionary containing hosts, new_tasks, remaining_tasks
        Returns:
            preprocessed_tensor: Normalized tensor ready for the network
        """
        # Extract features
        host_features = state['hosts']
        new_tasks = state['new_tasks']
        remaining_tasks = state['remaining_tasks']
        
        # Create fixed-size feature vector
        feature_size = self.config['model']['input_size']
        features = np.zeros(feature_size)
        
        idx = 0
        
        # Add host features (limit to first 10 hosts)
        for i, host in enumerate(host_features[:10]):
            if idx + 5 > feature_size:
                break
            features[idx] = host['cpu_available'] / host['cpu_total'] if host['cpu_total'] > 0 else 0
            features[idx + 1] = host['ram_available'] / host['ram_total'] if host['ram_total'] > 0 else 0
            features[idx + 2] = host['bandwidth'] / self.feature_stats['bandwidth_max']
            features[idx + 3] = 1.0 if host['is_edge'] else 0.0
            features[idx + 4] = len(host.get('running_tasks', [])) / 10.0
            idx += 5
        
        # Add task features (limit to first 20 tasks)
        all_tasks = new_tasks + remaining_tasks
        for i, task in enumerate(all_tasks[:20]):
            if idx + 5 > feature_size:
                break
            features[idx] = task['cpu_required'] / self.feature_stats['cpu_max']
            features[idx + 1] = task['ram_required'] / self.feature_stats['ram_max']
            features[idx + 2] = task['bandwidth_required'] / self.feature_stats['bandwidth_max']
            features[idx + 3] = task['priority'] / self.feature_stats['priority_max']
            features[idx + 4] = min(task['deadline'] / self.feature_stats['deadline_max'], 1.0)
            idx += 5
        
        # Convert to tensor with proper shape for RNN (batch_size, seq_len, input_size)
        state_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(1).to(self.device)
        
        return state_tensor
    
    def get_action(self, state, epsilon=0.1, training=False):
        """
        Get action from the model
        Args:
            state: Current environment state
            epsilon: Exploration rate
            training: If True, keep gradients for training
        Returns:
            action: Task-host assignments
            log_prob: Log probability of the action
            value: Predicted cumulative loss
        """
        preprocessed_state = self.preprocess_state(state)
        
        if training:
            # Keep gradients for training
            action, log_prob, value = self.model.select_action(preprocessed_state, epsilon)
            return action, log_prob, value
        else:
            # No gradients for inference
            with torch.no_grad():
                action, log_prob, value = self.model.select_action(preprocessed_state, epsilon)
            return action.cpu().numpy(), log_prob.item(), value.item()
    
    def compute_loss(self, state, actual_loss):
        """
        Compute training loss
        Args:
            state: Current environment state
            actual_loss: Actual loss from the environment
        Returns:
            loss_tensor: Loss tensor with gradients
        """
        preprocessed_state = self.preprocess_state(state)
        
        # Forward pass to get predictions
        _, predicted_value = self.model(preprocessed_state)
        
        # Compute MSE loss between predicted and actual cumulative loss
        actual_loss_tensor = torch.FloatTensor([actual_loss]).to(self.device)
        loss = F.mse_loss(predicted_value, actual_loss_tensor.unsqueeze(0))
        
        return loss
    
    def update_model(self, loss):
        """Update model parameters"""
        self.agent.update(loss)
        self.scheduler.step(loss.item())
    
    def save_model(self, path):
        """Save model checkpoint"""
        self.agent.save(path)
        
    def load_model(self, path):
        """Load model checkpoint"""
        self.agent.load(path)