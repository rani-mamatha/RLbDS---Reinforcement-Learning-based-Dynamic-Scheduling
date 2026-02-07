"""
Loss Function implementation as described in the paper
Equations 10 and 11
"""

import torch
import torch.nn as nn


class LossFunction:
    """
    Compute loss function for DRL model
    Combines multiple objectives: Energy, Response Time, Migration, Cost, SLA
    """
    
    def __init__(self, config, hyperparams):
        self.config = config
        
        # Hyperparameters (alpha, beta, gamma, delta, epsilon)
        self.alpha = hyperparams['alpha']
        self.beta = hyperparams['beta']
        self.gamma = hyperparams['gamma']
        self.delta = hyperparams['delta']
        self.epsilon = hyperparams['epsilon']
        
        # Penalty weight
        self.penalty_weight = hyperparams.get('penalty_weight', 100.0)
        
        # Validate hyperparameters sum to 1
        total = self.alpha + self.beta + self.gamma + self.delta + self.epsilon
        assert abs(total - 1.0) < 1e-6, f"Hyperparameters must sum to 1.0, got {total}"
    
    def compute_loss(self, metrics, penalty=0.0):
        """
        Compute loss as per Equation 10 in the paper
        
        Args:
            metrics: Dictionary containing normalized metrics
                - normalized_aec: Average Energy Consumption
                - normalized_art: Average Response Time
                - normalized_amt: Average Migration Time
                - normalized_cost: Cost (needs normalization)
                - sla_violation_rate: SLA violations
            penalty: Constraint violation penalty
        
        Returns:
            loss: Combined loss value
        """
        # Extract normalized metrics
        aec = metrics.get('normalized_aec', 0.0)
        art = metrics.get('normalized_art', 0.0)
        amt = metrics.get('normalized_amt', 0.0)
        cost = metrics.get('normalized_cost', 0.0)
        sla = metrics.get('sla_violation_rate', 0.0)
        
        # Compute weighted loss (Equation 10)
        loss = (
            self.alpha * aec +
            self.beta * art +
            self.gamma * amt +
            self.delta * cost +
            self.epsilon * sla
        )
        
        return loss
    
    def compute_loss_with_penalty(self, metrics, penalty=0.0):
        """
        Compute loss with penalty term as per Equation 11
        
        Args:
            metrics: Dictionary containing normalized metrics
            penalty: Constraint violation penalty (0 if satisfied, >0 otherwise)
        
        Returns:
            total_loss: Loss with penalty term
        """
        # Base loss
        base_loss = self.compute_loss(metrics)
        
        # Total loss with penalty (Equation 11)
        total_loss = base_loss + self.penalty_weight * penalty
        
        return total_loss
    
    def compute_normalized_cost(self, total_cost, max_cost):
        """Normalize cost to [0, 1] range"""
        if max_cost > 0:
            return total_cost / max_cost
        return 0.0
    
    def compute_reward(self, metrics, penalty=0.0):
        """
        Compute reward (negative loss) for reinforcement learning
        
        Args:
            metrics: Dictionary containing normalized metrics
            penalty: Constraint violation penalty
        
        Returns:
            reward: Negative of the loss (to maximize)
        """
        loss = self.compute_loss_with_penalty(metrics, penalty)
        reward = -loss  # Negative because we want to minimize loss
        
        return reward
    
    def to_tensor(self, value):
        """Convert value to tensor if needed"""
        if not isinstance(value, torch.Tensor):
            return torch.tensor(value, dtype=torch.float32)
        return value