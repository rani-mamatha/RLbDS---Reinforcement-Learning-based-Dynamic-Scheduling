"""
RLbDS Algorithm Implementation
Reinforcement Learning based Dynamic Scheduling
Algorithm 1 from the paper
"""

import torch
import numpy as np
from tqdm import tqdm
import copy


class RLbDSAlgorithm:
    """
    Reinforcement Learning based Dynamic Scheduling Algorithm
    Main algorithm implementing the paper's Algorithm 1
    """
    
    def __init__(self, config, drl_model, constraint_module, 
                 resource_module, metrics_calculator, loss_function):
        self.config = config
        self.drl_model = drl_model
        self.constraint_module = constraint_module
        self.resource_module = resource_module
        self.metrics_calculator = metrics_calculator
        self.loss_function = loss_function
        
        # Algorithm parameters
        self.batch_size = config['model']['batch_size']
        self.max_intervals = config['environment']['scheduling_intervals']
        self.interval_duration = config['environment']['interval_duration']
        
        # Exploration parameters
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.epsilon = self.epsilon_start
        
        # Memory for experience replay
        self.memory = []
        self.memory_size = 10000
        
        # Statistics
        self.training_stats = {
            'losses': [],
            'rewards': [],
            'metrics_history': []
        }
    
    def schedule_interval(self, interval_num, new_tasks, remaining_tasks):
        """
        Schedule tasks for a single interval
        
        Args:
            interval_num: Current interval number
            new_tasks: Newly arrived tasks
            remaining_tasks: Tasks from previous interval
        
        Returns:
            metrics: Performance metrics for this interval
            active_tasks: Tasks still running
            completed_tasks: Tasks completed in this interval
            loss_value: Loss value for this interval
            reward: Reward for this interval
        """
        # Combine tasks
        all_tasks = new_tasks + remaining_tasks
        
        if not all_tasks:
            return {}, [], [], 0.0, 0.0
        
        # Get current state
        state = self.resource_module.get_state(new_tasks, remaining_tasks)
        
        # Get action from DRL model
        action, log_prob, predicted_value = self.drl_model.get_action(
            state, epsilon=self.epsilon, training=False
        )
        
        # Validate action with constraint satisfaction
        is_valid, penalty, violations = self.constraint_module.validate_action(
            action, self.resource_module.hosts, all_tasks
        )
        
        # If invalid, try to get alternative assignments
        if not is_valid:
            action = self._get_corrected_action(action, all_tasks, violations)
            _, penalty, _ = self.constraint_module.validate_action(
                action, self.resource_module.hosts, all_tasks
            )
        
        # Allocate tasks
        allocated_tasks, failed_tasks = self.resource_module.allocate_tasks(
            action, all_tasks
        )
        
        # Execute tasks for this interval
        completed_tasks, active_tasks = self.resource_module.execute_tasks(
            allocated_tasks, self.interval_duration
        )
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_all_metrics(
            [h.to_dict() for h in self.resource_module.hosts],
            completed_tasks + active_tasks,
            self.interval_duration
        )
        
        # Compute loss
        loss_value = self.loss_function.compute_loss_with_penalty(metrics, penalty)
        
        # Compute reward
        reward = self.loss_function.compute_reward(metrics, penalty)
        
        # Store experience
        self._store_experience(state, action, reward, metrics, loss_value)
        
        # Update statistics
        self.training_stats['rewards'].append(reward)
        self.training_stats['metrics_history'].append(metrics)
        
        return metrics, active_tasks, completed_tasks, loss_value, reward
    
    def _get_corrected_action(self, action, tasks, violations):
        """
        Correct invalid action by suggesting alternatives
        
        Args:
            action: Original action
            tasks: List of tasks
            violations: List of constraint violations
        
        Returns:
            corrected_action: Corrected action
        """
        corrected_action = action.copy()
        
        for task_idx, host_idx in enumerate(action):
            if task_idx >= len(tasks):
                break
            
            task = tasks[task_idx]
            
            # Check if this assignment has violations
            task_violations = [v for v in violations if f"Task {task.task_id}" in v]
            
            if task_violations:
                # Suggest alternative host
                alternative_host = self.constraint_module.suggest_alternative(
                    task, self.resource_module.hosts
                )
                
                if alternative_host is not None:
                    corrected_action[task_idx] = alternative_host
        
        return corrected_action
    
    def _store_experience(self, state, action, reward, metrics, loss):
        """Store experience in memory for replay"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'metrics': metrics,
            'loss': loss
        }
        
        self.memory.append(experience)
        
        # Keep memory size limited
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
    
    def train_batch(self):
        """
        Train the DRL model on a batch of experiences
        
        Returns:
            avg_loss: Average loss for the batch
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from memory
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        total_loss = 0.0
        
        for experience in batch:
            # Compute loss with gradients using the stored state
            state = experience['state']
            actual_loss = experience['loss']
            
            # Use the DRL model's compute_loss method to get a loss tensor with gradients
            loss_tensor = self.drl_model.compute_loss(state, actual_loss)
            
            total_loss += loss_tensor.item()
            
            # Update model
            self.drl_model.update_model(loss_tensor)
        
        avg_loss = total_loss / self.batch_size
        self.training_stats['losses'].append(avg_loss)
        
        return avg_loss
    
    def run_training(self, workload_generator, num_intervals=None):
        """
        Run complete training process
        Algorithm 1 from the paper
        
        Args:
            workload_generator: Workload generator object
            num_intervals: Number of intervals to train (None = use config)
        
        Returns:
            training_stats: Statistics from training
        """
        if num_intervals is None:
            num_intervals = self.max_intervals
        
        print(f"\n{'='*60}")
        print(f"Starting RLbDS Training for {num_intervals} intervals")
        print(f"{'='*60}\n")
        
        remaining_tasks = []
        all_completed_tasks = []
        
        # Training loop
        for interval in tqdm(range(num_intervals), desc="Training Progress"):
            # Generate new tasks for this interval
            new_tasks = workload_generator.generate_tasks_for_interval(interval)
            
            # Schedule the interval
            metrics, active_tasks, completed_tasks, loss, reward = self.schedule_interval(
                interval, new_tasks, remaining_tasks
            )
            
            # Update remaining tasks for next interval
            remaining_tasks = active_tasks
            all_completed_tasks.extend(completed_tasks)
            
            # Train on batch
            if (interval + 1) % self.batch_size == 0:
                avg_loss = self.train_batch()
                if avg_loss is not None:
                    tqdm.write(f"Interval {interval+1}: Loss={avg_loss:.4f}, Reward={reward:.4f}")
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # Periodic reporting
            if (interval + 1) % 100 == 0:
                self._print_interval_stats(interval, metrics)
        
        print(f"\n{'='*60}")
        print(f"Training Completed!")
        print(f"Total Completed Tasks: {len(all_completed_tasks)}")
        print(f"{'='*60}\n")
        
        return self.training_stats
    
    def _print_interval_stats(self, interval, metrics):
        """Print statistics for current interval"""
        print(f"\n--- Interval {interval+1} Statistics ---")
        print(f"Energy (normalized): {metrics.get('normalized_aec', 0):.4f}")
        print(f"Response Time (normalized): {metrics.get('normalized_art', 0):.4f}")
        print(f"SLA Violations: {metrics.get('sla_violation_rate', 0):.4f}")
        print(f"Cost: ${metrics.get('total_cost', 0):.2f}")
        print(f"Completed Tasks: {metrics.get('num_completed_tasks', 0)}")
        print(f"Epsilon: {self.epsilon:.4f}")
        print("-" * 40)
    
    def evaluate(self, workload_generator, num_intervals):
        """
        Evaluate the trained model
        
        Args:
            workload_generator: Workload generator
            num_intervals: Number of intervals to evaluate
        
        Returns:
            evaluation_metrics: Aggregated metrics
        """
        print(f"\n{'='*60}")
        print(f"Evaluating RLbDS for {num_intervals} intervals")
        print(f"{'='*60}\n")
        
        # Set epsilon to 0 for evaluation (no exploration)
        old_epsilon = self.epsilon
        self.epsilon = 0.0
        
        remaining_tasks = []
        all_metrics = []
        all_completed_tasks = []
        
        for interval in tqdm(range(num_intervals), desc="Evaluation Progress"):
            new_tasks = workload_generator.generate_tasks_for_interval(interval)
            
            metrics, active_tasks, completed_tasks, loss, reward = self.schedule_interval(
                interval, new_tasks, remaining_tasks
            )
            
            remaining_tasks = active_tasks
            all_completed_tasks.extend(completed_tasks)
            all_metrics.append(metrics)
        
        # Restore epsilon
        self.epsilon = old_epsilon
        
        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(all_metrics)
        aggregated_metrics['total_completed_tasks'] = len(all_completed_tasks)
        
        print(f"\n{'='*60}")
        print(f"Evaluation Results:")
        print(f"{'='*60}")
        print(f"Avg Energy (normalized): {aggregated_metrics['avg_normalized_aec']:.4f}")
        print(f"Avg Response Time (ms): {aggregated_metrics['avg_response_time']:.2f}")
        print(f"Avg SLA Violations: {aggregated_metrics['avg_sla_violation_rate']:.4f}")
        print(f"Total Cost: ${aggregated_metrics['total_cost']:.2f}")
        print(f"Total Completed Tasks: {aggregated_metrics['total_completed_tasks']}")
        print(f"{'='*60}\n")
        
        return aggregated_metrics
    
    def _aggregate_metrics(self, metrics_list):
        """Aggregate metrics from multiple intervals"""
        if not metrics_list:
            return {}
        
        aggregated = {
            'avg_normalized_aec': np.mean([m.get('normalized_aec', 0) for m in metrics_list]),
            'avg_normalized_art': np.mean([m.get('normalized_art', 0) for m in metrics_list]),
            'avg_normalized_amt': np.mean([m.get('normalized_amt', 0) for m in metrics_list]),
            'avg_response_time': np.mean([m.get('avg_response_time', 0) for m in metrics_list]),
            'avg_sla_violation_rate': np.mean([m.get('sla_violation_rate', 0) for m in metrics_list]),
            'total_cost': sum([m.get('total_cost', 0) for m in metrics_list]),
            'total_energy': sum([m.get('total_energy', 0) for m in metrics_list]),
            'total_completed_tasks': sum([m.get('num_completed_tasks', 0) for m in metrics_list])
        }
        
        return aggregated