"""
Simulator for edge-cloud environment
Integrates all components for complete simulation
"""

import yaml
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class Simulator:
    """
    Main simulator orchestrating all components
    """
    
    def __init__(self, config_path='config/config.yaml'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.results_dir = self.config['results']['save_path']
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def setup_simulation(self, hyperparams_config='balanced'):
        """
        Setup simulation components
        
        Args:
            hyperparams_config: Hyperparameter configuration name
        
        Returns:
            All initialized components
        """
        from config.hyperparameters import HyperparameterConfig
        from models.drl_model import DRLModel
        from scheduling.rlbds_algorithm import RLbDSAlgorithm
        from scheduling.constraint_satisfaction import ConstraintSatisfaction
        from scheduling.resource_management import ResourceManagement
        from utils.workload_generator import WorkloadGenerator
        from utils.metrics import MetricsCalculator
        from utils.loss_functions import LossFunction
        
        print("Setting up simulation components...")
        
        # Get hyperparameters
        hyperparams = HyperparameterConfig.get_config(hyperparams_config)
        hyperparams['penalty_weight'] = self.config['hyperparameters']['penalty_weight']
        
        # Initialize components
        drl_model = DRLModel(self.config, device='cpu')
        constraint_module = ConstraintSatisfaction(self.config)
        resource_module = ResourceManagement(self.config)
        workload_generator = WorkloadGenerator(self.config)
        metrics_calculator = MetricsCalculator(self.config)
        loss_function = LossFunction(self.config, hyperparams)
        
        # Initialize RLbDS algorithm
        rlbds = RLbDSAlgorithm(
            self.config,
            drl_model,
            constraint_module,
            resource_module,
            metrics_calculator,
            loss_function
        )
        
        print(" All components initialized successfully!\n")
        
        return rlbds, workload_generator, hyperparams
    
    def run_training(self, hyperparams_config='balanced'):
        """
        Run complete training simulation
        
        Args:
            hyperparams_config: Hyperparameter configuration
        
        Returns:
            training_stats: Training statistics
        """
        rlbds, workload_generator, hyperparams = self.setup_simulation(hyperparams_config)
        
        # Run training
        num_train_intervals = self.config['training']['num_epochs'] * 10
        training_stats = rlbds.run_training(workload_generator, num_train_intervals)
        
        # Save model
        model_path = os.path.join(self.results_dir, f'rlbds_model_{self.timestamp}.pth')
        rlbds.drl_model.save_model(model_path)
        print(f" Model saved to: {model_path}\n")
        
        # Save training statistics
        stats_path = os.path.join(self.results_dir, f'training_stats_{self.timestamp}.json')
        self._save_stats(training_stats, stats_path)
        
        # Plot training curves
        if self.config['results']['plot_metrics']:
            self.plot_training_curves(training_stats)
        
        return training_stats, rlbds
    
    def run_evaluation(self, rlbds, workload_generator):
        """
        Run evaluation on trained model
        
        Args:
            rlbds: Trained RLbDS algorithm
            workload_generator: Workload generator
        
        Returns:
            evaluation_metrics: Evaluation results
        """
        num_test_intervals = 100
        evaluation_metrics = rlbds.evaluate(workload_generator, num_test_intervals)
        
        # Save evaluation results
        eval_path = os.path.join(self.results_dir, f'evaluation_{self.timestamp}.json')
        with open(eval_path, 'w') as f:
            json.dump(evaluation_metrics, f, indent=4)
        
        print(f" Evaluation results saved to: {eval_path}\n")
        
        return evaluation_metrics
    
    def _save_stats(self, stats, path):
        """Save statistics to JSON file"""
        # Convert numpy types to Python types
        serializable_stats = {}
        for key, value in stats.items():
            if isinstance(value, list):
                serializable_stats[key] = [float(v) if hasattr(v, 'item') else v for v in value]
            else:
                serializable_stats[key] = value
        
        with open(path, 'w') as f:
            json.dump(serializable_stats, f, indent=4)
        
        print(f" Statistics saved to: {path}\n")
    
    def plot_training_curves(self, training_stats):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('RLbDS Training Performance', fontsize=16)
        
        # Loss curve
        if training_stats['losses']:
            axes[0, 0].plot(training_stats['losses'])
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Batch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
        
        # Reward curve
        if training_stats['rewards']:
            axes[0, 1].plot(training_stats['rewards'])
            axes[0, 1].set_title('Rewards')
            axes[0, 1].set_xlabel('Interval')
            axes[0, 1].set_ylabel('Reward')
            axes[0, 1].grid(True)
        
        # Energy consumption
        if training_stats['metrics_history']:
            energy = [m.get('normalized_aec', 0) for m in training_stats['metrics_history']]
            axes[1, 0].plot(energy)
            axes[1, 0].set_title('Normalized Energy Consumption')
            axes[1, 0].set_xlabel('Interval')
            axes[1, 0].set_ylabel('Energy (normalized)')
            axes[1, 0].grid(True)
        
        # SLA violations
        if training_stats['metrics_history']:
            sla = [m.get('sla_violation_rate', 0) for m in training_stats['metrics_history']]
            axes[1, 1].plot(sla)
            axes[1, 1].set_title('SLA Violation Rate')
            axes[1, 1].set_xlabel('Interval')
            axes[1, 1].set_ylabel('Violation Rate')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.results_dir, f'training_curves_{self.timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f" Training curves saved to: {plot_path}\n")
        
        plt.close()
    
    def compare_hyperparameters(self):
        """Compare different hyperparameter configurations"""
        from config.hyperparameters import HyperparameterConfig
        
        configs = ['energy_focused', 'response_time_focused', 'cost_focused', 
                   'sla_focused', 'balanced']
        
        results = {}
        
        for config_name in configs:
            print(f"\n{'='*60}")
            print(f"Testing configuration: {config_name}")
            print(f"{'='*60}")
            
            stats, rlbds = self.run_training(config_name)
            
            # Quick evaluation
            _, workload_generator, _ = self.setup_simulation(config_name)
            eval_metrics = rlbds.evaluate(workload_generator, 50)
            
            results[config_name] = eval_metrics
        
        # Save comparison
        comparison_path = os.path.join(self.results_dir, f'hyperparameter_comparison_{self.timestamp}.json')
        with open(comparison_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\n Hyperparameter comparison saved to: {comparison_path}\n")
        
        return results