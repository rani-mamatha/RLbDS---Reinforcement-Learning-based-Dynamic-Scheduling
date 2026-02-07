"""
Testing/Evaluation script for trained RLbDS model
"""

import argparse
import yaml
import torch
import os
import sys
import json
import numpy as np
from simulation.simulator import Simulator
from models.drl_model import DRLModel
from scheduling.rlbds_algorithm import RLbDSAlgorithm
from scheduling.constraint_satisfaction import ConstraintSatisfaction
from scheduling.resource_management import ResourceManagement
from utils.workload_generator import WorkloadGenerator
from utils.metrics import MetricsCalculator
from utils.loss_functions import LossFunction
from config.hyperparameters import HyperparameterConfig


def load_trained_model(model_path, config):
    """Load a trained model"""
    drl_model = DRLModel(config, device='cpu')
    drl_model.load_model(model_path)
    return drl_model


def main():
    """Main testing function"""
    
    parser = argparse.ArgumentParser(description='Test/Evaluate RLbDS Algorithm')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--hyperparams', type=str, default='balanced',
                        choices=['energy_focused', 'response_time_focused', 
                                'migration_focused', 'cost_focused', 
                                'sla_focused', 'balanced'],
                        help='Hyperparameter configuration')
    parser.add_argument('--intervals', type=int, default=100,
                        help='Number of intervals to evaluate')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.output_dir is not None:
        config['results']['save_path'] = args.output_dir
        os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("  RLbDS EVALUATION - Testing Trained Model")
    print("="*70)
    print(f"Configuration: {args.config}")
    print(f"Model: {args.model}")
    print(f"Hyperparameters: {args.hyperparams}")
    print(f"Evaluation Intervals: {args.intervals}")
    print("="*70 + "\n")
    
    try:
        # Get hyperparameters
        hyperparams = HyperparameterConfig.get_config(args.hyperparams)
        hyperparams['penalty_weight'] = config['hyperparameters']['penalty_weight']
        
        # Load trained model
        print("Loading trained model...")
        drl_model = load_trained_model(args.model, config)
        print(" Model loaded successfully!\n")
        
        # Initialize components
        print("Initializing components...")
        constraint_module = ConstraintSatisfaction(config)
        resource_module = ResourceManagement(config)
        workload_generator = WorkloadGenerator(config)
        metrics_calculator = MetricsCalculator(config)
        loss_function = LossFunction(config, hyperparams)
        
        # Initialize RLbDS algorithm with loaded model
        rlbds = RLbDSAlgorithm(
            config,
            drl_model,
            constraint_module,
            resource_module,
            metrics_calculator,
            loss_function
        )
        print(" Components initialized successfully!\n")
        
        # Run evaluation
        print("Starting evaluation...\n")
        evaluation_metrics = rlbds.evaluate(workload_generator, args.intervals)
        
        # Save detailed results
        results_path = os.path.join(
            config['results']['save_path'], 
            'detailed_evaluation_results.json'
        )
        with open(results_path, 'w') as f:
            json.dump(evaluation_metrics, f, indent=4)
        
        print(f" Detailed results saved to: {results_path}\n")
        
        # Print summary
        print("\n" + "="*70)
        print("  EVALUATION SUMMARY")
        print("="*70)
        print(f"Total Completed Tasks: {evaluation_metrics['total_completed_tasks']}")
        print(f"Average Energy (normalized): {evaluation_metrics['avg_normalized_aec']:.4f}")
        print(f"Average Response Time: {evaluation_metrics['avg_response_time']:.2f} ms")
        print(f"Average SLA Violation Rate: {evaluation_metrics['avg_sla_violation_rate']:.4f}")
        print(f"Total Cost: ${evaluation_metrics['total_cost']:.2f}")
        print(f"Total Energy: {evaluation_metrics['total_energy']:.2f} Wh")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n\nError during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()