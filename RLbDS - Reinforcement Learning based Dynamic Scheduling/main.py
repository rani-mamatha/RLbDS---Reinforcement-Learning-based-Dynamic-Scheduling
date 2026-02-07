"""
RLbDS - Unified entry point
Supports training, testing, and hyperparameter comparison
"""

import argparse
import yaml
import os
import sys
from simulation.simulator import Simulator


def train_mode(args):
    """Training mode"""
    simulator = Simulator(args.config)
    
    print("\n" + "="*70)
    print("  MODE: TRAINING")
    print("="*70 + "\n")
    
    training_stats, rlbds = simulator.run_training(args.hyperparams)
    
    # Save model
    model_path = os.path.join(
        simulator.results_dir, 
        f'rlbds_model_{args.hyperparams}_{simulator.timestamp}.pth'
    )
    rlbds.drl_model.save_model(model_path)
    print(f" Model saved to: {model_path}\n")
    
    return rlbds, simulator


def test_mode(args):
    """Testing mode"""
    from models.drl_model import DRLModel
    from scheduling.rlbds_algorithm import RLbDSAlgorithm
    from scheduling.constraint_satisfaction import ConstraintSatisfaction
    from scheduling.resource_management import ResourceManagement
    from utils.workload_generator import WorkloadGenerator
    from utils.metrics import MetricsCalculator
    from utils.loss_functions import LossFunction
    from config.hyperparameters import HyperparameterConfig
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "="*70)
    print("  MODE: TESTING")
    print("="*70 + "\n")
    
    # Load model
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    drl_model = DRLModel(config, device='cpu')
    drl_model.load_model(args.model)
    print(f" Model loaded from: {args.model}\n")
    
    # Setup components
    hyperparams = HyperparameterConfig.get_config(args.hyperparams)
    hyperparams['penalty_weight'] = config['hyperparameters']['penalty_weight']
    
    constraint_module = ConstraintSatisfaction(config)
    resource_module = ResourceManagement(config)
    workload_generator = WorkloadGenerator(config)
    metrics_calculator = MetricsCalculator(config)
    loss_function = LossFunction(config, hyperparams)
    
    rlbds = RLbDSAlgorithm(
        config, drl_model, constraint_module, resource_module,
        metrics_calculator, loss_function
    )
    
    # Evaluate
    eval_metrics = rlbds.evaluate(workload_generator, args.test_intervals)
    
    print("\n Testing completed!\n")
    return eval_metrics


def compare_mode(args):
    """Hyperparameter comparison mode"""
    simulator = Simulator(args.config)
    
    print("\n" + "="*70)
    print("  MODE: HYPERPARAMETER COMPARISON")
    print("="*70 + "\n")
    
    results = simulator.compare_hyperparameters()
    
    print("\n Comparison completed!\n")
    return results


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='RLbDS - Deep Reinforcement Learning for Task Scheduling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Training:
    python main.py --mode train --hyperparams balanced
    
  Testing:
    python main.py --mode test --model results/model.pth --test_intervals 100
    
  Hyperparameter Comparison:
    python main.py --mode compare
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'test', 'compare'],
                        help='Operation mode: train, test, or compare')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--hyperparams', type=str, default='balanced',
                        choices=['energy_focused', 'response_time_focused', 
                                'migration_focused', 'cost_focused', 
                                'sla_focused', 'balanced'],
                        help='Hyperparameter configuration')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint (for test mode)')
    parser.add_argument('--test_intervals', type=int, default=100,
                        help='Number of intervals for testing')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'test' and args.model is None:
        parser.error("--model is required for test mode")
    
    print("\n" + "="*70)
    print("  RLbDS - DEEP REINFORCEMENT LEARNING FOR TASK SCHEDULING")

    print("="*70)
    
    try:
        if args.mode == 'train':
            train_mode(args)
        elif args.mode == 'test':
            test_mode(args)
        elif args.mode == 'compare':
            compare_mode(args)
        
        print("\n" + "="*70)
        print("  EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()