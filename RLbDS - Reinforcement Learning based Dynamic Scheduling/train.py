"""
Training script for RLbDS algorithm
"""

import argparse
import yaml
import os
import sys
from simulation.simulator import Simulator


def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description='Train RLbDS Algorithm')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--hyperparams', type=str, default='balanced',
                        choices=['energy_focused', 'response_time_focused', 
                                'migration_focused', 'cost_focused', 
                                'sla_focused', 'balanced'],
                        help='Hyperparameter configuration')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override configuration with command line arguments
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs
    
    if args.output_dir is not None:
        config['results']['save_path'] = args.output_dir
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Create simulator
    simulator = Simulator(args.config)
    
    print("\n" + "="*70)
    print("  RLbDS TRAINING - Deep Reinforcement Learning for Task Scheduling")
    print("="*70)
    print(f"Configuration: {args.config}")
    print(f"Hyperparameters: {args.hyperparams}")
    print(f"Training Epochs: {config['training']['num_epochs']}")
    print(f"Output Directory: {config['results']['save_path']}")
    print("="*70 + "\n")
    
    # Run training
    try:
        training_stats, rlbds = simulator.run_training(args.hyperparams)
        
        print("\n" + "="*70)
        print("  TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Total Intervals Trained: {len(training_stats['rewards'])}")
        print(f"Final Loss: {training_stats['losses'][-1] if training_stats['losses'] else 'N/A':.4f}")
        print(f"Average Reward: {sum(training_stats['rewards'])/len(training_stats['rewards']) if training_stats['rewards'] else 0:.4f}")
        print("="*70 + "\n")
        
        # Ask if user wants to run evaluation
        response = input("Do you want to run evaluation? (y/n): ")
        if response.lower() == 'y':
            print("\nRunning evaluation...")
            from utils.workload_generator import WorkloadGenerator
            workload_generator = WorkloadGenerator(config)
            eval_metrics = simulator.run_evaluation(rlbds, workload_generator)
            
            print("\n" + "="*70)
            print("  EVALUATION COMPLETED!")
            print("="*70)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()