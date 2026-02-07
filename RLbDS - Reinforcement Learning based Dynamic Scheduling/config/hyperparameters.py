"""
Hyperparameter configurations for the DRL model
"""

class HyperparameterConfig:
    """Configuration class for hyperparameters"""
    
    # Different hyperparameter configurations for various use cases
    CONFIGS = {
        'energy_focused': {
            'alpha': 1.0,
            'beta': 0.0,
            'gamma': 0.0,
            'delta': 0.0,
            'epsilon': 0.0
        },
        'response_time_focused': {
            'alpha': 0.0,
            'beta': 1.0,
            'gamma': 0.0,
            'delta': 0.0,
            'epsilon': 0.0
        },
        'migration_focused': {
            'alpha': 0.0,
            'beta': 0.0,
            'gamma': 1.0,
            'delta': 0.0,
            'epsilon': 0.0
        },
        'cost_focused': {
            'alpha': 0.0,
            'beta': 0.0,
            'gamma': 0.0,
            'delta': 1.0,
            'epsilon': 0.0
        },
        'sla_focused': {
            'alpha': 0.0,
            'beta': 0.0,
            'gamma': 0.0,
            'delta': 0.0,
            'epsilon': 1.0
        },
        'balanced': {
            'alpha': 0.2,
            'beta': 0.2,
            'gamma': 0.2,
            'delta': 0.2,
            'epsilon': 0.2
        }
    }
    
    @staticmethod
    def get_config(config_name='balanced'):
        """Get hyperparameter configuration by name"""
        if config_name not in HyperparameterConfig.CONFIGS:
            raise ValueError(f"Unknown config: {config_name}")
        return HyperparameterConfig.CONFIGS[config_name]
    
    @staticmethod
    def validate_hyperparams(alpha, beta, gamma, delta, epsilon):
        """Validate that hyperparameters sum to 1"""
        total = alpha + beta + gamma + delta + epsilon
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"Hyperparameters must sum to 1.0, got {total}")
        return True