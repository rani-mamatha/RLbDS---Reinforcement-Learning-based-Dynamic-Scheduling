# RLbDS - Reinforcement Learning based Dynamic Scheduling

## 📋 Overview

**RLbDS (Reinforcement Learning based Dynamic Scheduling)** is a deep reinforcement learning system for intelligent task scheduling in edge-cloud computing environments. This implementation is based on research in adaptive resource allocation and dynamic workload management.

### Key Features

- ✅ **Deep Reinforcement Learning** - Enhanced RNN architecture with GRU layers and skip connections
- ✅ **Dynamic Task Scheduling** - Real-time scheduling decisions across edge and cloud hosts
- ✅ **Constraint Satisfaction** - Validates resource constraints and computes penalties
- ✅ **Multi-Objective Optimization** - Balances energy, response time, cost, and SLA compliance
- ✅ **Experience Replay** - Batch training with memory buffer for stable learning
- ✅ **Multiple Hyperparameter Configurations** - Pre-configured profiles for different optimization goals

---

## 🏗️ Architecture

```
RLbDS/
│
├── models/
│   ├── drl_model.py              # Main DRL model with preprocessing
│   ├── rnn_architecture.py       # Enhanced RNN with skip connections
│   └── __init__.py
│
├── scheduling/
│   ├── rlbds_algorithm.py        # Core RLbDS scheduling algorithm
│   ├── constraint_satisfaction.py # Constraint validation module
│   ├── resource_management.py    # Resource allocation and tracking
│   └── __init__.py
│
├── simulation/
│   ├── simulator.py              # Training and evaluation simulator
│   └── __init__.py
│
├── utils/
│   ├── loss_functions.py         # Loss computation and rewards
│   ├── metrics.py                # Performance metrics calculator
│   ├── workload_generator.py     # Task workload generation
│   └── __init__.py
│
├── config/
│   ├── config.yaml               # Main configuration file
│   └── hyperparameters.py        # Hyperparameter configurations
│
├── data/
│   ├── results/                  # Saved models and statistics
│   └── bitbrain_traces/          # Workload trace data (optional)
│
├── main.py                       # Main entry point
├── train.py                      # Training script
├── test.py                       # Testing script
└── requirements.txt              # Python dependencies
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Installation

```bash
# 1. Clone or download the repository
cd "dir"

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Dependencies

```txt
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
pyyaml>=6.0
tqdm>=4.65.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
```

---

## 📖 Usage

### Training the Model

Train the RLbDS model with default balanced configuration:

```bash
python main.py --mode train --hyperparams balanced
```

**Training Options:**

```bash
# Energy-focused optimization
python main.py --mode train --hyperparams energy_focused

# Response time optimization
python main.py --mode train --hyperparams response_time_focused

# Migration cost optimization
python main.py --mode train --hyperparams migration_focused

# Cost optimization
python main.py --mode train --hyperparams cost_focused

# SLA compliance optimization
python main.py --mode train --hyperparams sla_focused
```

**Training Output:**
- Model checkpoint: `data/results/rlbds_model_balanced_<timestamp>.pth`
- Training statistics: `data/results/training_stats_<timestamp>.json`
- Training curves: `data/results/training_curves_<timestamp>.png`

### Testing the Model

Test a trained model:

```bash
python main.py --mode test --hyperparams balanced --model data/results/rlbds_model_balanced_<timestamp>.pth
```

**With custom test intervals:**

```bash
python main.py --mode test --hyperparams balanced --model data/results/rlbds_model_balanced_<timestamp>.pth --test_intervals 500
```

### Comparing Configurations

Compare multiple hyperparameter configurations:

```bash
python main.py --mode compare --hyperparams balanced
```

**⚠️ Note:** This will train all configurations from scratch, which can take significant time.

---

## ⚙️ Configuration

### Main Configuration (`config/config.yaml`)

```yaml
environment:
  num_hosts: 100              # Number of available hosts
  num_edge_hosts: 60          # Number of edge hosts
  num_cloud_hosts: 40         # Number of cloud hosts
  num_tasks: 100              # Tasks per interval
  scheduling_intervals: 1000   # Total training intervals
  interval_duration: 1000      # Duration per interval (ms)

model:
  input_size: 150             # Input feature dimension
  hidden_size: 256            # Hidden layer size
  learning_rate: 0.001        # Learning rate
  batch_size: 32              # Batch size for training
  gamma: 0.99                 # Discount factor

hyperparameters:
  balanced:
    energy_weight: 0.3
    response_time_weight: 0.3
    migration_weight: 0.2
    cost_weight: 0.1
    sla_weight: 0.1
```


---

## 🧠 Model Architecture

### Enhanced RNN Architecture

The RLbDS model uses a sophisticated RNN architecture:

1. **Input Layer**
   - State preprocessing and normalization
   - Feature extraction (host resources, task requirements)

2. **Fully Connected Layers**
   - FC1: Input → 512 dimensions
   - FC2: 512 → 256 dimensions

3. **Recurrent Layers with Skip Connections**
   - R1: GRU with skip connections (256 → hidden_size)
   - R2: GRU with skip connections (hidden_size → hidden_size)
   - R3: GRU with skip connections (hidden_size → hidden_size)

4. **Output Layers**
   - FC3: hidden_size → 128 dimensions
   - **Action Head**: 128 → (num_tasks × num_hosts) - Task-host assignments
   - **Value Head**: 128 → 1 - Cumulative loss prediction

5. **Regularization**
   - Dropout (0.3) after each layer
   - Gradient clipping (max_norm=1.0)

### Training Process

1. **Experience Collection**
   - Generate tasks for each interval
   - Select actions using epsilon-greedy policy
   - Validate constraints and apply penalties
   - Execute tasks and compute metrics
   - Store experiences in replay buffer

2. **Batch Training**
   - Sample random batch from memory
   - Compute loss between predicted and actual values
   - Backpropagate gradients
   - Update model parameters
   - Decay exploration rate (epsilon)

3. **Optimization**
   - Adam optimizer with adaptive learning rate
   - ReduceLROnPlateau scheduler
   - Experience replay for stable learning

---

## 📊 Performance Metrics

### Evaluation Metrics

The system tracks the following metrics:

- **Energy Consumption** (normalized AEC)
  - Total energy used by hosts
  - Normalized for comparison

- **Response Time** (normalized ART)
  - Average task completion time
  - Measured in milliseconds

- **Migration Time** (normalized AMT)
  - Time spent on task migrations
  - Overhead measurement

- **SLA Violation Rate**
  - Percentage of tasks missing deadlines
  - Critical quality metric

- **Total Cost**
  - Cumulative operational cost
  - Edge vs. cloud cost differential

- **Completed Tasks**
  - Number of successfully completed tasks
  - Throughput measurement

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Model File Not Found
```bash
Error: Model file not found: data/results/rlbds_model_balanced.pt
```
**Solution:** Train the model first before testing.
```bash
python main.py --mode train --hyperparams balanced
```

#### 2. Import Errors
```bash
ModuleNotFoundError: No module named 'torch'
```
**Solution:** Install dependencies.
```bash
pip install -r requirements.txt
```

#### 3. CUDA/GPU Issues
```bash
RuntimeError: CUDA out of memory
```
**Solution:** The system defaults to CPU. If using GPU, reduce batch size in `config/config.yaml`.

#### 4. Gradient Issues
```bash
RuntimeError: element 0 of tensors does not require grad
```
**Solution:** This has been fixed in the latest version. Ensure you're using the updated code.

### Performance Tips

1. **Faster Training**
   - Reduce `scheduling_intervals` in config
   - Increase `batch_size` if you have more RAM
   - Use GPU if available (auto-detected)

2. **Better Results**
   - Increase `scheduling_intervals` to 2000+
   - Adjust hyperparameter weights for your use case
   - Fine-tune learning rate and hidden size

3. **Memory Management**
   - Reduce `memory_size` in RLbDS algorithm
   - Lower `num_tasks` and `num_hosts` if needed

---

## 📁 File Descriptions

### Core Files

- **`main.py`** - Main entry point with argument parsing
- **`train.py`** - Training logic and orchestration
- **`test.py`** - Testing and evaluation logic

### Model Files

- **`models/drl_model.py`** - Main DRL model class with preprocessing
- **`models/rnn_architecture.py`** - Enhanced RNN and DRL agent implementation

### Scheduling Files

- **`scheduling/rlbds_algorithm.py`** - Core RLbDS algorithm (Algorithm 1 from paper)
- **`scheduling/constraint_satisfaction.py`** - Constraint validation and penalty computation
- **`scheduling/resource_management.py`** - Host and task resource management

### Utility Files

- **`utils/loss_functions.py`** - Loss and reward computation
- **`utils/metrics.py`** - Performance metrics calculation
- **`utils/workload_generator.py`** - Synthetic workload generation

### Configuration Files

- **`config/config.yaml`** - Main configuration parameters
- **`config/hyperparameters.py`** - Hyperparameter profile definitions

---

## 🎯 Use Cases

### 1. Edge-Cloud Resource Optimization

Optimize resource allocation across edge and cloud infrastructure:

```bash
python main.py --mode train --hyperparams balanced
```

### 2. Energy-Efficient Scheduling

Minimize energy consumption while maintaining performance:

```bash
python main.py --mode train --hyperparams energy_focused
```

### 3. Latency-Critical Applications

Optimize for minimal response time:

```bash
python main.py --mode train --hyperparams response_time_focused
```

### 4. Cost Optimization

Reduce operational costs:

```bash
python main.py --mode train --hyperparams cost_focused
```

### 5. SLA Compliance

Maximize SLA adherence:

```bash
python main.py --mode train --hyperparams sla_focused
```

---

## 📈 Advanced Usage

### Custom Configuration

Create a custom configuration file:

```yaml
# custom_config.yaml
environment:
  num_hosts: 200
  num_tasks: 150
  scheduling_intervals: 2000
  
model:
  hidden_size: 512
  learning_rate: 0.0005
```

Run with custom config:

```bash
python main.py --mode train --config custom_config.yaml
```

### Extending the System

#### Adding New Metrics

Edit `utils/metrics.py`:

```python
def calculate_custom_metric(self, hosts, tasks):
    # Your custom metric logic
    return metric_value
```

#### Adding New Hyperparameter Profiles

Edit `config/hyperparameters.py`:

```python
'my_custom_profile': {
    'energy_weight': 0.2,
    'response_time_weight': 0.4,
    'migration_weight': 0.2,
    'cost_weight': 0.1,
    'sla_weight': 0.1
}
```

#### Custom Workload Patterns

Edit `utils/workload_generator.py` to implement custom task generation patterns.

---

## 🔬 Research & Development

### Algorithm Details

The RLbDS algorithm implements:

1. **State Representation**
   - Host features: CPU, RAM, bandwidth, type (edge/cloud)
   - Task features: Resource requirements, priority, deadline
   - Normalized feature vectors (150-dimensional)

2. **Action Space**
   - Task-to-host assignment matrix
   - Discrete action space: num_tasks × num_hosts

3. **Reward Function**
   - Multi-objective reward combining:
     - Energy efficiency
     - Response time minimization
     - Migration cost reduction
     - Operational cost
     - SLA compliance

4. **Constraint Handling**
   - Resource capacity constraints
   - Bandwidth constraints
   - Host availability constraints
   - SLA deadline constraints

### Key Innovations

- **Skip Connections in RNN** - Better gradient flow
- **Dual-Head Architecture** - Separate action and value prediction
- **Dynamic Constraint Validation** - Real-time feasibility checking
- **Adaptive Exploration** - Epsilon decay for exploration-exploitation balance

---

## 📊 Results Visualization

After training, view the generated visualizations:

```bash
# Windows
start data\results\training_curves_<timestamp>.png

# Linux/Mac
xdg-open data/results/training_curves_<timestamp>.png
```

**Visualizations include:**
- Loss over time
- Reward progression
- Energy consumption trends
- Response time evolution
- SLA violation rates

---

## 🤝 Contributing

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests (if available)
pytest tests/

# Format code
black .

# Lint code
flake8 .
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Document all functions and classes
- Add docstrings for public methods

---

## 📝 Citation

If you use this implementation in your research, please cite:

```bibtex
@article{rlbds2024,
  title={Reinforcement Learning based Dynamic Scheduling for Edge-Cloud Computing},
  author={Mamatha Rani},
  year={2024},
  journal={Implementation Study}
}
```

---

## 📄 License

This project is provided for educational and research purposes.



## 🚦 Status

**Project Status:** ✅ Fully Functional

**Latest Test Results:**
- Training: ✅ Success (27,962 tasks completed)


---

## 📞 Contact

**Project:** RLbDS - Reinforcement Learning based Dynamic Scheduling  
**Author:** Mamatha Rani  


---

## 🙏 Acknowledgments

This implementation is based on research in:
- Deep Reinforcement Learning
- Edge-Cloud Computing
- Dynamic Task Scheduling
- Resource Optimization

Special thanks to the PyTorch team for the excellent deep learning framework.

---

**Happy Scheduling! 🚀**