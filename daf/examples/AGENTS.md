# DAF Examples: Agent & Policy Configuration Examples

Example configurations demonstrating agent and policy integration with DAF. Each example shows how to configure different policy types and workflows.

## Policy Types in Examples

Examples demonstrate three policy categories:

### 1. Built-in Policies (Shorthand)

Pre-implemented policies available via shorthand names:

```yaml
policy_class_path: "lstm"         # LSTM-based policy
policy_class_path: "stateless"    # Feedforward network
policy_class_path: "baseline"     # Scripted baseline
policy_class_path: "random"       # Random baseline
```

### 2. Custom Policies (Full Class Path)

User-defined policies implementing CoGames interfaces:

```yaml
policy_class_path: "mypackage.policies.MyCustomPolicy"
policy_class_path: "mypackage.MyTrainablePolicy"
```

### 3. Policies with Checkpoints

Pre-trained policies with saved weights:

```yaml
policy_class_path: "lstm"
policy_data_path: "checkpoints/pretrained_lstm.pt"
```

## Policy Interfaces

Examples use policies implementing these interfaces:

### MultiAgentPolicy (Training)

For batch processing during training and sweeps:

```python
from mettagrid.policy.policy import MultiAgentPolicy
import numpy as np

class MyPolicy(MultiAgentPolicy):
    def __init__(self, env_interface, seed=42):
        super().__init__(env_interface)
        self.network = MyNetwork()
    
    def step_batch(self, observations: np.ndarray, actions: np.ndarray) -> None:
        """Process batch of observations."""
        for agent_id in range(observations.shape[0]):
            actions[agent_id] = self.compute_action(observations[agent_id])
```

### AgentPolicy (Evaluation)

For single-agent evaluation and play:

```python
from mettagrid.policy.policy import AgentPolicy
from mettagrid.types import Action, AgentObservation

class MySingleAgentPolicy(AgentPolicy):
    def step(self, obs: AgentObservation) -> Action:
        """Select action for single agent."""
        return Action(name=self.get_action(obs))
```

## Example Configurations

### sweep_config.yaml

Hyperparameter sweep with grid search over LSTM parameters.

**Demonstrates:**
- Built-in policy usage
- Grid search strategy
- Multiple missions
- Parameter space definition

**Usage:**
```bash
python -c "
from daf import config, sweeps
cfg = config.DAFSweepConfig.from_yaml('sweep_config.yaml')
results = sweeps.daf_launch_sweep(cfg)
"
```

**Key Elements:**
- `policy_class_path: "lstm"` - Built-in LSTM policy
- `strategy: "grid"` - Grid search (exhaustive)
- `search_space` - Learning rate and hidden size ranges
- `episodes_per_trial: 3` - Evaluation episodes per trial

### comparison_config.yaml

Policy comparison across multiple missions.

**Demonstrates:**
- Multi-policy comparison
- Multiple missions
- Statistical analysis setup
- Built-in policy specifications

**Usage:**
```bash
python -c "
from daf import config, comparison
# Note: comparison uses code, not YAML directly
# See pipeline_config.yaml for orchestrated comparison
"
```

**Key Elements:**
- Multiple policy definitions
- Multiple mission targets
- Episodes per mission configuration

### pipeline_config.yaml

End-to-end pipeline from training through comparison.

**Demonstrates:**
- Multi-stage workflow
- Training configuration
- Sweep configuration
- Comparison setup
- Stop on failure handling

**Usage:**
```bash
python -c "
from daf import config, orchestrators
pipeline_cfg = config.DAFPipelineConfig.from_yaml('pipeline_config.yaml')
result = orchestrators.daf_run_benchmark_pipeline(
    config=pipeline_cfg
)
"
```

**Key Elements:**
- `stages` - Sequential execution order
- `training_config` - Training parameters
- `sweep_config` - Hyperparameter sweep
- `comparison_config` - Policy comparison
- `stop_on_failure` - Error handling strategy

### output_management_example.py

Python example showing output management and logging.

**Demonstrates:**
- Output directory organization
- Structured logging
- Metrics tracking
- Session management

**Usage:**
```bash
python daf/examples/output_management_example.py
```

**Key Concepts:**
- `OutputManager` for output coordination
- `DAFLogger` for structured logging
- Session metadata tracking

## Creating Custom Policy Examples

### Example 1: Simple Feedforward Policy

```python
# mypackage/policies.py
from mettagrid.policy.policy import MultiAgentPolicy
import numpy as np
import torch
import torch.nn as nn

class FeedforwardPolicy(MultiAgentPolicy):
    def __init__(self, env_interface, hidden_size=64, seed=42):
        super().__init__(env_interface)
        self.seed = seed
        self.hidden_size = hidden_size
        
        # Network
        obs_size = 100  # Example observation size
        action_size = 5  # Example action count
        
        self.network = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )
    
    def step_batch(self, observations, actions):
        """Process batch of observations."""
        num_agents = observations.shape[0]
        
        for agent_id in range(num_agents):
            obs = observations[agent_id].flatten()
            obs_tensor = torch.from_numpy(obs).float()
            
            with torch.no_grad():
                logits = self.network(obs_tensor)
                action = logits.argmax().item()
            
            actions[agent_id] = action

# sweep_custom.yaml
name: "feedforward_policy_sweep"
policy_class_path: "mypackage.policies.FeedforwardPolicy"
missions: ["training_facility_1"]
strategy: "grid"
search_space:
  hidden_size: [32, 64, 128]
episodes_per_trial: 5
```

### Example 2: LSTM-based Policy

```python
# mypackage/policies.py
from mettagrid.policy.policy import MultiAgentPolicy
import torch
import torch.nn as nn

class LSTMPolicy(MultiAgentPolicy):
    def __init__(self, env_interface, hidden_size=128, seed=42):
        super().__init__(env_interface)
        self.hidden_size = hidden_size
        
        obs_size = 100
        action_size = 5
        
        self.lstm = nn.LSTM(
            input_size=obs_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, action_size)
        
        self.h = None
        self.c = None
    
    def step_batch(self, observations, actions):
        """Process batch with LSTM state tracking."""
        num_agents = observations.shape[0]
        
        for agent_id in range(num_agents):
            obs = observations[agent_id]
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            
            with torch.no_grad():
                if self.h is None:
                    lstm_out, (self.h, self.c) = self.lstm(obs_tensor)
                else:
                    lstm_out, (self.h, self.c) = self.lstm(
                        obs_tensor,
                        (self.h, self.c),
                    )
                
                logits = self.head(lstm_out[:, -1, :])
                action = logits.argmax(dim=1).item()
            
            actions[agent_id] = action

# sweep_lstm.yaml
name: "lstm_policy_optimization"
policy_class_path: "mypackage.policies.LSTMPolicy"
missions: ["training_facility_1", "training_facility_2"]
strategy: "grid"
search_space:
  hidden_size: [64, 128, 256]
  learning_rate: [0.0001, 0.001]
episodes_per_trial: 5
objective_metric: "avg_reward"
optimize_direction: "maximize"
```

### Example 3: Policy with Checkpoint Loading

```python
# mypackage/policies.py
from mettagrid.policy.policy import MultiAgentPolicy
import torch

class CheckpointPolicy(MultiAgentPolicy):
    def __init__(self, env_interface, checkpoint_path=None, seed=42):
        super().__init__(env_interface)
        self.network = self.build_network()
        
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
    
    def build_network(self):
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
        )
    
    def load_checkpoint(self, path):
        """Load weights from checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        self.network.load_state_dict(checkpoint['model_state_dict'])
    
    def step_batch(self, observations, actions):
        """Process batch with checkpoint-loaded weights."""
        for agent_id in range(observations.shape[0]):
            obs = observations[agent_id].flatten()
            obs_tensor = torch.from_numpy(obs).float()
            
            with torch.no_grad():
                logits = self.network(obs_tensor)
                action = logits.argmax().item()
            
            actions[agent_id] = action

# sweep_pretrained.yaml
name: "pretrained_policy_sweep"
policy_class_path: "mypackage.policies.CheckpointPolicy"
policy_data_path: "checkpoints/base_model.pt"
missions: ["assembler_2"]
strategy: "random"
search_space:
  learning_rate: [0.00001, 0.0001, 0.001]
num_trials: 9
episodes_per_trial: 5
```

## Using Examples

### 1. Run Hyperparameter Sweep

```bash
cd /Users/4d/Documents/GitHub/cogames

python -c "
from daf.config import DAFSweepConfig
from daf import sweeps

# Load example config
cfg = DAFSweepConfig.from_yaml('daf/examples/sweep_config.yaml')

# Run sweep
results = sweeps.daf_launch_sweep(cfg)

# Analyze results
best = sweeps.daf_sweep_best_config(results)
print(f'Best learning rate: {best.hyperparameters[\"learning_rate\"]}')
print(f'Best hidden size: {best.hyperparameters[\"hidden_size\"]}')
"
```

### 2. Run Comparison

```bash
python -c "
from daf import comparison
from mettagrid.policy.policy import PolicySpec

policies = [
    PolicySpec(class_path='lstm'),
    PolicySpec(class_path='baseline'),
    PolicySpec(class_path='random'),
]

results = comparison.daf_compare_policies(
    policies=policies,
    missions=['training_facility_1', 'assembler_2'],
    episodes_per_mission=10,
)

print(results.summary_statistics)
"
```

### 3. Run Pipeline

```bash
python -c "
from daf.config import DAFSweepConfig
from daf.orchestrators import daf_run_sweep_pipeline

cfg = DAFSweepConfig.from_yaml('daf/examples/sweep_config.yaml')
result = daf_run_sweep_pipeline(config=cfg)

if result.is_success():
    print('Pipeline succeeded!')
    best = result.outputs['best_config']
    print(f'Best config: {best.hyperparameters}')
"
```

## Customizing Examples

### Modify for Different Missions

```yaml
# Change missions
missions:
  - "training_facility_2"
  - "assembler_1"
  - "assembler_2"
```

### Adjust Search Space

```yaml
# Grid search: larger space for broader exploration
search_space:
  learning_rate: [0.00001, 0.0001, 0.001, 0.01]
  hidden_size: [32, 64, 128, 256]

# Or random search: smaller list of ranges
strategy: "random"
search_space:
  learning_rate: [0.00001, 0.1]  # (min, max)
  hidden_size: [32, 512]
num_trials: 20  # Number of random samples
```

### Use Custom Policies

```yaml
policy_class_path: "mypackage.MyCustomPolicy"
```

Ensure your policy module is in `PYTHONPATH`:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Best Practices

1. **Start with built-in policies**
   ```yaml
   policy_class_path: "lstm"  # Easy to configure
   ```

2. **Use small episodes for testing**
   ```yaml
   episodes_per_trial: 1  # Fast iteration
   ```

3. **Validate with single trial first**
   ```yaml
   num_trials: 1  # Test setup
   ```

4. **Store checkpoints in daf_output**
   ```yaml
   policy_data_path: "daf_output/checkpoints/model.pt"
   ```

5. **Use multiple missions for robustness**
   ```yaml
   missions:
     - "training_facility_1"
     - "training_facility_2"
     - "assembler_2"
   ```

## See Also

- `README.md` - Example overview and usage
- `sweep_config.yaml` - Hyperparameter sweep example
- `comparison_config.yaml` - Policy comparison setup
- `pipeline_config.yaml` - Multi-stage workflow
- `output_management_example.py` - Output management
- `../AGENTS.md` - Policy interfaces and support
- `../src/AGENTS.md` - DAF modules
- `../../AGENTS.md` - Top-level agent architecture







