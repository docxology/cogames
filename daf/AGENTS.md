# DAF Module: Agent Support & Policy Infrastructure

The DAF (Distributed Agent Framework) module provides distributed training, evaluation, and analysis infrastructure for agent policies in CoGames.

## Module Overview

DAF extends CoGames by providing:

- **Hyperparameter Sweeps**: Grid/random search over policy hyperparameters
- **Policy Comparison**: Statistical analysis of multiple policies
- **Distributed Training**: Multi-machine training orchestration
- **Visualization**: HTML reports and performance dashboards
- **Mission Analysis**: Detailed per-mission performance breakdown
- **Environment Checks**: Pre-flight validation and device optimization

All DAF functions wrap underlying CoGames methods—DAF never duplicates core functionality.

## Agent Policy Support

DAF works with all policies implementing CoGames policy interfaces:

### Supported Policy Types

1. **Built-in Policies** (shorthand)
   - `lstm` - Stateful LSTM-based policy
   - `stateless` - Feedforward neural network
   - `baseline` - Scripted agent
   - `random` - Random baseline

2. **Custom Policies** (full class paths)
   - Must implement `MultiAgentPolicy` or `AgentPolicy`
   - Can be importable Python classes or file paths
   - Support optional checkpoint files

### Example: Using Different Policies

```python
from daf.config import DAFSweepConfig
from mettagrid.policy.policy import PolicySpec

# Built-in policy
sweep_cfg = DAFSweepConfig(
    policy_class_path="lstm",
    missions=["training_facility_1"],
    episodes_per_trial=5,
)

# Custom policy
sweep_cfg = DAFSweepConfig(
    policy_class_path="mypackage.policies.CustomPolicy",
    missions=["training_facility_1"],
)

# Policy with checkpoint
sweep_cfg = DAFSweepConfig(
    policy_class_path="lstm",
    policy_data_path="checkpoints/pretrained.pt",
    missions=["training_facility_1"],
)
```

## Core DAF Modules

### 1. Configuration (`config.py`)

Manages sweep, comparison, and training configurations:

- `DAFSweepConfig` - Hyperparameter sweep configuration
- `DAFComparisonConfig` - Multi-policy comparison setup
- `DAFTrainingConfig` - Distributed training configuration

```python
from daf.config import DAFSweepConfig

cfg = DAFSweepConfig.from_yaml("sweep_config.yaml")
cfg.policy_class_path = "lstm"
cfg.validate()
```

### 2. Sweeps (`sweeps.py`)

Launch and analyze hyperparameter sweeps:

```python
from daf import sweeps, config

cfg = config.DAFSweepConfig.from_yaml("sweep.yaml")
results = sweeps.daf_launch_sweep(cfg)
best = sweeps.daf_sweep_best_config(results)
print(f"Best hyperparameters: {best.hyperparameters}")
```

### 3. Comparison (`comparison.py`)

Compare multiple policies statistically:

```python
from daf import comparison
from mettagrid.policy.policy import PolicySpec

results = comparison.daf_compare_policies(
    policies=[
        PolicySpec(class_path="lstm"),
        PolicySpec(class_path="baseline"),
        PolicySpec(class_path="random"),
    ],
    missions=["training_facility_1", "assembler_2"],
    episodes_per_mission=10
)

print(results.summary_statistics)
print(results.pairwise_significance_tests)
```

### 4. Distributed Training (`distributed_training.py`)

Multi-machine training orchestration:

```python
from daf import distributed_training

job = distributed_training.daf_launch_distributed_training(
    policy_class_path="lstm",
    mission_name="training_facility_1",
    num_machines=4,
    steps_per_machine=250_000,
)

final_checkpoint = job.wait_for_completion()
```

### 5. Visualization (`visualization.py`)

Generate HTML reports and dashboards:

```python
from daf import visualization, comparison

results = comparison.daf_compare_policies(...)
visualization.daf_export_comparison_html(
    results,
    output_path="comparison_report.html"
)
```

### 6. Mission Analysis (`mission_analysis.py`)

Analyze per-mission performance:

```python
from daf import mission_analysis

analysis = mission_analysis.daf_analyze_mission_performance(
    policy_class_path="lstm",
    mission_name="assembler_2",
    episodes=20
)

print(f"Success rate: {analysis.success_rate}")
print(f"Mean reward: {analysis.mean_reward}")
print(f"Failure modes: {analysis.failure_modes}")
```

### 7. Environment Checks (`environment_checks.py`)

Pre-flight validation and device optimization:

```python
from daf import environment_checks

# Validate environment
result = environment_checks.daf_check_environment(
    missions=["training_facility_1", "assembler_2"]
)

if not result.is_healthy():
    print("Environment issues:", result.warnings)
    exit(1)

# Get recommended device
device = environment_checks.daf_get_recommended_device()
print(f"Using device: {device}")
```

### 8. Orchestrators (`orchestrators/`)

High-level orchestration patterns:

```python
from daf.orchestrators import daf_run_training_pipeline

pipeline = daf_run_training_pipeline(
    policy_class_path="lstm",
    missions=["training_facility_1"],
    num_training_steps=1_000_000,
    eval_missions=["assembler_2"],
    eval_episodes=10
)

print(f"Training complete. Checkpoint: {pipeline.final_checkpoint}")
```

## Testing Infrastructure

DAF includes comprehensive test coverage:

```bash
# Run all DAF tests
cd /Users/4d/Documents/GitHub/cogames
uv sync --all-extras
./daf/tests/run_daf_tests.sh

# Run specific test suite
uv run pytest daf/tests/test_sweeps.py -v
uv run pytest daf/tests/test_comparison.py -v
uv run pytest daf/tests/test_distributed_training.py -v
```

## Configuration Examples

### Hyperparameter Sweep
See `daf/examples/sweep_config.yaml`

### Policy Comparison
See `daf/examples/comparison_config.yaml`

### Training Pipeline
See `daf/examples/pipeline_config.yaml`

## Policy Interface Requirements

### For Sweep-based Training

Policies must implement `MultiAgentPolicy`:

```python
from mettagrid.policy.policy import MultiAgentPolicy
import numpy as np

class MyPolicy(MultiAgentPolicy):
    def __init__(self, env_interface, seed=42):
        super().__init__(env_interface)
        self.seed = seed
    
    def step_batch(self, observations: np.ndarray, actions: np.ndarray) -> None:
        """Required: Process batch of observations."""
        for i in range(len(observations)):
            actions[i] = self.compute_action(observations[i])
```

### For Comparison/Evaluation

Policies must implement `AgentPolicy`:

```python
from mettagrid.policy.policy import AgentPolicy

class MyPolicy(AgentPolicy):
    def step(self, observation):
        """Required: Compute single action."""
        return self.select_action(observation)
```

## Best Practices

1. **Always run environment checks first**
   ```python
   from daf import environment_checks
   result = environment_checks.daf_check_environment()
   assert result.is_healthy()
   ```

2. **Profile single runs before sweeping**
   ```python
   # Test one trial to ensure policy works
   cfg = DAFSweepConfig(...)
   cfg.num_trials = 1
   results = daf_launch_sweep(cfg)
   ```

3. **Use deterministic seeding for reproducibility**
   ```python
   cfg = DAFSweepConfig(seed=42)
   ```

4. **Monitor memory during distributed training**
   ```python
   job = daf_launch_distributed_training(...)
   job.monitor_memory(threshold_gb=8.0)
   ```

5. **Validate checkpoint paths**
   ```python
   from pathlib import Path
   ckpt = Path("checkpoints/model.pt")
   assert ckpt.exists(), f"Checkpoint not found: {ckpt}"
   ```

## Troubleshooting

### Module Import Errors
```python
# Ensure DAF is installed with development dependencies
# In project root:
uv sync --all-extras
```

### Policy Not Found
```python
# Verify policy is importable
import sys
print(sys.path)
# Policy class_path must be resolvable from PYTHONPATH
```

### Out of Memory During Sweeps
```python
# Reduce parallel jobs or episodes
cfg.max_parallel_jobs = 2
cfg.episodes_per_trial = 1
```

## See Also

- [DAF Developer Guide](docs/AGENTS.md) - Custom policy tutorials
- [DAF API Reference](docs/API.md) - Complete API documentation
- [CoGames AGENTS.md](../AGENTS.md) - Top-level agent architecture
- [Running Tests](docs/RUNNING_TESTS.md) - Test execution guide

