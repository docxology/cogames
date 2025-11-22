# DAF Examples

Example configurations and scripts demonstrating DAF usage with different policy types and workflows.

## Example Files

```
examples/
├── sweep_config.yaml                # Hyperparameter sweep with LSTM
├── comparison_config.yaml           # Multi-policy comparison setup
├── pipeline_config.yaml             # End-to-end workflow orchestration
└── output_management_example.py     # Output management and logging
```

## Quick Start

### Example 1: Hyperparameter Sweep

Optimize LSTM policy hyperparameters using grid search.

**File:** `sweep_config.yaml`

**Run:**
```bash
python -c "
from daf.config import DAFSweepConfig
from daf import sweeps

cfg = DAFSweepConfig.from_yaml('daf/examples/sweep_config.yaml')
results = sweeps.daf_launch_sweep(cfg)
best = sweeps.daf_sweep_best_config(results)
print(f'Best config: {best.hyperparameters}')
"
```

**What it shows:**
- Grid search strategy
- Built-in policy (LSTM)
- Multiple missions
- Hyperparameter space definition

### Example 2: Policy Comparison

Compare multiple policies with statistical testing.

**Usage:**
```bash
python -c "
from daf import comparison
from mettagrid.policy.policy import PolicySpec

policies = [
    PolicySpec(class_path='lstm'),
    PolicySpec(class_path='baseline'),
]

results = comparison.daf_compare_policies(
    policies=policies,
    missions=['training_facility_1'],
    episodes_per_mission=5,
)

print(results.summary_statistics)
"
```

**What it shows:**
- Multi-policy evaluation
- Statistical significance testing
- Report generation

### Example 3: Pipeline Orchestration

Chain multiple workflow stages (train → sweep → compare).

**File:** `pipeline_config.yaml`

**Run:**
```bash
python -c "
from daf.config import DAFSweepConfig
from daf.orchestrators import daf_run_sweep_pipeline

cfg = DAFSweepConfig.from_yaml('daf/examples/sweep_config.yaml')
result = daf_run_sweep_pipeline(config=cfg)

if result.is_success():
    print('Pipeline completed successfully')
"
```

**What it shows:**
- Multi-stage workflows
- Configuration-driven execution
- Error handling

### Example 4: Output Management

Organize outputs and track metrics.

**File:** `output_management_example.py`

**Run:**
```bash
python daf/examples/output_management_example.py
```

**What it shows:**
- Output directory structure
- Structured logging
- Metrics collection
- Session management

## Configuration Examples

### sweep_config.yaml

Grid search over learning rate and hidden size:

```yaml
name: "lstm_hyperparameter_grid_search"
missions: ["training_facility_1", "training_facility_2"]
policy_class_path: "lstm"
strategy: "grid"
search_space:
  learning_rate: [0.0001, 0.0005, 0.001]
  hidden_size: [64, 128, 256]
episodes_per_trial: 3
objective_metric: "avg_reward_per_agent"
optimize_direction: "maximize"
```

**Customize:**

Change search space:
```yaml
search_space:
  learning_rate: [0.00001, 0.0001, 0.001, 0.01]
  hidden_size: [32, 64, 128, 256, 512]
```

Change missions:
```yaml
missions:
  - "training_facility_1"
  - "assembler_1"
  - "assembler_2"
```

Use random search:
```yaml
strategy: "random"
num_trials: 20
```

### comparison_config.yaml

Multi-policy comparison setup (reference only):

```yaml
name: "policy_comparison"
policies:
  - "lstm"
  - "baseline"
  - "random"
missions:
  - "training_facility_1"
  - "assembler_2"
episodes_per_mission: 10
generate_html_report: true
```

Note: Direct YAML loading for comparison not supported; use programmatic API.

### pipeline_config.yaml

Multi-stage workflow orchestration:

```yaml
name: "full_agent_development_pipeline"
stages:
  - "training"
  - "sweep"
  - "comparison"

training_config:
  policy_class_path: "lstm"
  missions: ["training_facility_1"]
  num_training_steps: 1000000

sweep_config:
  policy_class_path: "lstm"
  missions: ["training_facility_2"]
  strategy: "grid"
  search_space:
    learning_rate: [0.0001, 0.001]
  episodes_per_trial: 5

comparison_config:
  policies: ["lstm", "baseline"]
  missions: ["assembler_2"]
  episodes_per_mission: 10

stop_on_failure: true
parallel_stages: false
```

## Using Examples with Custom Policies

### Create Custom Policy Package

```python
# myproject/policies.py
from mettagrid.policy.policy import MultiAgentPolicy
import torch
import torch.nn as nn

class MyPolicy(MultiAgentPolicy):
    def __init__(self, env_interface, learning_rate=0.001, seed=42):
        super().__init__(env_interface)
        self.learning_rate = learning_rate
        self.network = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
        )
    
    def step_batch(self, observations, actions):
        for agent_id in range(observations.shape[0]):
            obs = torch.from_numpy(
                observations[agent_id].flatten()
            ).float()
            with torch.no_grad():
                logits = self.network(obs)
                actions[agent_id] = logits.argmax().item()
```

### Create Sweep Config

```yaml
# my_sweep.yaml
name: "my_policy_sweep"
policy_class_path: "myproject.policies.MyPolicy"
missions: ["training_facility_1"]
strategy: "grid"
search_space:
  learning_rate: [0.0001, 0.001, 0.01]
episodes_per_trial: 3
```

### Run Sweep

```bash
python -c "
import sys
sys.path.insert(0, '.')  # Add current directory to path

from daf.config import DAFSweepConfig
from daf import sweeps

cfg = DAFSweepConfig.from_yaml('my_sweep.yaml')
results = sweeps.daf_launch_sweep(cfg)
best = sweeps.daf_sweep_best_config(results)
print(f'Best learning rate: {best.hyperparameters[\"learning_rate\"]}')
"
```

## Workflow Patterns

### Pattern 1: Simple Sweep

```python
from daf import config, sweeps

cfg = config.DAFSweepConfig.from_yaml('sweep.yaml')
results = sweeps.daf_launch_sweep(cfg)
```

### Pattern 2: Sweep with Comparison

```python
from daf import config, sweeps, comparison
from mettagrid.policy.policy import PolicySpec

cfg = config.DAFSweepConfig.from_yaml('sweep.yaml')
sweep_results = sweeps.daf_launch_sweep(cfg)
best = sweeps.daf_sweep_best_config(sweep_results)

comparison_results = comparison.daf_compare_policies(
    policies=[
        PolicySpec(
            class_path="lstm",
            data_path=best.checkpoint_path
        ),
        PolicySpec(class_path="baseline"),
    ],
    missions=["assembler_2"],
    episodes_per_mission=10,
)
```

### Pattern 3: Training Pipeline

```python
from daf import orchestrators

result = orchestrators.daf_run_training_pipeline(
    policy_class_path="lstm",
    missions=["training_facility_1"],
    num_training_steps=1_000_000,
    eval_missions=["assembler_2"],
    eval_episodes=10,
)
```

### Pattern 4: Benchmark Pipeline

```python
from daf import orchestrators

result = orchestrators.daf_run_benchmark_pipeline(
    policies=["lstm", "baseline"],
    missions=["training_facility_1", "assembler_2"],
    episodes_per_mission=5,
)
```

## Output Organization

Examples generate organized outputs:

```
daf_output/
├── sweeps/YYYYMMDD_HHMMSS/
│   ├── config.yaml              # Sweep configuration
│   ├── results.json             # Results data
│   ├── best_config.json         # Best configuration
│   └── trials/                  # Individual trial results
├── comparisons/YYYYMMDD_HHMMSS/
│   ├── results.json             # Comparison results
│   ├── summary_stats.json       # Statistical summary
│   └── report.html              # HTML report
├── visualizations/
│   ├── comparison_report.html
│   └── training_curves.html
└── logs/
    └── session_*.log
```

## Best Practices

1. **Start small, then scale**
   ```yaml
   episodes_per_trial: 1  # Test configuration first
   ```

2. **Use multiple missions**
   ```yaml
   missions:
     - "training_facility_1"
     - "training_facility_2"
     - "assembler_2"
   ```

3. **Set clear optimization objective**
   ```yaml
   objective_metric: "avg_reward"
   optimize_direction: "maximize"
   ```

4. **Store best checkpoints**
   ```yaml
   checkpoint_best_n: 3  # Keep 3 best checkpoints
   ```

5. **Enable reproducibility**
   ```yaml
   seed: 42
   ```

## Troubleshooting

### Policy Not Found

```bash
# Ensure policy module is in PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
python -c "from daf.config import DAFSweepConfig; ..."
```

### Mission Not Found

```python
# Verify mission exists
from daf import mission_analysis
missions = mission_analysis.daf_discover_missions_from_readme()
print(missions)  # Check if your mission is listed
```

### Configuration Validation Error

```python
# Validate configuration
from daf.config import DAFSweepConfig

try:
    cfg = DAFSweepConfig.from_yaml('sweep.yaml')
except Exception as e:
    print(f"Configuration error: {e}")
```

### Memory Issues

```yaml
# Reduce parallel jobs
max_parallel_jobs: 1

# Reduce episodes
episodes_per_trial: 1
```

## See Also

- `AGENTS.md` - Policy interface details and examples
- `sweep_config.yaml` - Sweep configuration reference
- `comparison_config.yaml` - Comparison configuration
- `pipeline_config.yaml` - Pipeline configuration
- `output_management_example.py` - Output management example
- `../README.md` - DAF overview
- `../src/README.md` - Source modules
- `../../README.md` - CoGames overview



