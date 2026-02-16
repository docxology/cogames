# DAF Source Modules

Core implementation of the Distributed Agent Framework. DAF is a sidecar utility that extends CoGames with distributed training, hyperparameter optimization, policy comparison, and deployment infrastructure.

## Module Structure

```
daf/src/
├── config.py                    # Configuration management
├── sweeps.py                    # Hyperparameter search
├── comparison.py                # Policy comparison
├── distributed_training.py      # Multi-node training
├── mission_analysis.py          # Performance analysis
├── environment_checks.py        # Pre-flight validation
├── visualization.py             # Reports & dashboards
├── deployment.py                # Deployment pipeline
├── logging_config.py            # Structured logging
├── output_manager.py            # Output organization
├── output_utils.py              # Output utilities
├── test_runner.py               # Test execution
├── generate_test_report.py      # Report generation
└── orchestrators/               # Workflow orchestration
    ├── __init__.py
    ├── training_pipeline.py
    ├── sweep_pipeline.py
    ├── comparison_pipeline.py
    └── benchmark_pipeline.py
```

## Quick Start

### 1. Validate Environment

```python
from daf import environment_checks

result = environment_checks.daf_check_environment(
    missions=["training_facility_1"]
)
assert result.is_healthy()
```

### 2. Run Hyperparameter Sweep

```python
from daf import config, sweeps

cfg = config.DAFSweepConfig.from_yaml("sweep.yaml")
results = sweeps.daf_launch_sweep(cfg)
best = sweeps.daf_sweep_best_config(results)
```

### 3. Compare Policies

```python
from daf import comparison
from mettagrid.policy.policy import PolicySpec

results = comparison.daf_compare_policies(
    policies=[
        PolicySpec(class_path="lstm"),
        PolicySpec(class_path="baseline"),
    ],
    missions=["training_facility_1"],
    episodes_per_mission=5
)
print(results.summary_statistics)
```

### 4. Generate Reports

```python
from daf import visualization

visualization.daf_export_comparison_html(
    results,
    output_path="comparison_report.html"
)
```

## Module Overview

| Module | Purpose | Primary Use |
|--------|---------|-------------|
| `config.py` | Configuration management | Define experiments in YAML or Python |
| `sweeps.py` | Hyperparameter search | Find optimal policy parameters |
| `comparison.py` | Policy comparison | Compare multiple policies statistically |
| `distributed_training.py` | Multi-node training | Scale training to multiple machines |
| `mission_analysis.py` | Performance analysis | Analyze per-mission results |
| `environment_checks.py` | Environment validation | Pre-flight checks before experiments |
| `visualization.py` | HTML reports | Generate dashboards and plots |
| `deployment.py` | Deployment pipeline | Package and deploy policies |
| `logging_config.py` | Structured logging | Track metrics and operations |
| `output_manager.py` | Output organization | Manage organized output directories |
| `orchestrators/` | Workflow orchestration | Chain operations into complete pipelines |

## Common Workflows

### Workflow 1: Hyperparameter Optimization

```python
from daf import environment_checks, config, sweeps

# 1. Validate environment
env_result = environment_checks.daf_check_environment()
assert env_result.is_healthy()

# 2. Load sweep configuration
sweep_cfg = config.DAFSweepConfig.from_yaml("sweep.yaml")

# 3. Run sweep
results = sweeps.daf_launch_sweep(sweep_cfg)

# 4. Get best configuration
best = sweeps.daf_sweep_best_config(results)
print(f"Best hyperparameters: {best.hyperparameters}")
```

### Workflow 2: Policy Comparison & Analysis

```python
from daf import comparison, visualization, mission_analysis
from mettagrid.policy.policy import PolicySpec

# 1. Compare policies across missions
policies = [
    PolicySpec(class_path="lstm", data_path="ckpt/v1.pt"),
    PolicySpec(class_path="lstm", data_path="ckpt/v2.pt"),
    PolicySpec(class_path="baseline"),
]

results = comparison.daf_compare_policies(
    policies=policies,
    missions=["training_facility_1", "assembler_2"],
    episodes_per_mission=10
)

# 2. Analyze per-mission performance
for mission in ["training_facility_1", "assembler_2"]:
    analysis = mission_analysis.daf_analyze_mission(mission)
    print(f"{mission}: success_rate={analysis.success_rate}")

# 3. Generate HTML report
visualization.daf_export_comparison_html(
    results,
    output_path="comparison_report.html"
)
```

### Workflow 3: Distributed Training

```python
from daf import environment_checks, distributed_training
import torch

# 1. Check environment
env_result = environment_checks.daf_check_environment()
device = environment_checks.daf_get_recommended_device()

# 2. Load mission and environment
from cogames.cli.mission import get_mission_name_and_config
_, env_cfg, _ = get_mission_name_and_config(None, "training_facility_1")

# 3. Launch distributed training
job = distributed_training.daf_launch_distributed_training(
    env_cfg=env_cfg,
    policy_class_path="lstm",
    device=device,
    num_steps=1_000_000,
    num_machines=4,
)

# 4. Wait for completion
final_checkpoint = job.wait_for_completion()
print(f"Training complete: {final_checkpoint}")
```

### Workflow 4: End-to-End Pipeline

```python
from daf import orchestrators

# Run complete training → sweep → comparison → deployment pipeline
result = orchestrators.daf_run_benchmark_pipeline(
    policies=["lstm"],
    missions=["training_facility_1", "assembler_2"],
    episodes_per_mission=5,
)

print(f"Pipeline status: {result.status}")
print(f"Stages completed: {result.stages_completed}")
```

## Configuration

Each module accepts YAML configuration files for reproducible experiments.

**Example sweep_config.yaml:**
```yaml
name: "lstm_optimization"
missions: ["training_facility_1"]
policy_class_path: "lstm"
strategy: "grid"
search_space:
  learning_rate: [0.0001, 0.001, 0.01]
  hidden_size: [64, 128, 256]
episodes_per_trial: 5
```

**Example pipeline_config.yaml:**
```yaml
name: "full_pipeline"
stages:
  - training
  - sweep
  - comparison

training_config:
  policy_class_path: "lstm"
  num_training_steps: 1000000

sweep_config:
  missions: ["training_facility_2"]
  strategy: "random"
  num_trials: 10
```

See `../examples/` for complete configuration examples.

## Output Organization

All DAF operations save organized outputs to structured subfolders:

```
daf_output/
├── sweeps/              # Sweep results by session
├── comparisons/         # Comparison reports
├── training/            # Training runs
├── deployment/          # Deployment packages
├── evaluations/         # Test results
├── visualizations/      # Generated HTML/plots
├── logs/                # Operation logs
├── reports/             # Generated reports
└── artifacts/           # Reusable artifacts
```

Each operation includes a session ID (`YYYYMMDD_HHMMSS`) for reproducibility.

## Development

### Adding a New Module

1. Create `module_name.py` in `daf/src/`
2. Implement functions with `daf_` prefix
3. Wrap underlying CoGames methods (don't duplicate)
4. Add comprehensive docstrings
5. Create tests in `daf/tests/test_module_name.py`
6. Document in `daf/src/README.md` and `daf/src/AGENTS.md`

### Module Pattern

```python
"""Module description and CoGames integration point."""

from daf.logging_config import create_daf_logger
from daf.output_manager import get_output_manager

logger = create_daf_logger(__name__)
output_mgr = get_output_manager()

def daf_operation(param1, param2, console=None):
    """Execute operation.
    
    Args:
        param1: Parameter description
        param2: Parameter description
        console: Optional Rich console for output
        
    Returns:
        Result object or dict with results
    """
    logger.info(f"Starting operation with {param1}")
    
    # Wrap underlying CoGames method
    result = cogames_operation(param1, param2)
    
    # Save to organized output
    output_mgr.save_json_results(result, operation="operation_type")
    
    logger.info("Operation complete")
    return result
```

## Testing

Run all source module tests:

```bash
python -m pytest daf/tests/test_*.py -v
```

Run specific module tests:

```bash
python -m pytest daf/tests/test_sweeps.py -v
python -m pytest daf/tests/test_comparison.py -v
```

See `../tests/README.md` for detailed testing guide.

## See Also

- `AGENTS.md` - Agent and policy support details
- `orchestrators/README.md` - Workflow orchestration
- `../AGENTS.md` - Top-level agent architecture
- `../docs/API.md` - Complete API reference
- `../docs/ARCHITECTURE.md` - System design
- `../tests/README.md` - Testing guide







