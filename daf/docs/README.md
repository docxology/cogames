# DAF: Distributed Agent Framework

**DAF is a sidecar utility and extension for CoGames.**

DAF extends CoGames with comprehensive tools for distributed training, hyperparameter optimization, policy comparison, and deployment. DAF wraps and extends core CoGames methods rather than replacing them. All DAF functions use the `daf_` prefix to clearly distinguish from core cogames methods.

## Architecture: Sidecar Pattern

DAF follows a **sidecar utility pattern**:
- **Location**: Lives in `daf/` folder, separate from core `src/cogames/`
- **Extension**: Extends CoGames functionality, never duplicates core methods
- **Invocation**: All DAF functions invoke underlying CoGames methods:
  - `daf_launch_distributed_training()` → wraps `cogames.train.train()`
  - `daf_compare_policies()` → uses `cogames.evaluate.evaluate()`
  - `daf_check_environment()` → uses `cogames.device.resolve_training_device()`
- **Optional**: Core CoGames works independently; DAF requires CoGames

## Overview

DAF provides orchestration for the entire policy development lifecycle:

- **Distributed Training**: Multi-node, multi-GPU training with centralized coordination
- **Hyperparameter Sweeps**: Grid, random, and Bayesian parameter search with parallel execution
- **Policy Comparison**: Head-to-head evaluation with statistical significance testing
- **Deployment Pipeline**: Policy packaging, validation, and production deployment
- **Experiment Tracking**: Centralized logging and analysis of all experiments

## Quick Start

### Environment Checks

Before starting any experiments, validate your environment:

```python
from daf import environment_checks

result = environment_checks.daf_check_environment(
    missions=["training_facility_1"],
    check_cuda=True,
    check_disk=True
)

if not result.is_healthy():
    print("Fix issues before proceeding")
```

Or from CLI:

```bash
cogames daf-check --mission training_facility_1
```

### Hyperparameter Sweep

Define sweeps in YAML:

```yaml
# sweep.yaml
name: "lstm_hyperparameter_search"
missions: ["training_facility_1", "assembler_2"]
policy_class_path: "cogames.policy.lstm.LSTMPolicy"
strategy: "grid"
search_space:
  learning_rate: [0.0001, 0.001, 0.01]
  hidden_size: [64, 128, 256]
episodes_per_trial: 3
```

Launch the sweep:

```python
from daf import config, sweeps

sweep_cfg = config.DAFSweepConfig.from_yaml("sweep.yaml")
results = sweeps.daf_launch_sweep(sweep_cfg)
best = sweeps.daf_sweep_best_config(results)
```

Or from CLI:

```bash
cogames daf-sweep --config sweep.yaml --output results.json
```

### Policy Comparison

Compare policies with statistical analysis:

```python
from daf import comparison
from mettagrid.policy.policy import PolicySpec

policies = [
    PolicySpec(class_path="cogames.policy.lstm.LSTMPolicy", data_path="checkpoints/lstm_v1.pt"),
    PolicySpec(class_path="cogames.policy.lstm.LSTMPolicy", data_path="checkpoints/lstm_v2.pt"),
    PolicySpec(class_path="cogames.policy.scripted_agent.baseline_agent.BaselinePolicy"),
]

missions = ["training_facility_1", "assembler_2"]

report = comparison.daf_compare_policies(
    policies=policies,
    missions=missions,
    episodes_per_mission=5
)

print(report.summary_statistics)
```

Or from CLI:

```bash
cogames daf-compare \
  -p lstm:checkpoints/lstm_v1.pt \
  -p lstm:checkpoints/lstm_v2.pt \
  -p baseline \
  -m training_facility_1 \
  -m assembler_2 \
  --output-html comparison_report.html
```

### Complete Workflows

Orchestrate multi-stage workflows:

```python
from daf import orchestrators, config

# Load pipeline definition
pipeline_cfg = config.DAFPipelineConfig.from_yaml("pipeline.yaml")

# Run complete pipeline with error handling and reporting
result = orchestrators.daf_run_benchmark_pipeline(
    policies=["lstm:ckpt1", "baseline"],
    missions=["training_facility_1", "assembler_2"],
    episodes_per_mission=10,
    generate_report=True
)
```

## Installation

DAF is included with cogames as a sidecar utility. Ensure you have the development dependencies:

```bash
uv pip install -e ".[dev]"
```

**Note**: DAF requires CoGames to function. Core CoGames functionality works independently of DAF.

## Configuration

Global DAF settings can be configured:

```python
from daf.config import DAFConfig

config = DAFConfig(
    output_dir="./my_daf_results",
    max_parallel_jobs=8,
    enable_gpu=True,
    verbose=True
)
```

Or from YAML:

```yaml
# daf_config.yaml
output_dir: "./my_daf_results"
checkpoint_dir: "./my_daf_results/checkpoints"
max_parallel_jobs: 8
enable_gpu: true
verbose: true
```

## Modules

### daf.config
Configuration management for sweeps, deployments, comparisons, and pipelines.

### daf.environment_checks
Environment validation before experiments (GPU, disk space, dependencies, missions).

### daf.distributed_training
Multi-node, multi-GPU training orchestration wrapping cogames.train.train().

### daf.sweeps
Hyperparameter search with grid, random, and Bayesian optimization strategies.

### daf.comparison
Policy comparison with statistical tests and comprehensive reporting.

### daf.visualization
Plotting and analysis visualization for training curves and comparisons.

### daf.deployment
Policy packaging, validation, and deployment pipeline management.

### daf.orchestrators
High-level workflow orchestrators for complete training→evaluation→deployment pipelines.

## Examples

See `/daf/examples/` for complete configuration examples:

- `sweep_config.yaml` - Hyperparameter search configuration
- `pipeline_config.yaml` - Multi-stage workflow orchestration
- `deployment_config.yaml` - Deployment endpoint configuration
- `comparison_config.yaml` - Policy comparison setup

## Development

Refer to `AGENTS.md` for guidance on integrating DAF with custom policies.

See `API.md` for complete API reference of all `daf_*` functions.

Consult `ARCHITECTURE.md` for system design and implementation details.

## Citation

If DAF is useful for your research, please cite the main CoGames paper and acknowledge DAF:

```bibtex
@software{cogames2025,
  title={CoGames: Multi-Agent Cooperative Game Environments},
  author={Metta AI},
  year={2025},
  url={https://github.com/metta-ai/cogames}
}
```

