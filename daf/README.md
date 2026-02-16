# DAF: Distributed Agent Framework

DAF extends CoGames with distributed training, evaluation, and analysis infrastructure for agent policies. It provides hyperparameter sweeps, policy comparisons, CogsGuard mission analysis, VOR-based scoring, authentication integration, and comprehensive visualization and reporting.

> **v2.1.0**: Added CogsGuard mission support, authentication integration, `metta_alo` scoring, policy framework analysis, VOR comparison, variant sweeps, and tournament submission workflow.

## Key Features

### 1. **Hyperparameter Sweeps** (`sweeps.py`)

Grid/random search over policy hyperparameters with result tracking.

```python
from daf.config import DAFSweepConfig
from daf import sweeps

config = DAFSweepConfig.from_yaml("sweep_config.yaml")
results = sweeps.daf_launch_sweep(config)
best = sweeps.daf_sweep_best_config(results)
```

### 2. **Policy Comparison** (`comparison.py`)

Statistical analysis and pairwise comparison of multiple policies.

```python
from daf import comparison

results = comparison.daf_compare_policies(
    policies=["cogames.policy.starter_agent.StarterPolicy", "cogames.policy.tutorial_policy.TutorialPolicy"],
    missions=["cogsguard_machina_1.basic"],
    episodes_per_mission=10,
)
print(results.summary_statistics)
```

### 3. **Distributed Training** (`distributed_training.py`)

Multi-machine training orchestration with automatic synchronization.

```python
from daf import distributed_training
import torch

job = distributed_training.daf_launch_distributed_training(
    env_cfg=env_cfg,  # From get_mission()
    policy_class_path="cogames.policy.tutorial_policy.TutorialPolicy",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    num_steps=250_000,
)
# Returns DistributedTrainingResult, not a job object in v2.2
print(f"Training complete: {job.status}")
```

### 4. **Organized Output Management** (`output_manager.py`)

**New in v2**: Centralized output organization with structured subfolders for all operations.

```python
from daf.output_manager import get_output_manager

output_mgr = get_output_manager(base_dir="./daf_output")
output_mgr.print_output_info()

# Save results to organized folders
output_mgr.save_json_results(
    results,
    operation="sweep",
    filename="results",
)
```

### 5. **Structured Logging** (`logging_config.py`)

**New in v2**: Enhanced logging with operation tracking and metrics collection.

```python
from daf.logging_config import create_daf_logger

logger = create_daf_logger("my_experiment")

with logger.track_operation("training"):
    logger.info("Starting training...")
    # Your code here
    
logger.print_metrics_summary()
```

### 6. **Unified Test Runner** (`test_runner.py`)

**New in v2**: Execute all tests with organized output and reporting.

```python
from daf.test_runner import TestRunner

runner = TestRunner(output_base_dir="./daf_output")
runner.run_test_batch([
    ("tests/test_cli.py", "cogames", "cli"),
    ("daf/tests/test_config.py", "daf", "config"),
])
runner.save_test_report()
runner.print_test_summary()
```

### 7. **Report Generation** (`generate_test_report.py`)

**New in v2**: Generate comprehensive test reports from saved outputs.

```bash
python daf/src/generate_test_report.py \
    ./daf_output/evaluations/tests \
    ./daf_output/reports/test_report.json \
    ./daf_output/reports/test_report.html
```

### 8. **Visualization** (`visualization.py`)

Generate HTML reports and dashboards for comparison results.

```python
from daf import visualization, comparison

results = comparison.daf_compare_policies(...)
visualization.daf_export_comparison_html(
    results,
    output_path="comparison_report.html"
)
```

### 9. **Mission Analysis** (`mission_analysis.py`)

Per-mission performance analysis and failure mode detection.

```python
from daf import mission_analysis

analysis = mission_analysis.daf_analyze_mission_performance(
    policy_class_path="lstm",
    mission_name="cogsguard_machina_1",
    episodes=20,
)
print(f"Success rate: {analysis.success_rate}")
```

### 10. **Environment Checks** (`environment_checks.py`)

Pre-flight validation and device optimization.

```python
from daf import environment_checks

result = environment_checks.daf_check_environment()
if not result.is_healthy():
    print("Environment issues:", result.warnings)
```

## Output Organization

**New in v2**: All DAF operations save organized outputs to structured subfolders:

```
daf_output/
├── sweeps/              # Sweep results
├── comparisons/         # Comparison results
├── training/            # Training runs
├── deployment/          # Deployment packages
├── evaluations/         # Test results & evals
├── visualizations/      # Generated plots/HTML
├── logs/                # Session logs
├── reports/             # Generated reports
└── artifacts/           # Reusable artifacts
```

Each operation is organized by session ID (`YYYYMMDD_HHMMSS`) for reproducibility. See [OUTPUT_ORGANIZATION.md](./docs/OUTPUT_ORGANIZATION.md) for details.

## Module Overview

| Module | Purpose |
|--------|---------|
| `config.py` | Configuration management (DAFConfig, DAFSweepConfig, DAFTournamentConfig, DAFVariantConfig) |
| `output_manager.py` | Centralized output directory management |
| `logging_config.py` | Structured logging with metrics tracking |
| `test_runner.py` | Unified test execution framework |
| `sweeps.py` | Hyperparameter sweep execution, variant sweeps |
| `comparison.py` | Policy comparison, statistical testing, VOR comparison |
| `training.py` | Policy training orchestration |
| `distributed_training.py` | Multi-machine training coordination |
| `deployment.py` | Policy packaging, deployment, tournament submission |
| `mission_analysis.py` | CogsGuard mission analysis, variant discovery |
| `visualization.py` | HTML reports and dashboards |
| `environment_checks.py` | Environment validation, auth checks |
| `auth_integration.py` | **[NEW 2.1]** OAuth2 authentication wrapper |
| `scoring_analysis.py` | **[NEW 2.1]** `metta_alo` scoring (VOR, weighted scores) |
| `policy_analysis.py` | **[NEW 2.1]** Policy framework discovery/comparison |
| `generate_test_report.py` | Test report generation from outputs |

## Quick Start

### 1. Install Dependencies

```bash
cd /Users/4d/Documents/GitHub/cogames
uv sync --all-extras
```

### 2. Run Tests

```bash
# Run all tests with organized output
./daf/tests/run_daf_tests.sh

# Or run specific test suite
python -m pytest daf/tests/test_sweeps.py -v
```

### 3. Generate Test Report

```bash
python daf/src/generate_test_report.py \
    ./daf_output/evaluations/tests \
    ./daf_output/reports/test_report.json \
    ./daf_output/reports/test_report.html

# View HTML report
open ./daf_output/reports/test_report.html
```

### 4. Run a Sweep

```bash
python -c "
from daf.config import DAFSweepConfig
from daf import sweeps

config = DAFSweepConfig.from_yaml('daf/examples/sweep_config.yaml')
results = sweeps.daf_launch_sweep(config)
"
```

## Examples

See `daf/examples/` for complete examples:

- `output_management_example.py` - Using output management and logging
- `sweep_config.yaml` - Hyperparameter sweep configuration
- `comparison_config.yaml` - Policy comparison setup
- `pipeline_config.yaml` - Workflow orchestration

Run examples:

```bash
python daf/examples/output_management_example.py
```

## Testing Infrastructure

DAF includes comprehensive tests:

```bash
# Run all DAF tests
python -m pytest daf/tests/ -v

# Run specific test module
python -m pytest daf/tests/test_sweeps.py -v

# Run with coverage
python -m pytest daf/tests/ --cov=daf/src

# Generate test report
./daf/tests/run_daf_tests.sh
```

Test outputs are organized in `daf_output/evaluations/tests/`:

- `cogames/` - CoGames framework tests
- `daf/` - DAF module tests
- `test_report.json` - Structured results
- `test_report.html` - Interactive HTML report

See [RUNNING_TESTS.md](./docs/RUNNING_TESTS.md) for detailed testing guide.

## Configuration

### DAFConfig - Global Settings

```python
from daf.config import DAFConfig

config = DAFConfig(
    output_dir="./daf_output",              # Root output directory
    checkpoint_dir="./daf_output/checkpoints",
    organize_by_operation=True,             # Use structured folders
    max_parallel_jobs=4,                    # Parallel execution
    enable_gpu=True,                        # GPU support
    verbose=False,                          # Logging level
    log_to_file=True,                       # File logging
    track_experiments=True,                 # Experiment tracking
)
```

### DAFSweepConfig - Sweep Settings

```yaml
name: lstm_hyperparameter_sweep
missions:
  - training_facility_1
policy_class_path: lstm
search_space:
  learning_rate: [0.001, 0.01, 0.1]
  batch_size: [32, 64, 128]
strategy: grid
episodes_per_trial: 5
optimize_direction: maximize
objective_metric: avg_reward
```

## Best Practices

1. **Always use OutputManager** for consistent organization:

   ```python
   output_mgr = get_output_manager()
   output_mgr.log_operation_start("sweep")
   ```

2. **Create loggers for experiments**:

   ```python
   logger = create_daf_logger("experiment_name")
   with logger.track_operation("training"):
       # Your code
   ```

3. **Save session metadata** when done:

   ```python
   output_mgr.save_session_metadata()
   ```

4. **Generate reports immediately** after runs:

   ```bash
   python daf/src/generate_test_report.py daf_output/evaluations/tests
   ```

5. **Clean up old sessions periodically**:

   ```bash
   find daf_output -type d -name "20*" -mtime +30 -exec rm -rf {} +
   ```

## Documentation

- [OUTPUT_ORGANIZATION.md](./docs/OUTPUT_ORGANIZATION.md) - Output folder structure
- [RUNNING_TESTS.md](./docs/RUNNING_TESTS.md) - Test execution guide
- [AGENTS.md](./AGENTS.md) - Agent and policy infrastructure
- [API.md](./docs/API.md) - Complete API reference
- [ARCHITECTURE.md](./docs/ARCHITECTURE.md) - System architecture

## Troubleshooting

### Output not being saved?

Check that `organize_by_operation=True` in DAFConfig and output manager is initialized.

### Tests not collecting outputs?

Ensure test script uses output manager:

```bash
python -m pytest tests/ -v 2>&1 | tee "$OUTPUT_DIR/test.txt"
```

### Logs not appearing?

Enable logging:

```python
output_mgr = get_output_manager(verbose=True, log_to_file=True)
```

### Reports not generating?

Run report generator:

```bash
python daf/src/generate_test_report.py ./daf_output/evaluations/tests
```

## Version History

### v2.0 (Current)

- **New**: Centralized OutputManager for organized outputs
- **New**: Structured logging with DAFLogger
- **New**: Unified TestRunner for test execution
- **New**: Report generation from saved outputs
- **Improved**: Output organization by operation type
- **Improved**: Structured session metadata tracking

### v1.0

- Initial DAF implementation with core modules

## Contributing

When adding new DAF operations:

1. Update configuration in `config.py`
2. Use OutputManager for output organization
3. Use DAFLogger for logging
4. Add tests in `daf/tests/`
5. Document in appropriate module docs
6. Add example configuration to `daf/examples/`

## See Also

- [CoGames AGENTS.md](../AGENTS.md) - Top-level agent architecture
- [CoGames README](../README.md) - Main project overview
- [Technical Manual](../TECHNICAL_MANUAL.md) - Technical specifications
