# DAF Output Organization

This document describes how DAF organizes all outputs, logs, and artifacts into a unified folder structure.

## Overview

DAF uses a centralized output management system that organizes all operations (sweeps, comparisons, training, etc.) into logical subfolders under a single `daf_output` directory.

### Benefits

- **Consistent Organization**: All DAF operations output to structured, predictable locations
- **Easy Discovery**: Find test results, logs, and reports quickly
- **Session Tracking**: Each session has unique ID for reproducibility
- **Scalability**: Supports large-scale distributed experiments

## Directory Structure

```
./daf_output/
├── sweeps/                  # Hyperparameter sweep results
│   ├── YYYYMMDD_HHMMSS/    # Session ID
│   │   ├── sweep_name/
│   │   │   ├── sweep_results.json       # Full trial results
│   │   │   ├── sweep_progress.png       # Performance over trials
│   │   │   ├── best_configuration.png   # Best hyperparameters
│   │   │   ├── hyperparameter_importance.png
│   │   │   ├── heatmap.png              # Parameter heatmap
│   │   │   └── parallel.png             # Parallel coordinates
│   │   └── ...
│   └── ...
│
├── comparisons/             # Policy comparison results
│   ├── YYYYMMDD_HHMMSS/
│   │   ├── comparison_results.json     # Full results with detailed_metrics
│   │   ├── leaderboard.json            # Policy rankings
│   │   ├── report.html                 # Interactive HTML report
│   │   │
│   │   │   # Basic comparison plots
│   │   ├── policy_rewards_comparison.png
│   │   ├── performance_by_mission.png
│   │   ├── reward_distributions.png
│   │   │
│   │   │   # Detailed metrics plots
│   │   ├── metrics_resources_gained.png
│   │   ├── metrics_resources_held.png
│   │   ├── metrics_energy.png
│   │   ├── metrics_actions.png
│   │   ├── metrics_inventory.png
│   │   ├── metrics_radar.png
│   │   └── action_distribution.png
│   └── ...
│
├── training/                # Training run outputs
│   ├── YYYYMMDD_HHMMSS/
│   │   ├── checkpoints/
│   │   ├── metrics.json
│   │   └── summary_report.json
│   └── ...
│
├── deployment/              # Deployment packages and logs
│   ├── YYYYMMDD_HHMMSS/
│   │   ├── packages/
│   │   ├── validation_results.json
│   │   └── deployment_log.json
│   └── ...
│
├── evaluations/             # Evaluation results
│   ├── tests/               # Test suite outputs
│   │   ├── cogames/
│   │   │   ├── cli_output.txt
│   │   │   ├── core_output.txt
│   │   │   └── ...
│   │   ├── daf/
│   │   │   ├── config_output.txt
│   │   │   ├── sweeps_output.txt
│   │   │   └── ...
│   │   └── test_report.json
│   │
│   ├── YYYYMMDD_HHMMSS/
│   │   ├── mission_analysis/
│   │   └── policy_evaluation/
│   └── ...
│
├── visualizations/          # Generated plots and dashboards
│   ├── YYYYMMDD_HHMMSS/
│   │   ├── training_curves.png
│   │   ├── comparison_plots.png
│   │   ├── comparison_report.html
│   │   └── leaderboard.html
│   └── ...
│
├── full_suite/              # Full evaluation suite outputs
│   ├── suite_YYYYMMDD_HHMMSS/
│   │   ├── SUITE_SUMMARY.json
│   │   ├── SUITE_SUMMARY.txt
│   │   │
│   │   ├── comparisons/     # Policy comparison outputs
│   │   │   ├── comparison_results.json
│   │   │   ├── report.html
│   │   │   ├── metrics_*.png           # All detailed metrics
│   │   │   └── ...
│   │   │
│   │   ├── sweeps/          # Hyperparameter sweep outputs
│   │   │   └── baseline/
│   │   │       ├── sweep_results.json
│   │   │       └── *.png
│   │   │
│   │   └── dashboard/       # Combined dashboard
│   │       ├── dashboard.html
│   │       ├── comparisons/
│   │       └── sweeps/
│   └── ...
│
├── logs/                    # Session and operation logs
│   ├── daf_YYYYMMDD_HHMMSS.log    # Main session log
│   ├── session_YYYYMMDD_HHMMSS.json  # Session metadata
│   ├── test_metrics_YYYYMMDD_HHMMSS.json
│   └── ...
│
├── reports/                 # Generated reports
│   ├── test_summary.json
│   ├── test_report.html
│   ├── comparison_summary.json
│   └── ...
│
├── artifacts/               # Reusable artifacts
│   ├── trained_models/
│   ├── policy_packages/
│   └── datasets/
│
├── .temp/                   # Temporary files (auto-cleaned)
│
├── checkpoints/             # (Deprecated: Use training/checkpoints instead)
│   └── [Kept for backward compatibility]
│
└── TEST_RUN_SUMMARY.txt    # Quick reference for last test run
```

## Session Organization

Each DAF operation creates a session with unique timestamp:

```
YYYYMMDD_HHMMSS
```

Example: `20241121_143022` (November 21, 2024, 14:30:22)

This allows multiple operations to coexist without conflicts while maintaining chronological ordering.

## Output Files

### Summary Reports

Each operation generates `summary_report.json` with standardized format:

```json
{
  "timestamp": "2024-11-21T14:30:22.000Z",
  "operation": "sweep",
  "session_id": "20241121_143022",
  "summary": {
    "total_trials": 10,
    "best_score": 0.85,
    "duration_seconds": 125.4
  }
}
```

### Test Outputs

Test suite outputs organized by category:

```
daf_output/evaluations/tests/
├── cogames/
│   ├── cli_output.txt
│   ├── core_output.txt
│   ├── cvc_assembler_output.txt
│   ├── procedural_output.txt
│   ├── scripted_output.txt
│   ├── train_integration_output.txt
│   ├── vector_alignment_output.txt
│   ├── all_describe_output.txt
│   ├── all_eval_output.txt
│   └── all_play_output.txt
│
├── daf/
│   ├── config_output.txt
│   ├── environment_output.txt
│   ├── sweeps_output.txt
│   ├── comparison_output.txt
│   ├── deployment_output.txt
│   ├── distributed_output.txt
│   ├── visualization_output.txt
│   └── mission_output.txt
│
└── test_report.json
```

### Logs

All logging goes to `daf_output/logs/`:

```
logs/
├── daf_20241121_143022.log        # Rotating file handler (10MB, 5 backups)
├── session_20241121_143022.json   # Session metadata
├── test_metrics_20241121_143022.json
└── ...
```

## Usage

### OutputManager API

```python
from daf.output_manager import get_output_manager

# Get or create output manager
output_mgr = get_output_manager(
    base_dir="./daf_output",
    verbose=True,
    log_to_file=True
)

# Print structure info
output_mgr.print_output_info()

# Save results
output_mgr.save_json_results(
    data={"key": "value"},
    operation="sweep",
    filename="results"
)

# Save summary
output_mgr.save_summary_report(
    operation="comparison",
    summary={"total": 10, "passed": 9}
)

# Save session metadata
output_mgr.save_session_metadata()

# Cleanup
output_mgr.cleanup_temp_files()
```

### TestRunner API

```python
from daf.test_runner import TestRunner

runner = TestRunner(output_base_dir="./daf_output")

# Run test suites
results = runner.run_test_batch([
    ("tests/test_cli.py", "cogames", "cli"),
    ("daf/tests/test_config.py", "daf", "config"),
])

# Save outputs
runner.save_test_outputs()

# Generate and save report
runner.save_test_report()

# Print summary
runner.print_test_summary()
runner.print_failed_tests()

# Cleanup
runner.cleanup()
```

### Generate Test Reports

```bash
# Generate report from saved test outputs
python daf/src/generate_test_report.py \
    ./daf_output/evaluations/tests \
    ./daf_output/reports/test_report.json \
    ./daf_output/reports/test_report.html
```

## Accessing Results

### Find Latest Session

```bash
ls -t daf_output/sweeps/ | head -1
```

### View Session Metadata

```bash
cat daf_output/logs/session_YYYYMMDD_HHMMSS.json | jq
```

### View Test Results

```bash
# Summary
cat daf_output/evaluations/tests/test_report.json | jq

# Detailed output
cat daf_output/evaluations/tests/cogames/cli_output.txt

# HTML report
open daf_output/reports/test_report.html
```

### View Logs

```bash
tail -f daf_output/logs/daf_YYYYMMDD_HHMMSS.log
```

## Configuration

### DAFConfig Settings

```python
from daf.config import DAFConfig

config = DAFConfig(
    output_dir=Path("./daf_output"),         # Base output directory
    checkpoint_dir=Path("./daf_output/checkpoints"),
    organize_by_operation=True,              # Use structured organization
    log_to_file=True,                        # Write to file
    verbose=False,                           # Logging verbosity
    monitor_system_stats=True,              # Track CPU/GPU/memory
)
```

## Best Practices

1. **Always save session metadata** after operations complete:
   ```python
   output_mgr.save_session_metadata()
   ```

2. **Use descriptive operation names** for easy discovery:
   ```python
   output_mgr.get_operation_dir("sweep", "hyperparameter_search_v2")
   ```

3. **Organize results hierarchically**:
   ```
   operation_type/
   └── session_id/
       └── named_subfolder/
           ├── summary_report.json
           ├── detailed_results.json
           └── artifacts/
   ```

4. **Clean up temporary files** when done:
   ```python
   output_mgr.cleanup_temp_files()
   ```

5. **Generate reports immediately** after runs:
   ```bash
   python daf/src/generate_test_report.py daf_output/evaluations/tests
   ```

## Cleaning Up Old Results

Remove old sessions while keeping recent ones:

```bash
# Keep only last 10 days of results
find daf_output -type d -name "20*" -mtime +10 -exec rm -rf {} +
```

## Migration from Old Structure

If migrating from non-organized outputs:

```bash
# Backup old structure
cp -r daf_output daf_output.backup

# Reorganize
python -m daf.utils.migrate_output_structure daf_output.backup daf_output
```

## See Also

- [Logging Configuration](./docs/LOGGING.md) - Structured logging setup
- [Test Infrastructure](./docs/RUNNING_TESTS.md) - Test execution guide
- [DAF README](./README.md) - Module overview







