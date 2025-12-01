# DAF Output Management Cursorrules

## OutputManager API

Use `OutputManager` for all DAF output organization.

### Initialization

```python
from daf.output_manager import get_output_manager, OutputManager

# Global singleton (recommended)
output_mgr = get_output_manager(
    base_dir="./daf_output",
    verbose=True,
    log_to_file=True,
)

# Or create new instance
output_mgr = OutputManager(
    base_dir="./daf_output",
    verbose=False,
    log_to_file=True,
)
```

### Core Methods

#### get_operation_dir()
```python
# Get output directory for operation
output_dir = output_mgr.get_operation_dir("sweep")
# Returns: Path("./daf_output/sweeps/YYYYMMDD_HHMMSS")

# With subdirectory
output_dir = output_mgr.get_operation_dir("sweep", "experiment_v2")
# Returns: Path("./daf_output/sweeps/experiment_v2")
```

#### Logging Operations

```python
# Log operation start
output_mgr.log_operation_start(
    "sweep",
    details={
        "policy": "lstm",
        "missions": ["training_facility_1"],
        "num_trials": 10,
    }
)

# Log operation complete
output_mgr.log_operation_complete(
    "sweep",
    status="success",  # or "warning", "error"
    details={"best_score": 0.92}
)
```

#### Saving Results

```python
# Save JSON results
output_mgr.save_json_results(
    data={"trial_results": [...]},
    operation="sweep",
    filename="results",  # No extension, adds .json
    subdir=None  # Optional subfolder
)

# Save text results
output_mgr.save_text_results(
    text="Human-readable report",
    operation="sweep",
    filename="report"
)

# Save standardized summary
output_mgr.save_summary_report(
    operation="sweep",
    summary={
        "total_trials": 10,
        "best_score": 0.92,
        "duration_seconds": 125.4,
    }
)
```

#### Session Management

```python
# Save session metadata
output_mgr.save_session_metadata()

# Print output structure
output_mgr.print_output_info()

# Cleanup temporary files
output_mgr.cleanup_temp_files()
```

## Output Directory Structure

### Operation Folders

```
daf_output/
├── sweeps/              # Hyperparameter sweeps
├── comparisons/         # Policy comparisons
├── training/            # Training runs
├── deployment/          # Deployments
├── evaluations/         # Evaluations & tests
├── visualizations/      # Plots & HTML
├── logs/                # Logs & metadata
├── reports/             # Reports
└── artifacts/           # Artifacts
```

### Session Organization

```
daf_output/sweeps/
├── YYYYMMDD_HHMMSS/           # Session ID
│   ├── summary_report.json    # Standardized summary
│   ├── results.json           # Full results
│   ├── trial_logs/            # Trial-specific outputs
│   └── checkpoint_01.pt       # Checkpoints
└── 20241121_143022/           # Another session
    └── ...
```

### Standardized JSON Format

All results follow this format:

```json
{
  "timestamp": "2024-11-21T14:30:22.000Z",
  "operation": "sweep",
  "session_id": "20241121_143022",
  "summary": {
    "total_trials": 10,
    "best_score": 0.92,
    "duration_seconds": 125.4
  },
  "detailed_results": [...]
}
```

## Usage Patterns

### Sweep Operation

```python
from daf.output_manager import get_output_manager
from daf import sweeps

output_mgr = get_output_manager()

# Log start
output_mgr.log_operation_start(
    "sweep",
    details={"num_trials": 10, "policy": "lstm"}
)

try:
    # Execute sweep
    config = DAFSweepConfig.from_yaml("sweep_config.yaml")
    results = sweeps.daf_launch_sweep(config)
    
    # Save results
    output_mgr.save_json_results(
        results.to_dict(),
        operation="sweep",
        filename="results"
    )
    
    # Save summary
    output_mgr.save_summary_report(
        operation="sweep",
        summary={
            "num_trials": len(results.trials),
            "best_score": results.best_score,
            "best_config": results.best_trial.config,
        }
    )
    
    # Complete
    output_mgr.log_operation_complete("sweep", status="success")
    
except Exception as e:
    output_mgr.log_operation_complete("sweep", status="error")
    raise
    
finally:
    output_mgr.save_session_metadata()
```

### Comparison Operation

```python
from daf.output_manager import get_output_manager
from daf import comparison

output_mgr = get_output_manager()
output_mgr.log_operation_start("comparison", details={...})

try:
    results = comparison.daf_compare_policies(...)
    
    output_mgr.save_json_results(
        results.to_dict(),
        operation="comparison",
        filename="comparison_results"
    )
    
    output_mgr.save_summary_report(
        operation="comparison",
        summary={
            "num_policies": len(results.policies),
            "num_missions": len(results.missions),
            "best_policy": results.best_policy,
        }
    )
    
    output_mgr.log_operation_complete("comparison", status="success")
    
except Exception as e:
    output_mgr.log_operation_complete("comparison", status="error")
    raise
finally:
    output_mgr.save_session_metadata()
```

## Output Utilities

### Finding Sessions

```python
from daf.output_utils import (
    find_latest_session,
    list_sessions,
    print_session_info,
)

# Find latest sweep session
latest = find_latest_session("./daf_output", "sweep")
print(f"Latest sweep: {latest}")

# List all recent sessions
sessions = list_sessions("./daf_output", max_results=10)
for session in sessions:
    print(f"{session['id']}: {session['operation']}")

# Print detailed info
print_session_info(latest)
```

### Exporting Sessions

```python
from daf.output_utils import export_session, cleanup_old_sessions

# Export/backup a session
export_session(
    session_dir="./daf_output/sweeps/20241121_143022",
    export_path="./backups/sweep.tar.gz",
    compress=True  # Create .tar.gz archive
)

# Cleanup old sessions (> 30 days)
cleanup_old_sessions(
    output_dir="./daf_output",
    days=30,
    dry_run=True  # Show what would be deleted
)
```

## Best Practices

### DO ✓

- Always use `get_output_manager()` for consistency
- Log operation start and complete
- Save session metadata after operations
- Use standardized JSON format
- Organize by operation type
- Create subdirectories for complex results
- Export/backup important sessions
- Cleanup temporary files

### DON'T ✗

- Create output files outside daf_output/
- Hardcode folder paths
- Forget to log operation lifecycle
- Mix operation types in same folder
- Skip session metadata
- Store very large files in logs/
- Leave temporary files on disk
- Overwrite previous sessions

## Session ID Usage

### Creating Session-Specific Paths

```python
# Session ID automatically set
session_id = output_mgr.session_id
# Returns: "20241121_143022"

# Get operation directory (includes session ID)
op_dir = output_mgr.get_operation_dir("sweep")
# Returns: Path("./daf_output/sweeps/20241121_143022")

# Create subdirectory within session
subdir = op_dir / "trial_01"
subdir.mkdir(parents=True, exist_ok=True)
```

### Reading Session Files

```python
from pathlib import Path
import json

# Read results from latest session
sweep_dir = find_latest_session("./daf_output", "sweep")
results_file = sweep_dir / "results.json"

with open(results_file, "r") as f:
    results = json.load(f)
```

## Error Handling

### Graceful Error Handling

```python
try:
    output_mgr.log_operation_start("operation")
    
    # Perform operation
    results = do_operation()
    
    output_mgr.save_json_results(results, operation="operation")
    output_mgr.log_operation_complete("operation", status="success")
    
except OperationError as e:
    logger.error(f"Operation failed: {e}")
    output_mgr.log_operation_complete("operation", status="error")
    raise

finally:
    # Always save metadata and cleanup
    output_mgr.save_session_metadata()
    output_mgr.cleanup_temp_files()
```

## Configuration

### DAFConfig Output Settings

```python
from daf.config import DAFConfig

config = DAFConfig(
    output_dir=Path("./daf_output"),     # Base output directory
    organize_by_operation=True,          # Use structured folders
    log_to_file=True,                   # Log to files
)
```

### Environment Variables

```bash
# Set output directory
export DAF_OUTPUT_DIR=/custom/path

# Enable verbose output
export DAF_VERBOSE=1
```

---

**Status**: Production Ready ✅
**Last Updated**: November 21, 2024






