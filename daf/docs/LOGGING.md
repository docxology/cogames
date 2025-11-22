# DAF Logging Configuration

This document describes the structured logging system for DAF operations.

## Overview

DAF provides a comprehensive logging system with:

- **Structured Output**: Organized logs with operation tracking
- **Performance Metrics**: Automatic timing and metric collection
- **Multiple Handlers**: Console and file output
- **Contextual Tracking**: Rich metadata for each operation
- **Session Management**: Unique session IDs for reproducibility

## DAFLogger

The `DAFLogger` class provides structured logging with metrics tracking.

### Basic Usage

```python
from daf.logging_config import create_daf_logger
from pathlib import Path

# Create logger
logger = create_daf_logger(
    name="my_experiment",
    log_dir=Path("./logs"),
    verbose=False,  # Set to True for DEBUG level
)

# Log messages
logger.info("Starting experiment")
logger.debug("Debug information")
logger.warning("Warning message")
logger.error("Error occurred")
```

### Operation Tracking

Track operations with automatic timing:

```python
# Using context manager
with logger.track_operation("training", metadata={"epochs": 10}):
    logger.info("Starting training...")
    # Your training code
    logger.debug("Epoch 1 complete")

# Check metrics
logger.print_metrics_summary()
```

### Console Output

Format output with styled sections and tables:

```python
# Print sections
logger.print_section("Training Phase 1", level=1)  # Bold header
logger.print_section("Data Loading", level=2)      # Underlined header
logger.print_section("Step 1", level=3)             # Bullet point

# Print tables
data = [
    {"Model": "LSTM", "Accuracy": 0.92, "Duration": "5.2s"},
    {"Model": "Baseline", "Accuracy": 0.85, "Duration": "2.1s"},
]
logger.print_table("Model Comparison", data)

# Print metrics
logger.print_metrics_summary()
```

### Saving Metrics

Save tracked metrics to JSON:

```python
metrics_file = logger.save_metrics_json(Path("./output/metrics.json"))
```

## OperationTracker

The `OperationTracker` class tracks multiple operations:

```python
from daf.logging_config import OperationTracker

tracker = OperationTracker()

# Track operations
with tracker.track("load_data"):
    # Load data
    pass

with tracker.track("train_model"):
    # Train model
    pass

# Get summary
summary = tracker.get_summary()
print(summary)
# {
#   "total_operations": 2,
#   "successful": 2,
#   "failed": 0,
#   "total_duration_seconds": 10.5,
#   "operations": [...]
# }
```

## OutputManager Integration

The `OutputManager` includes integrated logging:

```python
from daf.output_manager import get_output_manager

output_mgr = get_output_manager(
    base_dir="./daf_output",
    verbose=True,
    log_to_file=True,
)

# Logging is automatically set up
output_mgr.logger.info("Starting operation")

# Track operations
output_mgr.log_operation_start("sweep", details={"trials": 10})

# ... perform sweep ...

output_mgr.log_operation_complete("sweep", status="success")

# Save session logs
output_mgr.save_session_metadata()
```

## Log File Organization

Logs are saved to `daf_output/logs/`:

```
logs/
├── daf_20241121_143022.log           # Main session log (rotating)
├── session_20241121_143022.json      # Session metadata
└── ...
```

### Rotating Handler Configuration

The logging system uses rotating file handlers:

- **Max Size**: 10 MB per file
- **Backup Count**: 5 old files kept
- **Format**: `[timestamp] logger_name - level - message`

## Configuration

### Environment Variables

Control logging via environment variables:

```bash
# Enable verbose logging
export DAF_VERBOSE=1

# Disable file logging
export DAF_LOG_TO_FILE=0

# Set log directory
export DAF_LOG_DIR=/path/to/logs
```

### Python Configuration

```python
from daf.output_manager import get_output_manager

output_mgr = get_output_manager(
    base_dir="./daf_output",
    verbose=True,           # Enable DEBUG level
    log_to_file=True,       # Write to file
)

# Access logger
logger = output_mgr.logger
logger.info("Message")
```

## Log Levels

DAFLogger supports standard Python logging levels:

| Level | Usage | Example |
|-------|-------|---------|
| DEBUG | Detailed diagnostic info | `logger.debug("Variable x = 42")` |
| INFO | Confirmation of operations | `logger.info("Training started")` |
| WARNING | Warning messages | `logger.warning("Low memory")` |
| ERROR | Error messages | `logger.error("Failed to load")` |

Level is controlled by `verbose` flag:
- `verbose=False` → INFO level (production)
- `verbose=True` → DEBUG level (development)

## Structured Logging Example

Complete example with proper logging:

```python
from daf.logging_config import create_daf_logger
from pathlib import Path

# Create logger
logger = create_daf_logger(
    name="hyperparameter_sweep",
    log_dir=Path("./logs"),
    verbose=True,
)

logger.print_section("Hyperparameter Sweep", level=1)

# Track main operation
with logger.track_operation("sweep", metadata={
    "num_trials": 10,
    "policy": "lstm",
}):
    logger.info("Starting sweep")
    
    # Track sub-operations
    with logger.track_operation("load_missions"):
        logger.info("Loading missions...")
        # Load missions
        logger.debug("Loaded 5 missions")
    
    with logger.track_operation("run_trials"):
        logger.info("Running trials...")
        for i in range(10):
            logger.debug(f"Trial {i+1}/10 complete")
    
    logger.info("Sweep complete")

# Print summary
logger.print_section("Results", level=2)
logger.print_metrics_summary()

# Save metrics
logger.save_metrics_json(Path("./output/metrics.json"))
```

## Best Practices

1. **Use context managers for operations**:
   ```python
   with logger.track_operation("my_op", metadata={...}):
       # Your code
   ```

2. **Provide meaningful metadata**:
   ```python
   logger.start_operation(
       "training",
       details={
           "model": "lstm",
           "learning_rate": 0.01,
           "batch_size": 32,
       }
   )
   ```

3. **Use appropriate log levels**:
   ```python
   logger.info("Major milestone reached")
   logger.debug("Detailed diagnostic info")
   logger.warning("Potential issue")
   logger.error("Something failed")
   ```

4. **Print sections for organization**:
   ```python
   logger.print_section("Training Phase", level=1)
   logger.print_section("Loading Data", level=2)
   ```

5. **Save metrics after operations**:
   ```python
   logger.print_metrics_summary()
   logger.save_metrics_json(Path("./metrics.json"))
   ```

## Integrating with DAF Operations

### Sweeps

```python
from daf import sweeps
from daf.logging_config import create_daf_logger

logger = create_daf_logger("sweep_operation")

with logger.track_operation("sweep", metadata={
    "policy": "lstm",
    "missions": ["training_facility_1"],
}):
    results = sweeps.daf_launch_sweep(config)
    logger.info(f"Best score: {results.best_score}")
```

### Comparisons

```python
from daf import comparison
from daf.logging_config import create_daf_logger

logger = create_daf_logger("comparison_operation")

with logger.track_operation("comparison", metadata={
    "policies": ["lstm", "baseline", "random"],
    "missions": ["training_facility_1"],
}):
    results = comparison.daf_compare_policies(...)
    logger.info("Comparison complete")
```

### Tests

```python
from daf.test_runner import TestRunner
from daf.logging_config import create_daf_logger

runner = TestRunner(output_base_dir="./daf_output", verbose=True)

# TestRunner creates its own logger
runner.logger.info("Starting tests")
runner.run_test_batch([...])
runner.print_test_summary()
```

## Function Call Tracking

DAF provides infrastructure for tracking function calls during test execution and debugging:

### Basic Usage

```python
from daf.logging_config import create_daf_logger, track_function_calls

# Create logger with call tracking enabled
logger = create_daf_logger(
    name="debug_session",
    track_calls=True
)

# Use decorator to track function calls
@track_function_calls(logger)
def my_function(x, y):
    return x + y

# Call function (automatically logged)
result = my_function(10, 20)

# Save call traces
logger.save_call_traces("function_calls.json")
```

### Advanced Tracking

Record calls manually:

```python
logger.record_call(
    function_name="complex_operation",
    module="mymodule",
    args=(arg1, arg2),
    kwargs={"option": value},
    result=output_value,
    duration_ms=125.5
)
```

### Analyzing Traces

Load and analyze saved traces:

```python
import json
from pathlib import Path

traces_file = Path("function_calls.json")
with open(traces_file, "r") as f:
    traces = json.load(f)

print(f"Total calls: {traces['total_calls']}")
for call in traces['calls']:
    print(f"{call['function']}: {call['duration_ms']:.2f}ms")
```

## Troubleshooting

### Logs not appearing in console?
Check log level is not too high:
```python
logger = create_daf_logger(name="test", verbose=True)  # DEBUG level
```

### File logs not being created?
Ensure log directory is writable:
```python
import os
log_dir = Path("./logs")
log_dir.mkdir(parents=True, exist_ok=True)
assert os.access(log_dir, os.W_OK), "Log directory not writable"
```

### Performance metrics not tracking?
Use context manager for proper tracking:
```python
# ✓ Correct - uses context manager
with logger.track_operation("my_op"):
    pass

# ✗ Incorrect - no tracking
start = time.time()
# code
end = time.time()
```

### Log files too large?
Rotating handler is configured at 10MB with 5 backups. Old files are automatically compressed if space needed.

## See Also

- [OUTPUT_ORGANIZATION.md](./OUTPUT_ORGANIZATION.md) - Output folder structure
- [RUNNING_TESTS.md](./RUNNING_TESTS.md) - Test execution and logging
- [DAF README](../README.md) - Module overview

