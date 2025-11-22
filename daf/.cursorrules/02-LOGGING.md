# DAF Logging Cursorrules

## DAFLogger API

Use `DAFLogger` for all structured logging with metrics tracking.

### Initialization

```python
from daf.logging_config import create_daf_logger
from pathlib import Path

# Create logger
logger = create_daf_logger(
    name="operation_name",
    log_dir=Path("./daf_output/logs"),
    verbose=False  # Set to True for DEBUG level
)
```

### Core Methods

#### Basic Logging

```python
logger.info("Information message")
logger.debug("Debug diagnostic info")
logger.warning("Warning message")
logger.error("Error message")
```

#### Operation Tracking

```python
# Context manager (recommended)
with logger.track_operation("operation_name", metadata={"key": "value"}):
    logger.info("Doing work...")
    # Automatic timing and error tracking
    logger.debug("Progress...")

# Manual tracking
metrics = logger.start_operation("operation_name", details={...})
try:
    logger.info("Performing operation...")
except Exception as e:
    logger.end_operation(metrics, status="error")
    raise
```

#### Console Formatting

```python
# Print formatted sections
logger.print_section("Main Title", level=1)       # Bold centered
logger.print_section("Subtitle", level=2)        # Underlined
logger.print_section("Item", level=3)            # Bullet

# Print formatted table
data = [
    {"Model": "LSTM", "Accuracy": 0.92},
    {"Model": "Baseline", "Accuracy": 0.85},
]
logger.print_table("Model Comparison", data)

# Print metrics summary
logger.print_metrics_summary()
```

#### Saving Metrics

```python
# Save metrics to JSON
metrics_file = logger.save_metrics_json(Path("./output/metrics.json"))
```

## Logging Patterns

### Pattern 1: Simple Operation

```python
logger = create_daf_logger("simple_op")

with logger.track_operation("task"):
    logger.info("Starting task...")
    # Your code
    logger.debug("Task complete")

logger.print_metrics_summary()
```

### Pattern 2: Complex Operation with Sub-Tasks

```python
logger = create_daf_logger("complex_op")

with logger.track_operation("main_task", metadata={"stages": 3}):
    
    # Sub-task 1
    with logger.track_operation("stage_1"):
        logger.info("Stage 1 running...")
        # Code
    
    # Sub-task 2
    with logger.track_operation("stage_2"):
        logger.info("Stage 2 running...")
        # Code
    
    # Sub-task 3
    with logger.track_operation("stage_3"):
        logger.info("Stage 3 running...")
        # Code

logger.print_metrics_summary()
logger.save_metrics_json(Path("./metrics.json"))
```

### Pattern 3: Error Handling with Logging

```python
logger = create_daf_logger("error_handling_op")

try:
    with logger.track_operation("risky_task"):
        logger.info("Attempting risky operation...")
        # Code that might fail
except Exception as e:
    logger.error(f"Operation failed: {e}")
    # Logger automatically marks operation as failed
    raise
```

### Pattern 4: Sweeping with Logging

```python
logger = create_daf_logger("sweep_logging")

with logger.track_operation("sweep", metadata={
    "num_trials": 10,
    "policy": "lstm",
}):
    logger.print_section("Hyperparameter Sweep", level=1)
    
    for i in range(10):
        logger.debug(f"Trial {i+1}/10")
        # Run trial
    
    logger.info("Sweep complete")

logger.print_section("Results", level=2)
logger.print_metrics_summary()
```

## Log Levels

### When to Use Each Level

| Level | Use Case | Example |
|-------|----------|---------|
| **DEBUG** | Detailed diagnostic info | `logger.debug("Variable x = 42")` |
| **INFO** | Major milestones | `logger.info("Training started")` |
| **WARNING** | Potential issues | `logger.warning("Low memory")` |
| **ERROR** | Errors and failures | `logger.error("Failed to load")` |

### Level Selection by Context

```python
# INFO: Major operation milestones
logger.info("Starting training...")
logger.info("Training complete")
logger.info(f"Best score: {score}")

# DEBUG: Detailed progress during operations
logger.debug(f"Epoch {epoch}/100 complete")
logger.debug(f"Loss: {loss:.4f}")
logger.debug(f"Sample shape: {sample.shape}")

# WARNING: Potentially problematic conditions
logger.warning("Learning rate very high: 1.0")
logger.warning("Only 1 GPU available")
logger.warning("Memory usage high: 90%")

# ERROR: Actual failures
logger.error(f"Failed to load checkpoint: {e}")
logger.error(f"Model inference failed: {e}")
logger.error(f"Invalid configuration: {e}")
```

## Metrics Tracking

### OperationMetrics Dataclass

Each operation is automatically tracked:

```python
@dataclass
class OperationMetrics:
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"
    error_message: Optional[str] = None
    metadata: dict[str, Any] = None
    
    @property
    def duration_seconds(self) -> float:
        """Get operation duration."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
```

### Tracking Multiple Operations

```python
tracker = OperationTracker()

with tracker.track("load_data"):
    # Load data
    pass

with tracker.track("train_model"):
    # Train model
    pass

summary = tracker.get_summary()
# {
#   "total_operations": 2,
#   "successful": 2,
#   "failed": 0,
#   "total_duration_seconds": 10.5,
#   "operations": [...]
# }
```

## Console Output

### Section Headers

```python
# Level 1: Bold centered header (main section)
logger.print_section("Training Phase", level=1)
# ════════════════════════════════════════════
# Training Phase
# ════════════════════════════════════════════

# Level 2: Underlined header (subsection)
logger.print_section("Data Loading", level=2)
# Data Loading
# ────────────

# Level 3: Bullet point (item)
logger.print_section("Step 1", level=3)
# • Step 1
```

### Tables

```python
data = [
    {"Trial": 1, "LR": 0.001, "Score": 0.82},
    {"Trial": 2, "LR": 0.01, "Score": 0.89},
    {"Trial": 3, "LR": 0.1, "Score": 0.75},
]

logger.print_table("Trial Results", data)
# ┌───────────────────────────────────────┐
# │ Trial Results                         │
# ├──────┬────────┬───────────────────────┤
# │Trial │ LR     │ Score                 │
# ├──────┼────────┼───────────────────────┤
# │    1 │  0.001 │                  0.82 │
# │    2 │   0.01 │                  0.89 │
# │    3 │    0.1 │                  0.75 │
# └──────┴────────┴───────────────────────┘
```

### Metrics Summary

```python
logger.print_metrics_summary()
# ════════════════════════════════════════════
# Operation Metrics Summary
# ════════════════════════════════════════════
#
# Total Operations............3
# Successful.................3
# Failed.....................0
# Total Time (s)..........125.4
#
# Individual Operations
# ┌─────────────┬────────┬──────────────┐
# │ Operation   │ Status │ Duration (s) │
# ├─────────────┼────────┼──────────────┤
# │ load_data   │ success│         2.1  │
# │ train_model │ success│       120.3  │
# │ save_results│ success│         3.0  │
# └─────────────┴────────┴──────────────┘
```

## Configuration

### DAFLogger Settings

```python
logger = create_daf_logger(
    name="my_operation",           # Logger name
    log_dir=Path("./logs"),        # Log file directory
    verbose=False,                 # DEBUG level if True
)
```

### Environment Variables

```bash
# Enable verbose logging
export DAF_VERBOSE=1

# Disable file logging
export DAF_LOG_TO_FILE=0

# Set log directory
export DAF_LOG_DIR=/custom/logs
```

## Best Practices

### DO ✓

- Use context managers for operation tracking
- Provide meaningful metadata
- Use appropriate log levels
- Track sub-operations separately
- Print formatted sections for clarity
- Save metrics after operations
- Include timings in INFO messages
- Use logger names matching operations

### DON'T ✗

- Use print() instead of logger
- Mix log levels (WARNING for progress)
- Forget context manager `with` statement
- Ignore metric collection
- Skip metadata for operations
- Leave temporary files
- Log sensitive information
- Use very long debug messages

## Integration with OutputManager

### Combined Usage

```python
from daf.output_manager import get_output_manager
from daf.logging_config import create_daf_logger

output_mgr = get_output_manager()
logger = create_daf_logger("operation")

# Use together
with logger.track_operation("training"):
    output_mgr.log_operation_start("training")
    logger.info("Training started...")
    
    # Your code
    
    output_mgr.save_json_results(results, operation="training")
    output_mgr.log_operation_complete("training", status="success")

logger.print_metrics_summary()
output_mgr.save_session_metadata()
```

## Log Files

### Rotating Handlers

Logs use rotating file handlers:

```
logs/
├── daf_20241121_143022.log      # Current log
├── daf_20241121_143022.log.1    # Previous (if rolled over)
├── daf_20241121_143022.log.2
└── session_20241121_143022.json # Session metadata
```

Configuration:
- **Max Size**: 10 MB per file
- **Backup Count**: 5 old files kept
- **Format**: `[timestamp] logger_name - level - message`

### Reading Logs

```bash
# View live log
tail -f daf_output/logs/daf_*.log

# View session metadata
cat daf_output/logs/session_*.json | jq

# Search logs
grep "ERROR" daf_output/logs/daf_*.log
```

## Troubleshooting

### Logs not appearing in console?
Ensure verbose mode is enabled:
```python
logger = create_daf_logger("test", verbose=True)  # DEBUG level
```

### File logs not created?
Ensure log directory is writable:
```python
import os
log_dir = Path("./logs")
log_dir.mkdir(parents=True, exist_ok=True)
assert os.access(log_dir, os.W_OK)
```

### Metrics not tracking?
Use context manager for proper tracking:
```python
# ✓ Correct
with logger.track_operation("op"):
    pass

# ✗ Wrong (no tracking)
logger.start_operation("op")
# (no context manager)
```

---

**Status**: Production Ready ✅
**Last Updated**: November 21, 2024



