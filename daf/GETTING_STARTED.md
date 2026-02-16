# DAF Getting Started Guide

Welcome to DAF (Distributed Agent Framework)! This guide will help you get started with the new output management and logging features.

## 🚀 30-Second Quick Start

```bash
# Run all tests with organized output
cd /Users/4d/Documents/GitHub/cogames
./daf/tests/run_daf_tests.sh

# Generate report
python daf/src/generate_test_report.py \
    daf_output/evaluations/tests \
    daf_output/reports/test_report.json \
    daf_output/reports/test_report.html

# View results
open daf_output/reports/test_report.html
```

**Result**: All test outputs organized in `daf_output/`, report shows pass rate, test times, and categories.

---

## 📚 What's New in DAF v2?

DAF now includes powerful tools for managing outputs and logging:

### 1. **Organized Output Structure** 📁
All DAF operations (sweeps, comparisons, training, etc.) automatically save to organized subfolders.

```
daf_output/
├── sweeps/          # Hyperparameter sweep results
├── comparisons/     # Policy comparisons
├── training/        # Training runs
├── evaluations/     # Tests & evaluations
├── logs/            # Session logs
└── reports/         # Generated reports
```

### 2. **Structured Logging** 📝
Track operations with automatic timing and metrics collection.

```python
with logger.track_operation("training"):
    # Your code
    pass
logger.print_metrics_summary()
```

### 3. **Unified Test Runner** ✅
Execute all tests with organized outputs and HTML reports.

```bash
python daf/src/generate_test_report.py daf_output/evaluations/tests
```

### 4. **Session Management** 🔍
Every operation gets a unique session ID for reproducibility.

```
sweeps/20241121_143022/  # YYYYMMDD_HHMMSS format
comparisons/20241121_150045/
training/20241121_151230/
```

---

## 🎯 Common Tasks

### Task 1: Run Tests and Generate Report

**Goal**: Execute all tests and create an HTML report with statistics.

**Steps**:
```bash
# 1. Run tests
./daf/tests/run_daf_tests.sh

# 2. Generate report
python daf/src/generate_test_report.py \
    daf_output/evaluations/tests \
    daf_output/reports/test_report.json \
    daf_output/reports/test_report.html

# 3. View report
open daf_output/reports/test_report.html
```

**Output**:
- ✅ Test outputs organized by category
- ✅ JSON report with statistics
- ✅ Interactive HTML report with tables and metrics

---

### Task 2: Log a Hyperparameter Sweep

**Goal**: Run a sweep operation with organized output and logging.

**Code**:
```python
from daf.output_manager import get_output_manager
from daf.logging_config import create_daf_logger
from daf import sweeps

# Setup
output_mgr = get_output_manager(base_dir="./daf_output")
logger = create_daf_logger("lstm_sweep", verbose=True)

# Track operation
with logger.track_operation("sweep", metadata={
    "policy": "lstm",
    "missions": ["training_facility_1"],
    "num_trials": 10,
}):
    output_mgr.log_operation_start("sweep")
    logger.info("Loading sweep configuration...")
    
    config = DAFSweepConfig.from_yaml("sweep_config.yaml")
    results = sweeps.daf_launch_sweep(config)
    
    logger.info(f"Best score: {results.best_score}")
    
    # Save results
    output_mgr.save_json_results(
        results.to_dict(),
        operation="sweep",
        filename="results"
    )
    
    output_mgr.log_operation_complete("sweep", status="success")

# Summary
logger.print_metrics_summary()
output_mgr.save_session_metadata()
```

**Output**:
- ✅ Organized in `daf_output/sweeps/YYYYMMDD_HHMMSS/`
- ✅ Results saved to JSON
- ✅ Logs in `daf_output/logs/`
- ✅ Metrics printed to console

---

### Task 3: Compare Multiple Policies

**Goal**: Compare policies with logging and organized output.

**Code**:
```python
from daf.output_manager import get_output_manager
from daf import comparison

output_mgr = get_output_manager()

with output_mgr.logger.track_operation("comparison", metadata={
    "policies": ["lstm", "baseline", "random"],
    "missions": ["training_facility_1"],
}):
    output_mgr.log_operation_start("comparison")
    
    results = comparison.daf_compare_policies(
        policies=["lstm", "baseline", "random"],
        missions=["training_facility_1"],
        episodes_per_mission=10,
    )
    
    # Save results
    output_mgr.save_json_results(
        results.to_dict(),
        operation="comparison",
        filename="comparison_results"
    )
    
    output_mgr.log_operation_complete("comparison", status="success")

# View HTML report
visualization.daf_export_comparison_html(
    results,
    output_path=str(output_mgr.dirs.visualizations / "comparison.html")
)
```

---

### Task 4: Find and Explore Previous Results

**Goal**: Locate and examine results from previous operations.

**Code**:
```python
from daf.output_utils import (
    find_latest_session,
    list_sessions,
    print_session_info,
)

# Find latest sweep
latest = find_latest_session("./daf_output", "sweep")
print(f"Latest sweep: {latest}")

# List all recent sessions
sessions = list_sessions("./daf_output", max_results=5)
for session in sessions:
    print(f"{session['id']}: {session['operation']} ({session['created']})")

# Print detailed info
print_session_info(latest)
```

---

### Task 5: Archive and Backup Results

**Goal**: Export a session for backup or sharing.

**Code**:
```python
from daf.output_utils import export_session

# Create compressed archive
export_session(
    session_dir="./daf_output/sweeps/20241121_143022",
    export_path="./backups/sweep_backup.tar.gz",
    compress=True
)
```

---

## 📖 Documentation

### Quick References

| Need | Document | Read Time |
|------|----------|-----------|
| **Where do outputs go?** | [OUTPUT_ORGANIZATION.md](./docs/OUTPUT_ORGANIZATION.md) | 10 min |
| **How to set up logging?** | [LOGGING.md](./docs/LOGGING.md) | 8 min |
| **What's new in v2?** | [IMPROVEMENTS_SUMMARY.md](./IMPROVEMENTS_SUMMARY.md) | 15 min |
| **Full index** | [OUTPUT_AND_LOGGING_INDEX.md](./docs/OUTPUT_AND_LOGGING_INDEX.md) | 5 min |
| **Module overview** | [README.md](./README.md) | 12 min |

### Detailed Guides

1. **[OUTPUT_ORGANIZATION.md](./docs/OUTPUT_ORGANIZATION.md)**
   - Directory structure (9 operation types)
   - Session organization
   - Output files and formats
   - OutputManager & TestRunner APIs
   - Best practices
   - Cleanup and migration

2. **[LOGGING.md](./docs/LOGGING.md)**
   - DAFLogger usage
   - Operation tracking
   - Metrics collection
   - Console output formatting
   - Configuration
   - Troubleshooting

3. **[README.md](./README.md)**
   - Feature overview
   - Quick start
   - Configuration examples
   - Testing guide
   - Best practices

---

## 🔧 API Quick Reference

### OutputManager

```python
from daf.output_manager import get_output_manager

output_mgr = get_output_manager(base_dir="./daf_output", verbose=True)

# Log operations
output_mgr.log_operation_start("sweep", details={"trials": 10})
output_mgr.log_operation_complete("sweep", status="success")

# Save results
output_mgr.save_json_results(data, operation="sweep", filename="results")
output_mgr.save_summary_report(operation="sweep", summary={...})

# Session management
output_mgr.save_session_metadata()
output_mgr.cleanup_temp_files()
```

### DAFLogger

```python
from daf.logging_config import create_daf_logger

logger = create_daf_logger("experiment", verbose=True)

# Track operations
with logger.track_operation("training"):
    logger.info("Training...")

# Display
logger.print_section("Title", level=1)
logger.print_table("Results", data)
logger.print_metrics_summary()

# Save
logger.save_metrics_json(Path("./metrics.json"))
```

### Utilities

```python
from daf.output_utils import (
    find_latest_session,
    list_sessions,
    print_session_info,
    export_session,
    cleanup_old_sessions,
)

latest = find_latest_session("./daf_output", "sweep")
sessions = list_sessions("./daf_output")
print_session_info(latest)
export_session(latest, "./backup.tar.gz")
cleanup_old_sessions("./daf_output", days=30)
```

---

## 📁 Output Directory Guide

After running operations, here's what you'll find:

```
daf_output/
├── sweeps/
│   └── 20241121_143022/           # Session ID (YYYYMMDD_HHMMSS)
│       ├── summary_report.json    # Operation summary
│       └── sweep_results/         # Sweep-specific results
│
├── comparisons/
│   └── 20241121_150045/
│       ├── summary_report.json
│       └── statistical_tests.json
│
├── evaluations/tests/
│   ├── cogames/                   # CoGames framework tests
│   │   ├── cli_output.txt
│   │   ├── core_output.txt
│   │   └── ...
│   ├── daf/                       # DAF module tests
│   │   ├── config_output.txt
│   │   ├── sweeps_output.txt
│   │   └── ...
│   └── test_report.json           # Aggregated results
│
├── logs/
│   ├── daf_20241121_143022.log    # Session log (rotating)
│   └── session_20241121_143022.json # Session metadata
│
├── reports/
│   ├── test_report.html           # Interactive HTML report
│   └── test_report.json           # JSON report
│
└── TEST_RUN_SUMMARY.txt           # Quick reference
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# Enable verbose logging
export DAF_VERBOSE=1

# Disable file logging
export DAF_LOG_TO_FILE=0

# Set output directory
export DAF_OUTPUT_DIR=/custom/path
```

### Python Configuration

```python
from daf.config import DAFConfig

config = DAFConfig(
    output_dir="./daf_output",
    organize_by_operation=True,    # Use structured folders
    verbose=False,                  # Logging level
    log_to_file=True,              # Save to file
    max_parallel_jobs=4,           # Parallel execution
)
```

---

## 🐛 Troubleshooting

### Issue: Outputs not being saved

**Solution**: Ensure output manager is initialized before saving:
```python
output_mgr = get_output_manager()  # Initialize first
output_mgr.save_json_results(...)  # Then save
```

### Issue: Logs not appearing in console

**Solution**: Enable verbose mode:
```python
logger = create_daf_logger("test", verbose=True)  # DEBUG level
```

### Issue: Test report not generating

**Solution**: Check that test outputs exist:
```bash
ls daf_output/evaluations/tests/
python daf/src/generate_test_report.py daf_output/evaluations/tests
```

### Issue: Finding previous results

**Solution**: Use utilities to locate sessions:
```python
from daf.output_utils import list_sessions
sessions = list_sessions("./daf_output")
```

See [OUTPUT_AND_LOGGING_INDEX.md](./docs/OUTPUT_AND_LOGGING_INDEX.md#-troubleshooting) for more troubleshooting.

---

## 💡 Best Practices

### 1. Always Initialize Output Manager First

```python
output_mgr = get_output_manager()  # Early in your script
```

### 2. Use Context Managers for Operations

```python
with logger.track_operation("training"):
    # Your code
    pass
```

### 3. Log Operation Lifecycle

```python
output_mgr.log_operation_start("sweep")
# ... perform sweep ...
output_mgr.log_operation_complete("sweep", status="success")
```

### 4. Save Session Metadata

```python
output_mgr.save_session_metadata()  # At the end
```

### 5. Generate Reports Immediately

```bash
python daf/src/generate_test_report.py daf_output/evaluations/tests
```

### 6. Clean Up Old Sessions Periodically

```python
cleanup_old_sessions("./daf_output", days=30)
```

---

## 📊 Example Workflow

Complete example of a typical DAF workflow:

```python
#!/usr/bin/env python3
"""Example: Complete DAF workflow with logging and output management."""

from pathlib import Path
from daf.config import DAFSweepConfig
from daf.output_manager import get_output_manager
from daf.logging_config import create_daf_logger
from daf import sweeps

# 1. Setup
output_mgr = get_output_manager(base_dir="./daf_output", verbose=True)
logger = create_daf_logger("lstm_sweep_workflow")

# 2. Print output structure
output_mgr.print_output_info()

# 3. Track and execute operation
with logger.track_operation("sweep", metadata={
    "policy": "lstm",
    "missions": ["training_facility_1"],
    "num_trials": 10,
}):
    logger.print_section("Hyperparameter Sweep", level=1)
    output_mgr.log_operation_start("sweep")
    
    # Load configuration
    logger.print_section("Loading Configuration", level=2)
    config = DAFSweepConfig.from_yaml("daf/examples/sweep_config.yaml")
    logger.debug(f"Config: {config.name}")
    
    # Run sweep
    logger.print_section("Running Sweep", level=2)
    results = sweeps.daf_launch_sweep(config)
    logger.info(f"Best score: {results.best_score}")
    
    # Save results
    logger.print_section("Saving Results", level=2)
    output_mgr.save_json_results(
        results.to_dict(),
        operation="sweep",
        filename="results"
    )
    
    output_mgr.log_operation_complete("sweep", status="success")

# 4. Display metrics
logger.print_section("Summary", level=1)
logger.print_metrics_summary()

# 5. Save session
output_mgr.save_session_metadata()

print("✓ Workflow complete!")
```

---

## 🎓 Learning Path

1. **Start Here** (5 min)
   - Read this file (you are here!)
   - Run quick start commands

2. **Try Examples** (10 min)
   - Run `output_management_example.py`
   - Explore generated outputs

3. **Understand Output Structure** (10 min)
   - Read [OUTPUT_ORGANIZATION.md](./docs/OUTPUT_ORGANIZATION.md)
   - Review directory structure

4. **Set Up Logging** (10 min)
   - Read [LOGGING.md](./docs/LOGGING.md)
   - Try logging examples

5. **Integrate into Your Code** (30 min)
   - Update your DAF operations
   - Test output organization
   - Generate reports

6. **Advanced Usage** (Optional)
   - Session management with utilities
   - Report generation and analysis
   - Large-scale experiment tracking

---

## 📞 Support

### Documentation
- **Output Organization**: [OUTPUT_ORGANIZATION.md](./docs/OUTPUT_ORGANIZATION.md)
- **Logging**: [LOGGING.md](./docs/LOGGING.md)
- **Complete Index**: [OUTPUT_AND_LOGGING_INDEX.md](./docs/OUTPUT_AND_LOGGING_INDEX.md)
- **Module README**: [README.md](./README.md)

### Examples
- Run `python daf/examples/output_management_example.py`
- Check `daf/examples/sweep_config.yaml`
- Review `daf/tests/run_daf_tests.sh`

### Issues
1. Check [OUTPUT_AND_LOGGING_INDEX.md - Troubleshooting](./docs/OUTPUT_AND_LOGGING_INDEX.md#-troubleshooting)
2. Review logs in `daf_output/logs/`
3. Check documentation above

---

## ✅ Next Steps

1. **Run the quick start** (2 min)
   ```bash
   ./daf/tests/run_daf_tests.sh
   ```

2. **Generate a report** (1 min)
   ```bash
   python daf/src/generate_test_report.py daf_output/evaluations/tests
   ```

3. **Explore outputs** (5 min)
   ```bash
   ls -la daf_output/
   cat daf_output/TEST_RUN_SUMMARY.txt
   open daf_output/reports/test_report.html
   ```

4. **Try an example** (5 min)
   ```bash
   python daf/examples/output_management_example.py
   ```

5. **Read documentation** (15 min)
   - Start with [OUTPUT_ORGANIZATION.md](./docs/OUTPUT_ORGANIZATION.md)
   - Then read [LOGGING.md](./docs/LOGGING.md)

---

**Welcome to DAF v2!** 🚀

Get started with the quick start above, and explore the comprehensive documentation links for deeper learning.







