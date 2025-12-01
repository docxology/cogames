# DAF Output Management & Logging - Complete Index

Quick reference for the comprehensive output management and logging improvements in DAF.

## 📋 Quick Reference

| Need | Resource | Location |
|------|----------|----------|
| **How to run tests?** | Quick Start | [DAF README - Quick Start](#quick-start) |
| **Where do outputs go?** | Directory Structure | [OUTPUT_ORGANIZATION.md](./OUTPUT_ORGANIZATION.md) |
| **How to log operations?** | Logging Guide | [LOGGING.md](./LOGGING.md) |
| **How to organize results?** | API Reference | [OUTPUT_ORGANIZATION.md - Usage](./OUTPUT_ORGANIZATION.md#usage) |
| **Test execution?** | TestRunner Guide | [OUTPUT_ORGANIZATION.md - TestRunner](./OUTPUT_ORGANIZATION.md#testrunner-api) |
| **Generate reports?** | Report Guide | [OUTPUT_ORGANIZATION.md - Reports](./OUTPUT_ORGANIZATION.md#generate-test-reports) |
| **Examples?** | Working Code | [output_management_example.py](../examples/output_management_example.py) |

## 📚 Documentation Files

### Core Documentation

#### 1. **OUTPUT_ORGANIZATION.md** 📁
Complete guide to DAF's organized output structure.

**Sections**:
- Directory structure (9 operation types)
- Session organization (YYYYMMDD_HHMMSS)
- Output files (summary reports, test outputs, logs)
- Usage API (OutputManager, TestRunner, Report Generation)
- Best practices
- Cleanup and migration guides

**Start Here**: To understand where outputs go.

**Reference Link**: [OUTPUT_ORGANIZATION.md](./OUTPUT_ORGANIZATION.md)

---

#### 2. **LOGGING.md** 📝
Comprehensive logging configuration and usage guide.

**Sections**:
- DAFLogger usage
- OperationTracker
- OutputManager integration
- Log file organization
- Configuration and environment variables
- Log levels and structured logging
- Best practices and troubleshooting

**Start Here**: To set up logging in your code.

**Reference Link**: [LOGGING.md](./LOGGING.md)

---

#### 3. **DAF README.md** 📖
Updated module overview with new features highlighted.

**Key Sections**:
- Feature overview (10 features, 4 new)
- Quick start guide
- Module overview table
- Configuration examples
- Example commands
- Testing infrastructure
- Best practices
- Troubleshooting

**Start Here**: For general DAF overview and quick examples.

**Reference Link**: [../README.md](../README.md)

---

#### 4. **IMPROVEMENTS_SUMMARY.md** ✨
Summary of all improvements with implementation details.

**Sections**:
- Test run review (18 suites, 100+ tests)
- 8 improvements implemented
- Key improvements summary
- Integration points
- Testing coverage
- Backward compatibility
- Migration guide
- Quality metrics

**Start Here**: To see what's new and why it matters.

**Reference Link**: [../IMPROVEMENTS_SUMMARY.md](../IMPROVEMENTS_SUMMARY.md)

---

### Examples

#### 5. **output_management_example.py** 💻
Working examples demonstrating all new features.

**Examples Included**:
- OutputManager usage
- DAFLogger with tracking
- TestRunner execution
- Directory structure visualization

**How to Run**:
```bash
python daf/examples/output_management_example.py
```

**Reference Link**: [../examples/output_management_example.py](../examples/output_management_example.py)

---

## 🏗️ New Modules

### Core Modules

| Module | Purpose | Key Classes | Location |
|--------|---------|-------------|----------|
| **output_manager** | Unified output organization | OutputManager, OutputDirectories | `daf/src/output_manager.py` |
| **logging_config** | Structured logging | DAFLogger, OperationTracker, OperationMetrics | `daf/src/logging_config.py` |
| **test_runner** | Test execution framework | TestRunner, TestResult | `daf/src/test_runner.py` |
| **generate_test_report** | Report generation | TestReportGenerator | `daf/src/generate_test_report.py` |

---

## 🚀 Quick Start

### 1. Run Tests with Organized Output

```bash
cd /Users/4d/Documents/GitHub/cogames
./daf/tests/run_daf_tests.sh
```

**What Happens**:
- Tests organized by category (cogames/, daf/)
- Outputs saved to `daf_output/evaluations/tests/`
- Summary generated in `TEST_RUN_SUMMARY.txt`

---

### 2. Generate Test Report

```bash
python daf/src/generate_test_report.py \
    daf_output/evaluations/tests \
    daf_output/reports/test_report.json \
    daf_output/reports/test_report.html

# View HTML report
open daf_output/reports/test_report.html
```

---

### 3. Use Output Manager

```python
from daf.output_manager import get_output_manager

# Initialize
output_mgr = get_output_manager(base_dir="./daf_output")

# Log operation
output_mgr.log_operation_start("sweep", details={"trials": 10})

# ... perform sweep ...

# Save results
output_mgr.save_json_results(
    {"best_score": 0.92},
    operation="sweep",
    filename="results"
)

# Complete operation
output_mgr.log_operation_complete("sweep", status="success")

# Save metadata
output_mgr.save_session_metadata()
```

---

### 4. Use DAFLogger

```python
from daf.logging_config import create_daf_logger

# Create logger
logger = create_daf_logger("my_experiment", verbose=True)

# Track operations
with logger.track_operation("training", metadata={"epochs": 10}):
    logger.info("Starting training...")
    # Your code here

# Print summary
logger.print_metrics_summary()
logger.save_metrics_json(Path("./metrics.json"))
```

---

## 📁 Output Directory Structure

```
daf_output/                          # Root directory
├── sweeps/                          # Hyperparameter sweep results
│   └── YYYYMMDD_HHMMSS/
├── comparisons/                     # Policy comparison results
│   └── YYYYMMDD_HHMMSS/
├── training/                        # Training runs
│   └── YYYYMMDD_HHMMSS/
├── deployment/                      # Deployment packages
│   └── YYYYMMDD_HHMMSS/
├── evaluations/                     # Test & evaluation results
│   ├── tests/
│   │   ├── cogames/                 # CoGames test outputs
│   │   ├── daf/                     # DAF test outputs
│   │   └── test_report.json
│   └── YYYYMMDD_HHMMSS/
├── visualizations/                  # Generated plots/HTML
│   └── YYYYMMDD_HHMMSS/
├── logs/                            # Session logs
│   ├── daf_YYYYMMDD_HHMMSS.log
│   └── session_YYYYMMDD_HHMMSS.json
├── reports/                         # Generated reports
│   ├── test_report.json
│   └── test_report.html
├── artifacts/                       # Reusable artifacts
└── .temp/                           # Temporary files
```

**Key Point**: Each operation organized by type, then by session ID.

For detailed structure: [OUTPUT_ORGANIZATION.md - Directory Structure](./OUTPUT_ORGANIZATION.md#directory-structure)

---

## 🔍 API Reference

### OutputManager

```python
# Get global instance
output_mgr = get_output_manager(base_dir="./daf_output", verbose=True)

# Operations
output_mgr.get_operation_dir(operation, subdir=None)
output_mgr.log_operation_start(operation, details={})
output_mgr.log_operation_complete(operation, status, details={})

# Save results
output_mgr.save_json_results(data, operation, filename, subdir=None)
output_mgr.save_text_results(text, operation, filename, subdir=None)
output_mgr.save_summary_report(operation, summary, subdir=None)
output_mgr.save_session_metadata()

# Utilities
output_mgr.print_output_info()
output_mgr.cleanup_temp_files()
```

Full reference: [OUTPUT_ORGANIZATION.md - Usage](./OUTPUT_ORGANIZATION.md#usage)

---

### DAFLogger

```python
# Create logger
logger = create_daf_logger(name, log_dir, verbose)

# Logging
logger.info(msg)
logger.debug(msg)
logger.warning(msg)
logger.error(msg)

# Operations
with logger.track_operation(name, metadata={}):
    pass

# Output
logger.print_section(title, level)
logger.print_table(title, data, columns)
logger.print_metrics_summary()

# Save
logger.save_metrics_json(output_path)
```

Full reference: [LOGGING.md - DAFLogger](./LOGGING.md#daflogger)

---

### TestRunner

```python
# Create runner
runner = TestRunner(output_base_dir, verbose)

# Execute
runner.run_test_suite(test_path, category, suite_name)
runner.run_test_batch(tests)  # [(path, category, name), ...]

# Save
runner.save_test_outputs()
runner.save_test_report()

# Display
runner.print_test_summary()
runner.print_failed_tests()

# Cleanup
runner.cleanup()
```

Full reference: [OUTPUT_ORGANIZATION.md - TestRunner API](./OUTPUT_ORGANIZATION.md#testrunner-api)

---

### TestReportGenerator

```python
# Create generator
generator = TestReportGenerator(test_output_dir)

# Generate
results = generator.collect_all_results()
summary = generator.generate_summary(results)

# Display
generator.print_summary(summary)

# Save
generator.save_summary(summary, output_path)
generator.generate_html_report(summary, output_path)

# Standalone
generate_report_from_outputs(test_dir, output_json, output_html)
```

---

## 📋 Common Tasks

### Task: Run All Tests and Generate Report

```bash
# Run tests with organized output
./daf/tests/run_daf_tests.sh

# Generate report from outputs
python daf/src/generate_test_report.py \
    daf_output/evaluations/tests \
    daf_output/reports/test_report.json \
    daf_output/reports/test_report.html

# View report
open daf_output/reports/test_report.html
```

---

### Task: Log a Sweep Operation

```python
from daf.output_manager import get_output_manager
from daf.logging_config import create_daf_logger

output_mgr = get_output_manager()
logger = create_daf_logger("sweep_experiment")

with logger.track_operation("sweep", metadata={
    "policy": "lstm",
    "missions": ["training_facility_1"],
    "num_trials": 10,
}):
    output_mgr.log_operation_start("sweep")
    
    # Run sweep
    results = daf_launch_sweep(config)
    
    output_mgr.save_json_results(
        results.to_dict(),
        operation="sweep",
        filename="results"
    )
    output_mgr.log_operation_complete("sweep", status="success")

logger.print_metrics_summary()
output_mgr.save_session_metadata()
```

---

### Task: Save Comparison Results

```python
from daf.output_manager import get_output_manager

output_mgr = get_output_manager()

# Save detailed results
output_mgr.save_json_results(
    comparison_results.to_dict(),
    operation="comparison",
    filename="detailed_results"
)

# Save summary
output_mgr.save_summary_report(
    operation="comparison",
    summary={
        "num_policies": 3,
        "num_missions": 5,
        "best_policy": "lstm",
        "best_score": 0.92,
    }
)
```

---

### Task: Find Latest Session Results

```bash
# Show most recent session
ls -t daf_output/sweeps/ | head -1

# View session metadata
cat daf_output/logs/session_YYYYMMDD_HHMMSS.json | jq

# View results
cat "daf_output/sweeps/YYYYMMDD_HHMMSS/summary_report.json" | jq
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
export DAF_OUTPUT_DIR=/path/to/output
```

---

### Python Configuration

```python
from daf.config import DAFConfig

config = DAFConfig(
    output_dir="./daf_output",
    organize_by_operation=True,       # NEW
    verbose=False,
    log_to_file=True,
    max_parallel_jobs=4,
)
```

---

## 🐛 Troubleshooting

### Outputs not appearing?
- Check `organize_by_operation=True` in DAFConfig
- Verify output manager initialized before saving
- Check file permissions on output directory

**Solution**: [OUTPUT_ORGANIZATION.md - Troubleshooting](./OUTPUT_ORGANIZATION.md#cleaning-up-old-results)

---

### Logs not showing?
- Enable verbose: `create_daf_logger(verbose=True)`
- Check log file exists in `daf_output/logs/`
- Verify log directory is writable

**Solution**: [LOGGING.md - Troubleshooting](./LOGGING.md#troubleshooting)

---

### Test outputs not captured?
- Ensure test script uses output redirection: `2>&1 | tee file.txt`
- Check `daf_output/evaluations/tests/` folder created
- Verify pytest running correctly

**Solution**: Run `./daf/tests/run_daf_tests.sh` to see correct setup

---

## 📞 Support

For issues or questions:

1. **Check documentation**: Start with appropriate .md file above
2. **See examples**: Run `output_management_example.py`
3. **Review code**: Look at new module source code with inline docs
4. **Check logs**: Review `daf_output/logs/daf_*.log` for errors

---

## 📌 Key Files Summary

| File | Lines | Purpose |
|------|-------|---------|
| `output_manager.py` | ~240 | Core output management |
| `logging_config.py` | ~380 | Structured logging system |
| `test_runner.py` | ~310 | Unified test execution |
| `generate_test_report.py` | ~300 | Report generation |
| `OUTPUT_ORGANIZATION.md` | ~380 | Output structure guide |
| `LOGGING.md` | ~350 | Logging guide |
| `README.md` | ~350 | Module overview |
| `IMPROVEMENTS_SUMMARY.md` | ~400 | What's new summary |

**Total**: ~2,750 lines of code + documentation

---

## ✅ Checklist

- [x] All modules implemented
- [x] All tests passing
- [x] All documentation complete
- [x] Examples working
- [x] Backward compatible
- [x] Zero linting errors
- [x] Type hints complete
- [x] Ready for production

---

## 🎯 Next Steps

1. **Try it out**: Run `output_management_example.py`
2. **Run tests**: Execute `./daf/tests/run_daf_tests.sh`
3. **Generate report**: Use `generate_test_report.py`
4. **Integrate**: Update your DAF operations to use new infrastructure
5. **Explore**: Check out the organized outputs in `daf_output/`

---

**Updated**: November 21, 2024
**Status**: Production Ready ✅







