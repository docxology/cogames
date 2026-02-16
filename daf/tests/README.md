# CoGames Test Suite

Comprehensive testing infrastructure for CoGames and the DAF (Distributed Agent Framework) modules.

## Test Structure

### Phase 1: CoGames Core Tests (10 suites)

Tests for the main CoGames functionality:

- **CLI Tests** (`test_cli.py`) - Command-line interface validation
- **Core Game Tests** (`test_cogs_vs_clips.py`) - Core game mechanics
- **CVC Assembler Hearts** (`test_cvc_assembler_hearts.py`) - Specialized mission tests
- **Procedural Maps** (`test_procedural_maps.py`) - Map generation and variants
- **Scripted Policies** (`test_scripted_policies.py`) - Built-in policy validation
- **Train Integration** (`test_train_integration.py`) - Training pipeline
- **Train Vector Alignment** (`test_train_vector_alignment.py`) - Batch alignment
- **All Games Describe** (`test_all_games_describe.py`) - Mission descriptions
- **All Games Eval** (`test_all_games_eval.py`) - Mission evaluation
- **All Games Play** (`test_all_games_play.py`) - Interactive play testing

### Phase 2: DAF Module Tests (8 suites)

Tests for the Distributed Agent Framework:

- **Configuration** (`test_config.py`) - Config loading and validation
- **Environment Checks** (`test_environment_checks.py`) - System validation
- **Sweeps** (`test_sweeps.py`) - Hyperparameter sweep functionality
- **Comparison** (`test_comparison.py`) - Policy comparison analysis
- **Deployment** (`test_deployment.py`) - Model packaging and deployment
- **Distributed Training** (`test_distributed_training.py`) - Multi-machine training
- **Visualization** (`test_visualization.py`) - Report generation
- **Mission Analysis** (`test_mission_analysis.py`) - Per-mission performance breakdown

## Running Tests

### Full Test Suite

Run all tests from project root:

```bash
cd /Users/4d/Documents/GitHub/cogames
./daf/tests/run_daf_tests.sh
```

This runs:
1. All CoGames core tests
2. All DAF module tests
3. Generates comprehensive reports

### Individual Test Suites

Run specific test suites:

```bash
# CoGames tests
uv run pytest tests/test_cli.py -v
uv run pytest tests/test_scripted_policies.py -v

# DAF tests
uv run pytest daf/tests/test_config.py -v
uv run pytest daf/tests/test_sweeps.py -v
```

### Test Markers

Use pytest markers to select test types:

```bash
# Run only fast unit tests
uv run pytest -m unit -v

# Run only integration tests
uv run pytest -m integration -v

# Skip slow tests
uv run pytest -m "not slow" -v

# Run performance benchmarks
uv run pytest -m performance -v
```

### Verbose Output

For detailed information about test execution:

```bash
# Show detailed output
uv run pytest tests/test_cli.py -vv

# Show print statements
uv run pytest tests/test_cli.py -s

# Show both
uv run pytest tests/test_cli.py -vv -s

# Enable logging
uv run pytest tests/test_cli.py --log-cli-level=DEBUG
```

## Test Reports

### Markdown Report

After running the full test suite, view the comprehensive Markdown report:

```bash
cat ./daf_output/TEST_RUN_SUMMARY.md
```

The report includes:
- Executive summary with pass rates
- Per-suite test counts and timing
- Performance analysis
- Detailed results for each suite
- Recommendations based on results

### JSON Report

For programmatic access to test results:

```bash
cat ./daf_output/TEST_RUN_SUMMARY.json
```

Structure:
```json
{
  "generated": "2024-01-15T10:30:45.123456",
  "phase1": [
    {
      "suite": "CLI Tests",
      "status": "PASSED",
      "tests": 6,
      "passed": 6,
      "failed": 0,
      "skipped": 0,
      "duration": 8.51
    }
  ],
  "phase2": [...]
}
```

### Plain Text Summary

Quick reference summary:

```bash
cat ./daf_output/TEST_RUN_SUMMARY.txt
```

### Individual Test Outputs

Detailed output from each test suite:

```bash
ls -la ./daf_output/evaluations/tests/
cat ./daf_output/evaluations/tests/cogames_test_cli_output.txt
cat ./daf_output/evaluations/tests/daf_test_config_output.txt
```

### Log Files

Detailed execution logs:

```bash
ls -la ./daf_output/logs/
tail -f ./daf_output/logs/session_*.log
```

## Test Infrastructure

### Conftest Configuration

`conftest.py` provides:
- Custom markers (`@pytest.mark.slow`, `@pytest.mark.integration`, etc.)
- Logging configuration with file output
- Shared fixtures (timers, directories, execution tracker, etc.)
- Enhanced test reporting
- Performance assertion helpers
- Test execution tracking for debugging

### Report Generation

`generate_test_report.py` creates:
- Markdown report with statistics and analysis
- JSON report for programmatic access
- Performance metrics and timing breakdown
- Pass rate calculations
- Failure summaries and recommendations

### Logging Configuration

`test_logging_config.py` provides:
- Color-coded log output
- Timestamped entries
- Suite execution tracking
- Performance metrics

### Pytest Plugin

`pytest_output_plugin.py` offers:
- Detailed test execution capture
- JSON result output
- Per-test timing data
- Status tracking
- Execution trace logging
- Error message capture for debugging

## Key Features

### Comprehensive Reporting

- **Markdown Output**: Beautiful, readable reports with statistics and test plan integration
- **JSON Export**: Machine-readable format for CI/CD integration
- **Plain Text**: Quick reference summaries
- **Individual Logs**: Full pytest output for each test suite
- **Expected vs Actual**: Test count validation against test plan

### Performance Tracking

- Execution time per test suite
- Performance analysis highlighting slowest tests
- Duration metrics in all reports
- Per-test timing data

### Detailed Logging

- Color-coded output for easy scanning
- Timestamped entries for debugging
- Separate log files per run
- Session-level logging

### Enhanced Diagnostics

- Test markers for categorization
- Performance assertion helpers
- Timing fixtures for measurements
- Environment validation
- Test execution tracking with execution_tracker fixture
- Function call recording for performance analysis
- Error message capture for debugging

## Configuration

### Default Settings

- **Timeout**: 5 minutes per test (configurable)
- **Log Level**: DEBUG for files, INFO for console
- **Output Directory**: `./daf_output/`
- **Test Output**: `./daf_output/evaluations/tests/`
- **Logs**: `./daf_output/logs/`

### Customization

Edit `conftest.py` to:
- Add custom markers
- Adjust logging levels
- Modify fixture behavior
- Change assertion helpers

Edit `run_daf_tests.sh` to:
- Add or remove test suites
- Modify output directories
- Adjust pytest options
- Change report generation

## Logging and Tracing Features

### Function Call Tracking

DAF includes infrastructure for tracking function calls during test execution:

```python
from daf.src.logging_config import create_daf_logger, track_function_calls

# Create logger with call tracking enabled
logger = create_daf_logger("my_test", track_calls=True)

# Use decorator on functions you want to track
@track_function_calls(logger)
def my_function(arg1, arg2):
    return arg1 + arg2

# Save call traces to file
logger.save_call_traces("calls.json")
```

### Test Execution Tracking

Use the `execution_tracker` fixture in your tests:

```python
def test_my_feature(execution_tracker):
    execution_tracker.record_event("setup_complete", {"setup_time": 1.5})
    
    # Test code here
    
    execution_tracker.record_event("test_complete", {"status": "success"})
    summary = execution_tracker.get_summary()
```

### Pytest Plugin Features

The pytest plugin now captures:
- Test execution timing
- Failure reasons and error messages
- Skip reasons for skipped tests
- Execution trace events

All information is automatically saved in JSON format for analysis.

## Troubleshooting

### Tests Won't Run

```bash
# Ensure environment is set up
uv sync --all-extras

# Check Python path
python3 -c "import cogames; print(cogames.__file__)"

# Run with verbose output
./daf/tests/run_daf_tests.sh 2>&1 | head -100
```

### Missing Output Files

```bash
# Create output directories manually
mkdir -p ./daf_output/evaluations/tests
mkdir -p ./daf_output/logs

# Check permissions
ls -la ./daf_output/
```

### Report Generation Fails

```bash
# Run report generation directly
python3 daf/tests/generate_test_report.py ./daf_output/evaluations/tests

# Check Python version (requires 3.8+)
python3 --version
```

### Slow Test Execution

```bash
# Run only fast tests
uv run pytest -m "not slow" -v

# Run tests in parallel
uv run pytest -n auto tests/

# Profile test execution
uv run pytest --durations=10 tests/
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
- name: Run CoGames Tests
  run: |
    cd /Users/4d/Documents/GitHub/cogames
    ./daf/tests/run_daf_tests.sh

- name: Upload Test Reports
  if: always()
  uses: actions/upload-artifact@v3
  with:
    name: test-reports
    path: ./daf_output/

- name: Comment PR with Results
  if: always()
  uses: actions/github-script@v6
  with:
    script: |
      const fs = require('fs');
      const report = fs.readFileSync('./daf_output/TEST_RUN_SUMMARY.md', 'utf8');
      github.rest.issues.createComment({
        issue_number: context.issue.number,
        owner: context.repo.owner,
        repo: context.repo.repo,
        body: report
      });
```

## Best Practices

1. **Run Full Suite Before Committing**
   ```bash
   ./daf/tests/run_daf_tests.sh
   ```

2. **Review Reports for Performance Regressions**
   - Check execution time trends
   - Monitor individual test timing

3. **Use Markers for Test Organization**
   ```python
   @pytest.mark.slow
   @pytest.mark.integration
   def test_distributed_training():
       pass
   ```

4. **Check Logs for Warnings**
   - Review logs even when tests pass
   - Catch deprecation warnings early

5. **Archive Reports**
   - Save reports from release tests
   - Compare performance over time

## Performance Benchmarks

Typical execution times on modern hardware:

| Phase | Suites | Tests | Duration |
|-------|--------|-------|----------|
| Phase 1 | 10 | 130+ | ~3 minutes |
| Phase 2 | 8 | 95+ | ~2 minutes |
| **Total** | **18** | **225+** | **~5 minutes** |

Actual times vary based on:
- Hardware capabilities
- System load
- Network latency (for distributed tests)
- Mission complexity

## See Also

- [CoGames README](../../README.md) - Main project documentation
- [DAF Documentation](../README.md) - Distributed Agent Framework
- [CoGames AGENTS.md](../../AGENTS.md) - Agent policy architecture
- [Testing Guide](../docs/RUNNING_TESTS.md) - Additional testing info
