# DAF Tests: Agent & Policy Test Infrastructure

DAF test suite validates policy interfaces, agent integration, and all DAF modules. Tests cover both CoGames policies and custom policy implementations.

## Test Organization

```
daf/tests/
├── test_config.py                   # Configuration management tests
├── test_environment_checks.py       # Environment validation tests
├── test_sweeps.py                   # Hyperparameter sweep tests
├── test_comparison.py               # Policy comparison tests
├── test_distributed_training.py     # Distributed training tests
├── test_deployment.py               # Deployment pipeline tests
├── test_mission_analysis.py         # Mission analysis tests
├── test_visualization.py            # Report generation tests
├── test_orchestrators.py            # Pipeline orchestration tests
├── run_daf_tests.sh                 # Test runner script
├── generate_test_report.py          # Markdown report generator
├── test_logging_config.py           # Logging configuration
├── pytest_output_plugin.py          # Pytest output plugin
├── conftest.py                      # Pytest configuration
├── README.md                        # Test documentation
└── daf_output/                      # Test output directory
    ├── evaluations/
    │   ├── tests/                   # Individual test outputs
    │   └── TEST_RUN_SUMMARY.md      # Markdown report
    └── logs/                        # Execution logs
```

## Key Test Infrastructure Files

### `run_daf_tests.sh`
Main test runner orchestrating all phases:
- Phase 1: 10 CoGames core test suites (200+ tests)
- Phase 2: 8 DAF module test suites (100+ tests)
- Generates comprehensive reports in Markdown and JSON

### `generate_test_report.py`
Python script for report generation:
- Parses pytest output files
- Creates beautiful Markdown reports with statistics
- Generates JSON for CI/CD integration
- Calculates pass rates and performance metrics

### `conftest.py`
Pytest configuration providing:
- Custom test markers (@pytest.mark.slow, @pytest.mark.integration, etc.)
- Shared fixtures (timers, directories, logging)
- Enhanced test reporting hooks
- Performance assertion helpers

### `test_logging_config.py`
Logging infrastructure:
- Color-coded log output
- Timestamped entries
- Suite execution tracking
- Performance metrics collection

### `pytest_output_plugin.py`
Pytest plugin for detailed capture:
- Test execution data capture
- JSON result export
- Per-test timing
- Status tracking

## Test Output & Reporting

### Markdown Report (`TEST_RUN_SUMMARY.md`)
Beautiful, readable report with:
- Executive summary with pass rates
- Per-suite test counts and timing
- Performance analysis (slowest tests)
- Detailed results for each suite
- Recommendations based on results

Example section:
```markdown
### Test Suites

| Status | Suite | Results | Time |
|--------|-------|---------|------|
| ✅ | CLI Tests | 6/6 | 8.56s |
| ✅ | Core Game Tests | 4/4 | 0.21s |
```

### JSON Report (`TEST_RUN_SUMMARY.json`)
Structured format for programmatic access:
```json
{
  "generated": "2025-11-21T06:08:50.366176",
  "phase1": [
    {
      "suite": "CLI Tests",
      "status": "PASSED",
      "tests": 6,
      "passed": 6,
      "duration": 8.56
    }
  ],
  "phase2": [...]
}
```

### Individual Test Outputs
Each test suite produces a detailed output file with full pytest output, timing, and diagnostics.

### Log Files
Session and execution logs saved to `daf_output/logs/` with:
- Color-coded output
- Timestamped entries
- Hierarchical structure

## Running Tests

### Full Test Suite
```bash
cd /Users/4d/Documents/GitHub/cogames
./daf/tests/run_daf_tests.sh
```

### Individual Test Suites
```bash
# CoGames tests
uv run pytest tests/test_cli.py -v
uv run pytest tests/test_scripted_policies.py -v

# DAF tests
uv run pytest daf/tests/test_config.py -v
uv run pytest daf/tests/test_sweeps.py -v
```

### Using Test Markers
```bash
# Run only slow tests
uv run pytest -m slow -v

# Skip slow tests
uv run pytest -m "not slow" -v

# Run only integration tests
uv run pytest -m integration -v
```

### Verbose Output
```bash
# Show detailed output
uv run pytest tests/test_cli.py -vv

# Show print statements
uv run pytest tests/test_cli.py -s

# Enable debug logging
uv run pytest tests/test_cli.py --log-cli-level=DEBUG
```

## Test Reports Access

### View Markdown Report
```bash
cat ./daf_output/TEST_RUN_SUMMARY.md
```

### View JSON Report (programmatic)
```bash
cat ./daf_output/TEST_RUN_SUMMARY.json
python3 -m json.tool ./daf_output/TEST_RUN_SUMMARY.json
```

### View Plain Text Summary
```bash
cat ./daf_output/TEST_RUN_SUMMARY.txt
```

### View Individual Test Output
```bash
ls ./daf_output/evaluations/tests/
cat ./daf_output/evaluations/tests/cogames_test_cli_output.txt
```

### View Logs
```bash
ls ./daf_output/logs/
tail -f ./daf_output/logs/session_*.log
```

## Performance Benchmarks

Typical execution times on modern hardware:

| Phase | Suites | Tests | Duration |
|-------|--------|-------|----------|
| Phase 1 | 10 | 130+ | ~3 minutes |
| Phase 2 | 8 | 100+ | ~2 minutes |
| **Total** | **18** | **230+** | **~5 minutes** |

Actual times vary based on:
- Hardware capabilities
- System load
- Network latency
- Mission complexity

## Integration with CI/CD

### GitHub Actions Example
```yaml
- name: Run CoGames Tests
  run: |
    cd /Users/4d/Documents/GitHub/cogames
    ./daf/tests/run_daf_tests.sh

- name: Upload Test Reports
  uses: actions/upload-artifact@v3
  with:
    name: test-reports
    path: ./daf_output/
```

## Best Practices

1. **Always run full suite before committing**
   ```bash
   ./daf/tests/run_daf_tests.sh
   ```

2. **Review reports for performance trends**
   - Check execution time trends over time
   - Monitor individual test timing

3. **Use markers for test categorization**
   ```python
   @pytest.mark.slow
   @pytest.mark.integration
   def test_expensive_operation():
       pass
   ```

4. **Check logs for warnings**
   - Review logs even when tests pass
   - Catch deprecation warnings early

5. **Archive reports from releases**
   - Save reports from production releases
   - Compare performance over time

## Troubleshooting

### Tests Won't Run
```bash
# Ensure environment is set up
uv sync --all-extras

# Activate virtual environment
source .venv/bin/activate

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

## File Descriptions

### Test Infrastructure Files

| File | Purpose |
|------|---------|
| `conftest.py` | Pytest configuration, fixtures, and hooks |
| `generate_test_report.py` | Markdown/JSON report generation |
| `test_logging_config.py` | Logging setup and execution tracking |
| `pytest_output_plugin.py` | Pytest plugin for enhanced output |
| `run_daf_tests.sh` | Main test orchestration script |
| `README.md` | Test suite documentation |

### Test Suites

| File | Purpose |
|------|---------|
| `test_config.py` | Configuration loading and validation |
| `test_environment_checks.py` | Environment and dependency validation |
| `test_sweeps.py` | Hyperparameter sweep functionality |
| `test_comparison.py` | Policy comparison and analysis |
| `test_distributed_training.py` | Multi-machine training |
| `test_deployment.py` | Model packaging and deployment |
| `test_mission_analysis.py` | Per-mission performance analysis |
| `test_visualization.py` | Report and dashboard generation |

## Key Features

### Comprehensive Reporting
- **Markdown Output**: Beautiful, readable reports with statistics
- **JSON Export**: Machine-readable format for CI/CD integration
- **Plain Text**: Quick reference summaries
- **Individual Logs**: Full pytest output for each test suite

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

## See Also

- `README.md` - Comprehensive test documentation
- `../AGENTS.md` - DAF module architecture
- `../README.md` - DAF source organization
- `../../AGENTS.md` - Top-level agent architecture
- `../../TESTING_GUIDE.md` - Extended testing guide
