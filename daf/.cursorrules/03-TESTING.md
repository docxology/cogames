# DAF Testing Cursorrules

## Test Infrastructure

DAF includes unified test runner and report generation infrastructure.

## TestRunner API

### Initialization

```python
from daf.test_runner import TestRunner

runner = TestRunner(
    output_base_dir="./daf_output",
    verbose=True
)
```

### Running Tests

```python
# Run single test suite
result = runner.run_test_suite(
    test_path="tests/test_cli.py",
    category="cogames",
    suite_name="cli"  # Optional friendly name
)

# Run multiple test suites
runner.run_test_batch([
    ("tests/test_cli.py", "cogames", "cli"),
    ("daf/tests/test_config.py", "daf", "config"),
    ("daf/tests/test_sweeps.py", "daf", "sweeps"),
])
```

### Saving and Reporting

```python
# Save all test outputs
test_output_dir = runner.save_test_outputs()

# Generate comprehensive report
report_file = runner.save_test_report()

# Print formatted summary
runner.print_test_summary()
runner.print_failed_tests()

# Cleanup resources
runner.cleanup()
```

## Report Generation

### TestReportGenerator API

```python
from daf.generate_test_report import TestReportGenerator

generator = TestReportGenerator("./daf_output/evaluations/tests")

# Collect results
results = generator.collect_all_results()

# Generate summary
summary = generator.generate_summary(results)

# Print summary
generator.print_summary(summary)

# Save reports
generator.save_summary(summary, "./report.json")
generator.generate_html_report(summary, "./report.html")
```

### Standalone Usage

```python
from daf.generate_test_report import generate_report_from_outputs

# Generate all reports at once
summary = generate_report_from_outputs(
    test_output_dir="./daf_output/evaluations/tests",
    output_json="./daf_output/reports/test_report.json",
    output_html="./daf_output/reports/test_report.html"
)
```

## Test Execution

### Running DAF Tests

```bash
# Run all DAF tests
python -m pytest daf/tests/ -v

# Run specific test module
python -m pytest daf/tests/test_sweeps.py -v

# Run with coverage
python -m pytest daf/tests/ --cov=daf/src

# Run with specific marker
python -m pytest daf/tests/ -m "not slow" -v
```

### Complete Test Workflow

```bash
# 1. Run all tests with organized output
./daf/tests/run_daf_tests.sh

# 2. Generate comprehensive report
python daf/src/generate_test_report.py \
    daf_output/evaluations/tests \
    daf_output/reports/test_report.json \
    daf_output/reports/test_report.html

# 3. View HTML report
open daf_output/reports/test_report.html
```

## Writing Tests

### Test File Structure

```python
# tests/test_module_name.py

import pytest
from pathlib import Path
from daf.module_name import ClassUnderTest, function_under_test

class TestClassName:
    """Tests for ClassName."""
    
    @pytest.fixture
    def instance(self):
        """Create instance for testing."""
        return ClassUnderTest(param="value")
    
    def test_basic_functionality(self, instance):
        """Test basic operation."""
        result = instance.method()
        assert result is not None
        assert isinstance(result, dict)
    
    def test_error_handling(self, instance):
        """Test error cases."""
        with pytest.raises(ValueError):
            instance.method(invalid_arg)
    
    def test_with_file_operations(self, tmp_path):
        """Test with temporary files."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        assert test_file.read_text() == "content"

def test_function():
    """Test standalone function."""
    result = function_under_test(arg1, arg2)
    assert result == expected_value
```

### Test Organization

```
daf/tests/
├── test_config.py              # Configuration tests
├── test_output_manager.py      # Output manager tests
├── test_logging_config.py      # Logging tests
├── test_test_runner.py         # Test runner tests
├── test_sweeps.py              # Sweep tests
├── test_comparison.py          # Comparison tests
└── run_daf_tests.sh            # Test runner script
```

### Test Output Organization

Tests automatically save outputs to:

```
daf_output/evaluations/tests/
├── cogames/
│   ├── cli_output.txt
│   ├── core_output.txt
│   └── ... (10 test suites)
├── daf/
│   ├── config_output.txt
│   ├── logging_output.txt
│   └── ... (8 test suites)
└── test_report.json
```

## Test Patterns

### Pattern 1: Simple Unit Test

```python
def test_function_returns_expected_value():
    """Test basic function."""
    result = my_function(arg1, arg2)
    assert result == expected_value
```

### Pattern 2: Testing with Fixtures

```python
@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return {"key": "value"}

def test_process_data(sample_data):
    """Test with fixture data."""
    result = process(sample_data)
    assert result["key"] == "processed_value"
```

### Pattern 3: Testing Error Handling

```python
def test_invalid_input_raises_error():
    """Test error handling."""
    with pytest.raises(ValueError, match="Invalid"):
        my_function(invalid_input)
```

### Pattern 4: Integration Test

```python
def test_full_workflow(tmp_path):
    """Test complete workflow."""
    # Setup
    config = Config(output_dir=tmp_path)
    
    # Execute
    result = workflow(config)
    
    # Verify
    assert result.success
    assert (tmp_path / "output.json").exists()
```

### Pattern 5: Testing OutputManager

```python
def test_output_manager_organizes_by_operation(tmp_path):
    """Test output organization."""
    output_mgr = OutputManager(base_dir=tmp_path)
    
    # Save results
    output_mgr.save_json_results(
        {"key": "value"},
        operation="sweep",
        filename="results"
    )
    
    # Verify organization
    results_file = tmp_path / "sweeps" / output_mgr.session_id / "results.json"
    assert results_file.exists()
```

### Pattern 6: Testing DAFLogger

```python
def test_logger_tracks_operations():
    """Test operation tracking."""
    logger = create_daf_logger("test")
    
    with logger.track_operation("test_op"):
        pass
    
    summary = logger.tracker.get_summary()
    assert summary["total_operations"] == 1
    assert summary["successful"] == 1
```

## Test Fixtures

### Common Fixtures

```python
@pytest.fixture
def output_manager(tmp_path):
    """Provide OutputManager with temp directory."""
    return OutputManager(base_dir=tmp_path)

@pytest.fixture
def daf_logger(tmp_path):
    """Provide DAFLogger with temp directory."""
    return create_daf_logger(
        "test",
        log_dir=tmp_path / "logs",
        verbose=True
    )

@pytest.fixture
def sample_config():
    """Provide sample DAF configuration."""
    return DAFConfig(
        output_dir=Path("./daf_output"),
        verbose=False,
    )
```

## Best Practices

### DO ✓

- Write tests for all new modules
- Use descriptive test names
- Test both success and error cases
- Use fixtures for setup/teardown
- Test with temporary directories
- Use pytest markers for categorization
- Run tests before committing
- Keep tests focused and fast

### DON'T ✗

- Skip error case testing
- Use hardcoded paths in tests
- Create permanent files in tests
- Use print() for test output
- Make tests depend on each other
- Skip integration tests
- Ignore test failures
- Make tests too slow (>1s each)

## Test Markers

### Using Pytest Markers

```python
# Mark test as slow
@pytest.mark.slow
def test_long_operation():
    pass

# Mark test as needing network
@pytest.mark.network
def test_api_integration():
    pass

# Run specific markers
# pytest -m "not slow" -v
# pytest -m "network" -v
```

## Coverage

### Running Coverage

```bash
# Generate coverage report
python -m pytest daf/tests/ --cov=daf/src --cov-report=html

# View coverage
open htmlcov/index.html

# Coverage thresholds
python -m pytest daf/tests/ --cov=daf/src --cov-fail-under=80
```

## Continuous Integration

### Pre-Commit Checks

```bash
# Before committing, run:
./daf/tests/run_daf_tests.sh

# Check for linting errors
ruff check daf/src/

# Generate test report
python daf/src/generate_test_report.py daf_output/evaluations/tests
```

## Troubleshooting

### Test not found?
Ensure test functions start with `test_`:
```python
# ✓ Correct
def test_my_function():
    pass

# ✗ Wrong (won't run)
def my_test_function():
    pass
```

### Fixture not working?
Ensure fixture is in same file or conftest.py:
```python
# conftest.py
@pytest.fixture
def shared_fixture():
    return value
```

### Test passing locally but failing in CI?
Check for:
- Hardcoded paths
- System-specific assumptions
- Missing dependencies
- File permission issues

---

**Status**: Production Ready ✅
**Last Updated**: November 21, 2024






