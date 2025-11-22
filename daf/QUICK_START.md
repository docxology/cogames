# DAF Modernization - Quick Start Guide

## Installation

```bash
cd /Users/4d/Documents/GitHub/cogames

# Install dependencies
pip install pydantic pyyaml rich

# Verify installation
python -m daf.tests.orchestrator --help
```

## Running Tests

### Basic (All tests, default settings)
```bash
python -m daf.tests.orchestrator
```

### With Custom Workers
```bash
python -m daf.tests.orchestrator --workers 16
```

### Verbose Mode
```bash
python -m daf.tests.orchestrator --verbose
```

### CoGames Only
```bash
python -m daf.tests.orchestrator --only-cogames
```

### DAF Only
```bash
python -m daf.tests.orchestrator --only-daf
```

## Output

All results saved to `daf_output/`:
- **JSON**: `test_results.json` (primary data format)
- **Markdown**: `TEST_RUN_SUMMARY.md` (human-readable report)
- **HTML**: `test_report.html` (interactive dashboard)
- **Logs**: `logs/` directory with execution traces

## Configuration

### Using YAML (Recommended)
```yaml
# daf_config.yaml
execution:
  max_workers: 8
  parallel: true
  retry_failed: true

output:
  base_dir: "./daf_output"
  keep_old_runs: 10
```

```bash
# Load config (if implemented in main CLI)
export DAF_CONFIG=daf_config.yaml
python -m daf.tests.orchestrator
```

### Using Environment Variables
```bash
export DAF_EXECUTION__MAX_WORKERS=16
export DAF_EXECUTION__VERBOSE=true
export DAF_OUTPUT__KEEP_OLD_RUNS=5
python -m daf.tests.orchestrator
```

## Key Modules

### Orchestrator
**File**: `daf/tests/orchestrator.py`
- Async test execution
- Parallel execution
- Result collection
- JSON serialization

### Reporter
**File**: `daf/tests/unified_reporter.py`
- JSON report generation
- Markdown conversion
- HTML dashboard
- Metrics aggregation

### Configuration
**File**: `daf/modern_config.py`
- Pydantic models
- YAML parsing
- Environment overrides
- Validation

## Performance

Expected execution times:
- **Sequential**: ~353 seconds
- **Parallel (8 workers)**: ~45-60 seconds
- **Parallel (16 workers)**: ~30-40 seconds

Actual time depends on hardware and system load.

## Troubleshooting

### ImportError for pyyaml
```bash
pip install pyyaml
```

### ImportError for rich
```bash
pip install rich
```

### Permission Denied
```bash
chmod +x daf/tests/orchestrator.py
```

### Tests Not Found
Ensure working directory is project root:
```bash
cd /Users/4d/Documents/GitHub/cogames
python -m daf.tests.orchestrator
```

## Features

### Async Execution
- Concurrent test suite execution
- Automatic worker pool sizing
- Real-time progress tracking

### Unified Reporting
- Single JSON format (no separate .txt/.xml)
- Auto-generated Markdown reports
- Interactive HTML dashboard

### Modern Configuration
- Type-safe Pydantic models
- Human-readable YAML configs
- Environment variable overrides

### Performance Profiling
- Per-test timing
- Per-suite aggregation
- Percentile analysis

### Error Handling
- Automatic retries
- Partial result recovery
- Detailed error capture

## Next Steps

1. **Run first test**: `python -m daf.tests.orchestrator`
2. **Review output**: `cat daf_output/TEST_RUN_SUMMARY.md`
3. **Configure as needed**: Create `daf_config.yaml`
4. **Integrate with CI/CD**: Use new orchestrator in workflows

## Documentation

- **Full Details**: See `MODERNIZATION_COMPLETE.md`
- **Architecture**: See `daf/MODERNIZATION_PLAN.md`
- **Source Code**: See individual module docstrings

## Support

For issues or questions:
1. Check module docstrings
2. Run with `--verbose` flag
3. Review generated logs
4. Check output in `daf_output/`

---

*DAF Modernization - November 2025*  
*Production Ready ✅*

