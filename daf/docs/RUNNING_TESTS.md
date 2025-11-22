# Running Tests

**CoGames includes comprehensive test suites for both core functionality and DAF sidecar utilities.**

## Quick Start

### Run Full Test Suite (CoGames + DAF)

From the **project root directory**:

```bash
cd /Users/4d/Documents/GitHub/cogames
./daf/tests/run_daf_tests.sh
```

Or use the convenience wrapper:

```bash
./daf/scripts/run_all_tests.sh
```

This runs **both phases** in sequence:
- **Phase 1**: Top-level CoGames tests (core functionality)
- **Phase 2**: DAF tests (sidecar utilities)

### Run CoGames Tests Only

```bash
cd /Users/4d/Documents/GitHub/cogames
python -m pytest tests/ -v
```

### Run DAF Tests Only

```bash
cd /Users/4d/Documents/GitHub/cogames
python -m pytest daf/tests/ -v
```

Or use the DAF-only variant:

```bash
./daf/scripts/run_daf_tests.sh
```

**Note**: DAF tests are located in `daf/tests/` to consolidate DAF as a self-contained sidecar utility.

## Upfront Test Reporting

The test runner automatically collects test counts before execution and includes them in the test summary report.

### What Happens Automatically

1. **Collection Phase**: Before running any tests, `pytest --collect-only` counts tests in each suite
2. **Test Plan**: Expected test counts are saved to `daf_output/test_plan.json`
3. **Comparison**: The markdown report compares expected vs actual test counts
4. **Variance Detection**: Highlights if tests were added, removed, or skipped

### Test Plan Output

Example output from the collection phase:

```
==========================================
TEST COLLECTION PHASE: Counting Tests Before Execution
==========================================

Collecting CoGames test counts...
  CLI Tests: 6 tests
  Core Game Tests: 4 tests
  Procedural Maps Tests: 11 tests
  ...

Collecting DAF test counts...
  Configuration Tests: 15 tests
  Environment Check Tests: 13 tests
  ...

==========================================
TEST PLAN SUMMARY
==========================================
Phase 1 (CoGames): 159 tests across 10 suites
Phase 2 (DAF):     96 tests across 8 suites
Total to execute:  255 tests
Test plan saved to: daf_output/test_plan.json
==========================================
```

### Expected vs Actual Report Section

The markdown report includes a "Test Plan" section showing the Expected vs Actual test counts:

| Phase | Expected | Actual | Variance |
|-------|----------|--------|----------|
| Phase 1 (CoGames) | 159 | 159 | 0 |
| Phase 2 (DAF) | 96 | 96 | 0 |
| **Total** | **255** | **255** | **0** |

**Note:** Positive variance indicates more tests collected than planned, negative indicates fewer.

### Interpreting Variance

- **Zero variance (0)**: Actual test count matches expected - no tests added or removed
- **Positive variance (+)**: More tests than expected - new tests have been added
- **Negative variance (-)**: Fewer tests than expected - tests have been removed or skipped

## Important Notes

### ⚠️ Run from Project Root

**Always run pytest from the project root**, not from the `daf/` directory:

```bash
# ✅ CORRECT - Run from project root
cd /Users/4d/Documents/GitHub/cogames
uv run pytest daf/tests/test_daf_*.py -v

# ❌ WRONG - Running from daf/ directory may not find tests correctly
cd /Users/4d/Documents/GitHub/cogames/daf
uv run pytest  # May not find tests correctly!
```

### Why?

- Pytest is configured to look in the `tests/` directory (`testpaths = ["tests"]` in `pyproject.toml`)
- The `daf` module is added to Python path via `pythonpath = ["src", "."]`
- When you're in `daf/`, pytest doesn't find the `daf/tests/` directory correctly

## Running Specific Test Suites

### CoGames Core Tests

```bash
# CLI tests
python -m pytest tests/test_cli.py -v

# Core game tests
python -m pytest tests/test_cogs_vs_clips.py -v

# Specific mission variant tests
python -m pytest tests/test_cvc_assembler_hearts.py -v

# Procedural map generation
python -m pytest tests/test_procedural_maps.py -v

# Scripted policies
python -m pytest tests/test_scripted_policies.py -v

# Training integration
python -m pytest tests/test_train_integration.py -v

# All CoGames tests
python -m pytest tests/ -v
```

### DAF-Specific Tests

```bash
# Config tests
python -m pytest daf/tests/test_config.py -v

# Environment checks
python -m pytest daf/tests/test_environment_checks.py -v

# Hyperparameter sweeps
python -m pytest daf/tests/test_sweeps.py -v

# Policy comparison
python -m pytest daf/tests/test_comparison.py -v

# Deployment infrastructure
python -m pytest daf/tests/test_deployment.py -v

# Distributed training
python -m pytest daf/tests/test_distributed_training.py -v

# Visualization/reporting
python -m pytest daf/tests/test_visualization.py -v

# Mission-level analysis
python -m pytest daf/tests/test_mission_analysis.py -v

# All DAF tests
python -m pytest daf/tests/ -v
```

## Test Coverage

### Phase 1: CoGames Core Tests

- ✅ `tests/test_cli.py` - CLI command tests
- ✅ `tests/test_cogs_vs_clips.py` - Core game mechanics
- ✅ `tests/test_cvc_assembler_hearts.py` - Mission variants
- ✅ `tests/test_procedural_maps.py` - Map generation
- ✅ `tests/test_scripted_policies.py` - Baseline policies
- ✅ `tests/test_train_integration.py` - Training pipeline
- ✅ `tests/test_train_vector_alignment.py` - Vector math verification
- ✅ `tests/test_all_games_describe.py` - Mission discovery
- ✅ `tests/test_all_games_eval.py` - Evaluation harness
- ✅ `tests/test_all_games_play.py` - Interactive play

### Phase 2: DAF Tests

- ✅ `daf/tests/test_config.py` - 15 tests
- ✅ `daf/tests/test_environment_checks.py` - 13 tests  
- ✅ `daf/tests/test_sweeps.py` - 16 tests
- ✅ `daf/tests/test_comparison.py` - 12 tests
- ✅ `daf/tests/test_deployment.py` - 10 tests
- ✅ `daf/tests/test_distributed_training.py` - 10 tests
- ✅ `daf/tests/test_visualization.py` - 10 tests
- ✅ `daf/tests/test_mission_analysis.py` - 10 tests

**Total: 100+ CoGames tests + 100+ DAF tests = 200+ test cases**

## Troubleshooting

### "No tests collected"

If you see `collected 0 items`, you're likely in the wrong directory:

```bash
# Check current directory
pwd

# Should be: /Users/4d/Documents/GitHub/cogames
# If not, cd to project root
cd /Users/4d/Documents/GitHub/cogames
```

### "ModuleNotFoundError: No module named 'daf'"

This is fixed by the `pythonpath = ["src", "."]` configuration in `pyproject.toml`. If you still see this error, ensure you're running from the project root.

### "ModuleNotFoundError: No module named 'mettagrid'"

This is expected - `mettagrid` needs to be installed. Tests that don't require mettagrid will still pass. Tests that import cogames modules requiring mettagrid will be skipped gracefully.
