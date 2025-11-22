# DAF Sidecar Architecture

## Overview

**DAF (Distributed Agent Framework) is a sidecar utility and extension for CoGames.**

DAF follows a sidecar pattern where it lives alongside CoGames and extends its functionality without duplicating or replacing core methods. All DAF code is contained in the `daf/` folder, separate from the core `src/cogames/` framework.

## Sidecar Principles

### 1. Separation
- **DAF Location**: `daf/` folder (top-level)
- **Core Location**: `src/cogames/` folder
- **Clear Boundaries**: DAF never modifies core CoGames code

### 2. Extension, Not Replacement
- DAF **wraps** and **extends** CoGames methods
- DAF **never duplicates** core functionality
- DAF **never replaces** core methods

### 3. Invocation Pattern
All DAF functions invoke underlying CoGames methods:

| DAF Function | CoGames Method Invoked |
|-------------|------------------------|
| `daf_launch_distributed_training()` | `cogames.train.train()` |
| `daf_compare_policies()` | `cogames.evaluate.evaluate()` |
| `daf_check_environment()` | `cogames.device.resolve_training_device()` |
| `daf_analyze_mission()` | `cogames.cli.mission.get_mission_name_and_config()` |
| `daf_launch_sweep()` | `cogames.evaluate.evaluate()` |

### 4. Clear Prefixing
- All DAF functions use `daf_` prefix
- Distinguishes DAF extensions from core methods
- Prevents naming conflicts

### 5. Optional Dependency
- **Core CoGames**: Works independently without DAF
- **DAF**: Requires CoGames to function
- **Users**: Can use CoGames without DAF, or add DAF for advanced workflows

## Module Integration Map

### distributed_training.py
- **Wraps**: `cogames.train.train()`
- **Adds**: Multi-node coordination, distributed backends
- **Pattern**: Single-node → direct invocation, Multi-node → wrapper layer

### sweeps.py
- **Uses**: `cogames.evaluate.evaluate()`
- **Adds**: Hyperparameter search strategies, trial management
- **Pattern**: Each trial invokes `cogames.evaluate.evaluate()`

### comparison.py
- **Uses**: `cogames.evaluate.evaluate()`
- **Adds**: Statistical analysis, pairwise comparisons, reporting
- **Pattern**: Evaluation → statistical analysis → reporting

### environment_checks.py
- **Uses**: `cogames.device.resolve_training_device()`
- **Adds**: Comprehensive environment validation
- **Pattern**: Device resolution → additional checks → validation report

### mission_analysis.py
- **Uses**: `cogames.cli.mission.get_mission_name_and_config()`
- **Adds**: Mission discovery, metadata extraction, validation
- **Pattern**: Mission loading → metadata extraction → analysis

### deployment.py
- **Uses**: `cogames.auth`, `cogames.cli.submit` patterns
- **Adds**: Packaging, validation, monitoring
- **Pattern**: Policy loading → packaging → validation → deployment

### visualization.py
- **Extends**: Patterns from `scripts/run_evaluation.py`
- **Adds**: DAF-specific visualizations, HTML reports
- **Pattern**: Uses matplotlib patterns, adds DAF reporting

### orchestrators.py
- **Chains**: All DAF modules into workflows
- **Pattern**: Environment check → DAF operations → reporting
- **Integration**: Each stage invokes CoGames methods via DAF modules

## File Structure

```
cogames/
├── daf/                          # DAF sidecar utility
│   ├── __init__.py              # Package entry point
│   ├── config.py                # Configuration management
│   ├── environment_checks.py    # Environment validation
│   ├── distributed_training.py  # Wraps cogames.train.train()
│   ├── sweeps.py                # Uses cogames.evaluate.evaluate()
│   ├── comparison.py            # Uses cogames.evaluate.evaluate()
│   ├── visualization.py          # Extends evaluation patterns
│   ├── deployment.py             # Uses cogames.auth patterns
│   ├── orchestrators.py         # Chains DAF modules
│   ├── mission_analysis.py     # Uses cogames.cli.mission
│   ├── examples/                # Example configurations
│   └── *.md                     # Documentation
├── src/cogames/                  # Core CoGames framework
│   ├── train.py                 # Core training (invoked by DAF)
│   ├── evaluate.py              # Core evaluation (invoked by DAF)
│   ├── device.py                # Device resolution (invoked by DAF)
│   └── ...
├── daf/
│   ├── tests/                    # DAF test suite (self-contained)
│   │   ├── test_daf_*.py        # DAF tests
│   │   └── run_daf_tests.sh     # DAF test runner
│   └── ...
├── tests/                        # Core CoGames tests
└── scripts/                      # Utility scripts
```

## Development Guidelines

### When Adding DAF Features

1. **Check if CoGames already provides it**
   - If yes: Wrap/extend it, don't duplicate
   - If no: Add to DAF, document integration points

2. **Use CoGames methods internally**
   - Always invoke CoGames methods
   - Never reimplement CoGames functionality
   - Document which CoGames methods are used

3. **Follow naming conventions**
   - All functions: `daf_*` prefix
   - All classes: `DAF*` prefix
   - Clear distinction from core methods

4. **Document integration**
   - Docstrings should mention CoGames methods invoked
   - Architecture docs should show integration points
   - Examples should show DAF invoking CoGames

### Testing Sidecar Integration

Tests verify DAF correctly invokes CoGames methods:
- Integration tests use real CoGames methods
- No mocks of CoGames core functionality
- Tests verify sidecar pattern is maintained

## Benefits of Sidecar Pattern

1. **Separation of Concerns**: Core framework vs. advanced utilities
2. **Optional Enhancement**: Users can use CoGames without DAF
3. **Clear Boundaries**: Easy to understand what's core vs. extension
4. **Maintainability**: Changes to DAF don't affect core CoGames
5. **Extensibility**: Easy to add more sidecar utilities

## Migration Notes

If you're migrating from thinking of DAF as part of core CoGames:

- **Before**: DAF functions mixed with core functions
- **After**: DAF clearly separated in `daf/` folder
- **Before**: DAF might duplicate CoGames methods
- **After**: DAF always invokes CoGames methods
- **Before**: Unclear what's core vs. extension
- **After**: Clear sidecar architecture documented

## References

- `.cursorrules` - Development rules documenting sidecar pattern
- `daf/ARCHITECTURE.md` - Detailed architecture documentation
- `daf/README.md` - Quick start guide
- `daf/API.md` - Complete API reference with integration points

