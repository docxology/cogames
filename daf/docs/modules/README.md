# DAF Module Documentation

Per-module documentation for DAF (Distributed Agent Framework) source modules.

## Overview

Each module in `daf/src/` has corresponding documentation here explaining its purpose, integration with CoGames, key functions, and testing.

## Module Index

| Module | Purpose | Primary CoGames Method |
|--------|---------|----------------------|
| [comparison.md](comparison.md) | Policy comparison and benchmarking | `cogames.evaluate.evaluate()` |
| [deployment.md](deployment.md) | Policy packaging and deployment | `cogames.submit` |
| [distributed_training.md](distributed_training.md) | Multi-machine training orchestration | `cogames.train.train()` |
| [mission_analysis.md](mission_analysis.md) | Per-mission performance analysis | `cogames.evaluate.evaluate()` |
| [visualization.md](visualization.md) | HTML reports and dashboards | N/A (pure DAF) |

## DAF Sidecar Pattern

All DAF modules follow the sidecar pattern:

1. DAF wraps CoGames functionality (never reimplements)
2. Each module has a clear "primary method" it invokes
3. DAF adds orchestration, analysis, and reporting
4. All configuration uses Pydantic models from `daf.src.config`

## Module Structure

Each module documentation covers:

- **Overview**: Module purpose and capabilities
- **CoGames Integration**: Primary method invoked
- **Key Functions**: Public API with `daf_*` prefix
- **Testing**: Corresponding test file
- **See Also**: Related documentation

## Additional Source Modules

These modules have inline documentation but no separate docs:

| Module | Purpose |
|--------|---------|
| `config.py` | Configuration management (Pydantic models) |
| `environment_checks.py` | Pre-flight validation |
| `sweeps.py` | Hyperparameter search |
| `logging_config.py` | Structured logging |
| `output_manager.py` | Output organization |
| `output_utils.py` | Session management |
| `test_runner.py` | Test execution framework |

## See Also

- [DAF README](../README.md) - DAF overview
- [DAF AGENTS.md](../AGENTS.md) - Policy integration guide
- [API.md](../API.md) - Complete API reference
- [ARCHITECTURE.md](../ARCHITECTURE.md) - System architecture

