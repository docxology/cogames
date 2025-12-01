# DAF Modular Cursorrules - Complete Index

## Overview

This directory contains comprehensive, modular cursorrules files for the DAF (Distributed Agent Framework) sidecar to CoGames. These files document architecture, patterns, standards, and best practices for all DAF development.

## Files Summary

### 📋 README.md
**Navigation and guide to all cursorrules files**

- How to use these files
- Quick reference for common tasks
- Learning paths by experience level
- Quick checklist for development
- File details and purposes

**Start here** if you're new to DAF cursorrules.

---

### 🏗️ 00-ARCHITECTURE.md (15 min read)
**Core architectural principles and module organization**

**Key Topics**:
- Core design principles (non-duplicating, modular, organized, logged, etc.)
- Module responsibilities and organization
- Operation flow pattern (init → configure → log → execute → save → complete)
- Data flow through DAF
- Integration with CoGames (no duplication rules)
- Configuration patterns
- Session ID convention (YYYYMMDD_HHMMSS)
- Error handling strategy
- Testing and documentation architecture
- Development workflow

**When to Read**:
- Starting new module
- Understanding project structure
- Planning feature design
- Learning DAF principles

**Key Pattern**:
```python
1. Initialize → 2. Configure → 3. Log Start → 4. Execute → 
5. Save Results → 6. Complete → 7. Generate Report
```

---

### 📁 01-OUTPUT.md (12 min read)
**Output management patterns and organized results**

**Key Topics**:
- OutputManager API (initialization, methods, patterns)
- Output directory structure (operation folders, session organization)
- Standardized JSON format for all results
- Usage patterns (sweeps, comparisons)
- Output utilities (finding sessions, exporting, cleanup)
- Best practices
- Configuration
- Error handling
- Session ID usage

**When to Read**:
- Saving results from operations
- Understanding output organization
- Finding previous results
- Backing up/exporting sessions

**Key Pattern**:
```python
output_mgr = get_output_manager()
output_mgr.log_operation_start("sweep", details={...})
results = perform_operation()
output_mgr.save_json_results(results, operation="sweep")
output_mgr.log_operation_complete("sweep", status="success")
```

**Output Structure**:
```
daf_output/
├── sweeps/sweeps/YYYYMMDD_HHMMSS/
│   ├── summary_report.json
│   └── results.json
├── comparisons/...
├── training/...
└── logs/
    ├── daf_YYYYMMDD_HHMMSS.log
    └── session_YYYYMMDD_HHMMSS.json
```

---

### 📝 02-LOGGING.md (12 min read)
**Structured logging with operation tracking and metrics**

**Key Topics**:
- DAFLogger API (initialization, methods)
- Logging patterns (simple, complex, error handling, sweeping)
- Log levels (DEBUG, INFO, WARNING, ERROR)
- Metrics tracking (OperationMetrics, OperationTracker)
- Console output formatting (sections, tables, summaries)
- Configuration
- Best practices
- Integration with OutputManager
- Log files
- Troubleshooting

**When to Read**:
- Adding logging to operations
- Tracking operation metrics
- Formatting console output
- Debugging

**Key Pattern**:
```python
logger = create_daf_logger("operation_name")
with logger.track_operation("task", metadata={...}):
    logger.info("Doing work...")
logger.print_metrics_summary()
logger.save_metrics_json(Path("./metrics.json"))
```

**Log Levels**:
- **DEBUG**: Detailed diagnostic info
- **INFO**: Major milestones
- **WARNING**: Potential issues
- **ERROR**: Actual failures

---

### ✅ 03-TESTING.md (12 min read)
**Test infrastructure and best practices**

**Key Topics**:
- TestRunner API (execution, reporting, saving)
- Report generation (JSON, HTML)
- Test execution
- Writing tests (patterns, fixtures, organization)
- Test output organization
- Best practices
- Test markers
- Coverage
- Continuous integration
- Troubleshooting

**When to Read**:
- Running tests
- Writing tests
- Generating test reports
- Setting up CI

**Key Pattern**:
```python
runner = TestRunner(output_base_dir="./daf_output")
runner.run_test_batch([...])
runner.save_test_outputs()
runner.save_test_report()
runner.print_test_summary()
```

**Test Organization**:
```
daf_output/evaluations/tests/
├── cogames/              # CoGames tests
│   ├── cli_output.txt
│   └── ...
├── daf/                  # DAF tests
│   ├── config_output.txt
│   └── ...
└── test_report.json      # Aggregated results
```

---

### 🎨 04-CODESTYLE.md (15 min read)
**Python code style and standards**

**Key Topics**:
- Python version and standards (3.9+)
- Imports organization (stdlib → 3rd party → local)
- Type hints (required for all public functions)
- Docstrings (Google-style)
- Formatting (line length, indentation, spacing)
- Naming conventions (snake_case, PascalCase, UPPER_SNAKE_CASE)
- Error handling (explicit errors, messages)
- Code organization (modules, classes, functions)
- Comments
- Function guidelines
- Linting (ruff)
- Common patterns
- Version control

**When to Read**:
- Writing code
- Code review
- Fixing style issues
- Setting up linting

**Key Rules**:
- **Type Hints**: Required for all public functions
- **Line Length**: 100 characters max
- **Docstrings**: Required for all public APIs
- **Naming**: snake_case for functions, PascalCase for classes
- **Linting**: Zero errors (ruff)

**Import Order**:
```python
# 1. __future__
from __future__ import annotations

# 2. Standard library
import json
from pathlib import Path

# 3. Third-party
import numpy as np
from pydantic import BaseModel

# 4. CoGames
from cogames.game import Game

# 5. DAF
from daf.config import DAFConfig
```

---

### 🔗 05-INTEGRATION.md (12 min read)
**Integration with CoGames policies and missions**

**Key Topics**:
- Core integration principle (non-duplicating)
- Policy interfaces (MultiAgentPolicy, AgentPolicy)
- Policy loading (PolicySpec)
- Mission configuration
- Environment interaction (Game class)
- Evaluation pattern
- Training integration
- Device management
- Verbose/logging coordination
- Seeding for reproducibility
- Dependency management
- Version compatibility
- No-duplication checklist
- Integration testing
- Common patterns
- Troubleshooting

**When to Read**:
- Working with policies
- Loading missions
- Training and evaluation
- Integrating new features

**Key Principle**:
- **Never reimplements** CoGames functionality
- Use `PolicySpec` for policy loading
- Use `load_mission_config` for missions
- Use `Game` class for environment interaction

**No-Duplication Checklist**:
- [ ] Using PolicySpec (not reimplementing policy loading)
- [ ] Using CoGames Game (not custom environments)
- [ ] Using CoGames missions (not duplicating)
- [ ] Using cogames.train (not reimplementing training)
- [ ] Using cogames.evaluate (not duplicating evaluation)
- [ ] Using CoGames error types

---

## Quick Navigation

### By Task

**"I want to..."** → **Read This**

| Task | File |
|------|------|
| Understand DAF architecture | `00-ARCHITECTURE.md` |
| Save results from operation | `01-OUTPUT.md` |
| Add logging to operation | `02-LOGGING.md` |
| Write tests | `03-TESTING.md` |
| Write Python code | `04-CODESTYLE.md` |
| Integrate with CoGames | `05-INTEGRATION.md` |
| Get started | `README.md` |

### By Experience Level

**Beginner (First Time)**:
1. `README.md` - Overview
2. `00-ARCHITECTURE.md` - Principles (sections 1-2)
3. `01-OUTPUT.md` - Usage Pattern
4. `02-LOGGING.md` - Logging Pattern

**Developer (Adding Features)**:
1. `00-ARCHITECTURE.md` - Full read
2. `01-OUTPUT.md` - Usage patterns
3. `02-LOGGING.md` - Logging patterns
4. `04-CODESTYLE.md` - Code standards
5. `03-TESTING.md` - Writing tests

**Advanced (Extending DAF)**:
1. All files thoroughly
2. Study module organization
3. Reference patterns as needed
4. Design with these patterns

### By Module

**Working on output_manager.py**:
- `00-ARCHITECTURE.md: Module Responsibilities`
- `01-OUTPUT.md` - Complete read
- `04-CODESTYLE.md` - Code standards

**Working on logging_config.py**:
- `02-LOGGING.md` - Complete read
- `04-CODESTYLE.md` - Code standards
- `03-TESTING.md` - Testing patterns

**Working on test_runner.py**:
- `03-TESTING.md` - Complete read
- `04-CODESTYLE.md` - Code standards
- `00-ARCHITECTURE.md: Testing Architecture`

**Working on policy integration**:
- `05-INTEGRATION.md` - Complete read
- `04-CODESTYLE.md` - Code standards
- `00-ARCHITECTURE.md: Integration Points`

## Key Concepts at a Glance

### Core Principle
DAF is a **non-duplicating wrapper** around CoGames that adds:
- Orchestration (sweeps, distributed training)
- Analysis (comparisons, mission analysis)
- Reporting (HTML reports, metrics)
- Organization (unified output structure)

### Key Patterns

```python
# Output Management
output_mgr = get_output_manager()
output_mgr.save_json_results(data, operation="sweep", filename="results")

# Logging
logger = create_daf_logger("operation_name")
with logger.track_operation("task"):
    pass

# Testing
runner = TestRunner()
runner.run_test_batch([...])
runner.save_test_report()

# Code Style
def function_name(param: str) -> Path:
    """Google-style docstring."""
    pass

# Integration
spec = PolicySpec(class_path="lstm")
policy = spec.load()
```

### Directory Structure

```
daf/
├── src/                   # Implementation
├── tests/                 # Tests
├── docs/                  # Documentation
├── examples/              # Examples
└── .cursorrules/         # Rules (this directory)
```

### Session Organization

```
daf_output/
├── sweeps/YYYYMMDD_HHMMSS/      # Sweep session
├── comparisons/YYYYMMDD_HHMMSS/ # Comparison session
├── training/YYYYMMDD_HHMMSS/    # Training session
├── logs/                          # All session logs
└── reports/                       # Generated reports
```

## Cross-References

Files frequently reference each other:

| From | References |
|------|-----------|
| `00-ARCHITECTURE.md` | `01-OUTPUT.md`, `02-LOGGING.md`, `03-TESTING.md` |
| `01-OUTPUT.md` | `00-ARCHITECTURE.md`, `02-LOGGING.md` |
| `02-LOGGING.md` | `00-ARCHITECTURE.md`, `01-OUTPUT.md` |
| `03-TESTING.md` | `01-OUTPUT.md`, `04-CODESTYLE.md` |
| `04-CODESTYLE.md` | All (used throughout) |
| `05-INTEGRATION.md` | `00-ARCHITECTURE.md`, `04-CODESTYLE.md` |

## Usage in Cursor/IDE

### Load Single File
```
@.cursorrules-output
```

### Load with Main Cursorrules
Reference from main `.cursorrules` file:
```
See daf/.cursorrules/ for modular rules:
- @.cursorrules-architecture
- @.cursorrules-output
- @.cursorrules-logging
- etc.
```

### Reference in Chat
```
Can you review this code following @04-CODESTYLE.md?
Help me add logging per @02-LOGGING.md patterns
Let me run tests following @03-TESTING.md
```

## Standards Met

Each file includes:
- ✅ Clear purpose and scope
- ✅ Well-organized sections
- ✅ Code examples for all patterns
- ✅ Best practices (✓ DO / ✗ DON'T)
- ✅ Troubleshooting sections
- ✅ Status and date
- ✅ Internal consistency
- ✅ External cross-references

## Maintenance

### When Updating Rules

1. Update relevant file
2. Update internal links if structure changes
3. Update this INDEX.md if adding files
4. Update README.md if changing organization
5. Increment version number
6. Update "Last Updated" date

### Version History

| Version | Date | Notes |
|---------|------|-------|
| 1.0 | 2024-11-21 | Initial creation |

## Related Documentation

- **Parent README**: `../README.md`
- **Getting Started**: `../GETTING_STARTED.md`
- **Documentation Index**: `../docs/OUTPUT_AND_LOGGING_INDEX.md`
- **Complete Documentation**: `../docs/`

## Status

✅ **Production Ready**
- All 6 modular cursorrules files complete
- Comprehensive coverage of all DAF aspects
- Professional documentation quality
- Ready for use in development

**Total Coverage**: 
- 6 modular files
- 78 sections
- 120+ code examples
- Complete architecture documentation

**Last Updated**: November 21, 2024
**Version**: 1.0

---

## Quick Links

- [Architecture Rules](./00-ARCHITECTURE.md)
- [Output Management](./01-OUTPUT.md)
- [Logging Infrastructure](./02-LOGGING.md)
- [Testing Framework](./03-TESTING.md)
- [Code Style Guide](./04-CODESTYLE.md)
- [CoGames Integration](./05-INTEGRATION.md)
- [Navigation Guide](./README.md)







