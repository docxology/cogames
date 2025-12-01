# DAF Code Style Cursorrules

## Python Standards

### Version
- **Minimum**: Python 3.9
- **Target**: Python 3.10+
- **Current**: Python 3.12

### Imports

Order imports strictly:

```python
# 1. __future__ imports (always first)
from __future__ import annotations

# 2. Standard library (alphabetical)
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 3. Third-party (alphabetical)
import numpy as np
import yaml
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

# 4. CoGames local imports
from cogames.game import load_mission_config
from cogames.policy import PolicySpec

# 5. DAF local imports (relative)
from daf.config import DAFConfig
from daf.output_manager import OutputManager
```

### Type Hints

**Required** for all public functions and methods:

```python
# ✓ Good - complete type hints
def process_results(
    results: Dict[str, Any],
    output_dir: Path | str,
    verbose: bool = False,
) -> Path:
    """Process results and save to directory."""

# ✗ Bad - missing type hints
def process_results(results, output_dir, verbose=False):
    """Process results and save to directory."""
```

### Modern Type Syntax

Use `|` for unions (Python 3.10+):

```python
# ✓ Good (modern)
def process(path: Path | str) -> dict | None:
    pass

# ✗ Old (works but avoid)
from typing import Union
def process(path: Union[Path, str]) -> Optional[dict]:
    pass
```

### Docstrings

Google-style docstrings required for all public APIs:

```python
def load_config(path: Path | str) -> DAFConfig:
    """Load DAF configuration from file.
    
    Supports both YAML and JSON formats with validation.
    
    Args:
        path: Path to configuration file (YAML or JSON)
        
    Returns:
        DAFConfig instance with validated settings
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        ValueError: If configuration format is invalid
        
    Example:
        >>> config = load_config("config.yaml")
        >>> print(config.output_dir)
        ./daf_output
    """
    # Implementation
```

### Class Docstrings

```python
class OutputManager:
    """Centralized manager for all DAF outputs and logging.
    
    Provides organized output directory structure, structured logging,
    automatic results tracking, and session management for all DAF
    operations (sweeps, comparisons, training, etc.).
    
    Attributes:
        base_dir: Root output directory path
        dirs: OutputDirectories instance with all organized paths
        logger: Configured logger instance
        session_id: Unique session ID (YYYYMMDD_HHMMSS)
        
    Example:
        >>> output_mgr = OutputManager(base_dir="./daf_output")
        >>> output_mgr.save_json_results(data, operation="sweep")
    """
```

## Formatting

### Line Length
- **Max**: 100 characters (ruff configured)
- **Soft**: 88 characters (black style)

### Indentation
- **Style**: 4 spaces (never tabs)
- **Continuation**: 4 spaces additional

```python
# ✓ Good
result = long_function_name(
    arg1,
    arg2,
    arg3,
)

# ✗ Bad (tabs)
result = long_function_name(
	arg1,
	arg2,
)
```

### Spacing

```python
# ✓ Good spacing
x = 1
y = 2 + 3
list_items = [1, 2, 3]
dict_items = {"key": "value"}

# ✗ Bad spacing
x=1
y=2+3
list_items=[1,2,3]
dict_items={"key":"value"}
```

## Naming Conventions

### Variables and Functions

- **Style**: snake_case
- **Length**: Descriptive (avoid single letters except i, j for loops)

```python
# ✓ Good
sweep_results = sweeps.daf_launch_sweep(config)
best_score = sweep_results.best_score
for i, trial in enumerate(trials):
    pass

# ✗ Bad
sr = sweeps.daf_launch_sweep(config)
bs = sr.best_score
for x, y in enumerate(trials):
    pass
```

### Classes

- **Style**: PascalCase
- **Suffix**: Add type (e.g., Manager, Config, Error)

```python
# ✓ Good
class OutputManager:
    pass

class DAFConfig:
    pass

class OperationError(Exception):
    pass

# ✗ Bad
class output_manager:
    pass

class Config:
    pass

class CustomError:
    pass
```

### Constants

- **Style**: UPPER_SNAKE_CASE
- **Location**: Module level

```python
# ✓ Good
DEFAULT_OUTPUT_DIR = Path("./daf_output")
MAX_PARALLEL_JOBS = 4
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024

# ✗ Bad
default_output_dir = Path("./daf_output")
maxParallelJobs = 4
```

## Error Handling

### Explicit Errors

```python
# ✓ Good - specific error with context
if not config_path.exists():
    raise FileNotFoundError(f"Config not found: {config_path}")

# ✗ Bad - generic error
if not config_path.exists():
    raise Exception("File error")
```

### Error Messages

Include context and suggestions:

```python
# ✓ Good error message
raise ValueError(
    f"Invalid learning rate: {lr}. "
    f"Expected 0.0 < lr <= 1.0, got {lr}"
)

# ✗ Bad error message
raise ValueError("Invalid parameter")
```

### Exception Handling

```python
# ✓ Good - specific exception
try:
    result = risky_operation()
except OperationError as e:
    logger.error(f"Operation failed: {e}")
    raise

# ✗ Bad - bare except
try:
    result = risky_operation()
except:
    pass
```

## Code Organization

### Module Structure

```python
# 1. Module docstring
"""Module for output management and organization.

Provides centralized output directory management for all DAF operations.
"""

# 2. Imports
from __future__ import annotations
import json
from pathlib import Path

# 3. Constants
DEFAULT_OUTPUT_DIR = Path("./daf_output")
SESSION_ID_FORMAT = "%Y%m%d_%H%M%S"

# 4. Classes
class OutputManager:
    pass

# 5. Functions
def get_output_manager():
    pass

# 6. Module initialization
if __name__ == "__main__":
    pass
```

### Class Organization

```python
class MyClass:
    """Class docstring."""
    
    # Class variables
    class_var = 10
    
    def __init__(self, param: str) -> None:
        """Initialize."""
        self.param = param
    
    def public_method(self) -> str:
        """Public method."""
        return self.param
    
    def _private_method(self) -> None:
        """Private method (single underscore)."""
        pass
    
    @property
    def computed_property(self) -> str:
        """Computed property."""
        return self.param.upper()
```

## Comments

### When to Comment

```python
# ✓ Good - explains why, not what
# Sort by date first to ensure chronological order,
# then by score for ties
results = sorted(results, key=lambda x: (x.date, -x.score))

# ✗ Bad - obvious from code
# Sort results
results = sorted(results)

# ✓ Good - explains complex logic
# Rotating handler prevents massive log files
# Max 10MB per file, keep 5 backups
handler = RotatingFileHandler(
    filename,
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
)
```

### Comment Style

```python
# Single line comment (space after #)
# This is clear

#Bad comment (no space)
# Good comment

"""
Docstring for multi-line documentation.
This is separate from comments.
"""
```

## Function Guidelines

### Function Length
- **Target**: <50 lines
- **Maximum**: <100 lines
- **If longer**: Refactor into smaller functions

### Function Parameters
- **Maximum**: 5-6 parameters
- **If more**: Use dataclass or config object

```python
# ✓ Good - reasonable parameters
def run_sweep(
    config: DAFSweepConfig,
    output_dir: Path | str,
    verbose: bool = False,
) -> SweepResults:
    pass

# ✗ Bad - too many parameters
def run_sweep(
    missions,
    policy,
    lr,
    batch_size,
    epochs,
    num_trials,
    seed,
    verbose,
):
    pass

# Better - use config object
def run_sweep(config: DAFSweepConfig) -> SweepResults:
    pass
```

### Default Parameters
```python
# ✓ Good - immutable defaults
def process(data: list[int], threshold: int = 0) -> list[int]:
    return [x for x in data if x > threshold]

# ✗ Bad - mutable defaults
def process(data: list[int], cache: dict = {}) -> dict:
    # Mutable default will be shared across calls!
    pass

# Fix mutable defaults
def process(data: list[int], cache: dict | None = None) -> dict:
    if cache is None:
        cache = {}
    return cache
```

## Linting

### Ruff Configuration

```toml
[tool.ruff]
line-length = 100
target-version = "py39"

[tool.ruff.lint]
select = ["E", "F", "W"]
```

### Checking Code

```bash
# Check for linting errors
ruff check daf/src/

# Format code (if needed)
ruff format daf/src/

# Show errors
ruff check daf/src/ --show-fixes
```

### CI Checks

All code must pass:
```bash
# Linting (zero errors expected)
ruff check daf/src/

# Type hints verification
# (all functions must have type hints)

# Tests passing
pytest daf/tests/ -v
```

## Common Patterns

### Context Managers

```python
# ✓ Good - use with statement
with open(file_path) as f:
    content = f.read()

# ✗ Bad - manual close
f = open(file_path)
content = f.read()
f.close()
```

### Path Handling

```python
# ✓ Good - use pathlib
from pathlib import Path
path = Path("./daf_output") / "results.json"

# ✗ Bad - string concatenation
path = "./daf_output" + "/" + "results.json"
```

### String Formatting

```python
# ✓ Good - f-strings (Python 3.6+)
name = "World"
greeting = f"Hello {name}!"

# ✗ Old - .format()
greeting = "Hello {}!".format(name)

# ✗ Very old - %
greeting = "Hello %s!" % name
```

## Version Control

### Commit Messages

```
Keep it clear and concise.

Good:
  "Add OutputManager for organized outputs"
  "Fix logging metrics collection in sweeps"

Bad:
  "stuff"
  "WIP"
  "fix bug"
```

### Branches

```
Feature: feature/output-manager
Bugfix: bugfix/logging-error
Docs: docs/logging-guide
```

---

**Status**: Production Ready ✅
**Last Updated**: November 21, 2024






