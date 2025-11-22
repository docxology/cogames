# DAF Core Architecture Cursorrules

## Overview

DAF (Distributed Agent Framework) is a sidecar framework providing distributed training, evaluation, and analysis infrastructure for CoGames agent policies. This file documents core architectural principles.

## Core Design Principles

### 1. Non-Duplicating Wrapper
- DAF wraps CoGames methods; never duplicates core functionality
- All policy management delegates to CoGames
- All environment interaction goes through CoGames
- DAF adds orchestration, analysis, and reporting layers

### 2. Modular Organization
Each concern in separate module with clear responsibility:

```
daf/src/
├── config.py              # Configuration management
├── output_manager.py      # Output organization
├── logging_config.py      # Logging infrastructure
├── test_runner.py         # Test execution
├── generate_test_report.py # Report generation
├── output_utils.py        # Session utilities
├── sweeps.py              # Hyperparameter sweeps
├── comparison.py          # Policy comparison
├── training.py            # Training orchestration
├── distributed_training.py # Multi-machine training
├── deployment.py          # Policy deployment
├── mission_analysis.py    # Mission analysis
├── visualization.py       # Visualization utilities
└── environment_checks.py  # Environment validation
```

### 3. Unified Output Structure
All operations organize outputs into single `daf_output` folder:

```
daf_output/
├── sweeps/              # Hyperparameter sweep results
├── comparisons/         # Policy comparison results
├── training/            # Training run outputs
├── deployment/          # Deployment packages
├── evaluations/         # Test & evaluation results
├── visualizations/      # Generated plots/HTML
├── logs/                # Session logs & metadata
├── reports/             # Generated reports
└── artifacts/           # Reusable artifacts
```

### 4. Structured Logging
All operations automatically tracked:

```python
with logger.track_operation("sweep", metadata={...}):
    # Automatic timing, status, error tracking
    pass
```

### 5. Production Quality
Professional infrastructure built-in:

- Comprehensive logging with metrics
- Automatic output organization
- Unified test runner and reporting
- Session tracking for reproducibility
- HTML report generation

### 6. Backward Compatible
- Existing APIs unchanged
- New features opt-in
- Default behavior preserved
- Smooth upgrade path

## Operation Flow Pattern

All DAF operations follow this pattern:

```
1. Initialize → 2. Configure → 3. Log Start → 4. Execute → 
5. Save Results → 6. Complete → 7. Generate Report
```

### Example Operation

```python
from daf.output_manager import get_output_manager
from daf.logging_config import create_daf_logger
from daf import sweeps

# 1. Initialize
output_mgr = get_output_manager(base_dir="./daf_output", verbose=True)
logger = create_daf_logger("sweep_operation")

# 2. Configure
config = DAFSweepConfig.from_yaml("sweep_config.yaml")

# 3-4. Execute with logging
with logger.track_operation("sweep", metadata={"trials": 10}):
    output_mgr.log_operation_start("sweep")
    logger.info("Running sweep...")
    
    results = sweeps.daf_launch_sweep(config)
    
    logger.info(f"Best score: {results.best_score}")

# 5. Save Results
output_mgr.save_json_results(
    results.to_dict(),
    operation="sweep",
    filename="results"
)

# 6. Complete
output_mgr.log_operation_complete("sweep", status="success")

# 7. Report & Metadata
logger.print_metrics_summary()
output_mgr.save_session_metadata()
```

## Module Responsibilities

### config.py
- Pydantic configuration models
- Environment variable integration
- Configuration file loading (YAML/JSON)
- **No**: Business logic, I/O operations

### output_manager.py
- Centralized output directory management
- Session-based organization
- JSON results tracking
- **No**: Logging details, operation logic

### logging_config.py
- Structured logging infrastructure
- Operation metrics collection
- Rich console formatting
- **No**: Output file management, operation execution

### test_runner.py
- Unified test execution
- Batch test processing
- Output capture and organization
- **No**: Pytest internals, result analysis

### generate_test_report.py
- Parse test outputs
- Generate JSON/HTML reports
- Statistical analysis
- **No**: Test execution, output organization

### output_utils.py
- Session discovery and management
- Session export/backup
- Output summary generation
- **No**: Core operations, logging setup

### sweeps.py
- Hyperparameter sweep execution
- Grid/random search strategies
- Result aggregation
- **No**: Output organization, logging (delegates to OutputManager/DAFLogger)

### comparison.py
- Multi-policy comparison
- Statistical analysis
- Result organization
- **No**: Visualization (delegates to visualization.py)

### Other modules
Follow similar pattern of clear responsibility separation.

## Data Flow

### Input Flow
```
User Config → Pydantic Model → Operation Code → CoGames
```

### Output Flow
```
Operation Results → OutputManager → Organized Folders
     ↓
   Logging → Session Metadata
     ↓
   Report Generation → JSON/HTML Reports
```

### Session Organization Flow
```
Operation Execution
     ↓
SessionID (YYYYMMDD_HHMMSS) created
     ↓
Operation-specific folder: daf_output/[operation]/[session_id]/
     ↓
Results saved to standard JSON format
     ↓
Metadata tracked in session_*.json
```

## Integration Points with CoGames

### Policy Loading
```python
from mettagrid.policy.policy import PolicySpec

spec = PolicySpec(class_path="lstm", data_path="checkpoint.pt")
policy = spec.load()
```

### Mission Configuration
```python
from cogames.game import load_mission_config

config = load_mission_config("training_facility_1")
```

### Environment Interaction
```python
from cogames.game import Game

game = Game.from_config(mission_config)
game.reset()
game.step(action)
```

### No Duplication Rules
- ✓ Use CoGames for policy management
- ✓ Use CoGames for mission loading
- ✓ Use CoGames for environment stepping
- ✗ Don't reimplement policy loading
- ✗ Don't recreate mission definitions
- ✗ Don't duplicate environment logic

## Configuration Pattern

All DAF operations use Pydantic configuration:

```python
from pydantic import BaseModel, Field, ConfigDict

class OperationConfig(BaseModel):
    """Configuration for operation."""
    
    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)
    
    name: str = Field(description="Operation name")
    output_dir: Path = Field(default=Path("./daf_output"))
    verbose: bool = Field(default=False, description="Enable verbose logging")
    
    @field_validator("output_dir", mode="before")
    @classmethod
    def ensure_path(cls, v: Any) -> Path:
        if isinstance(v, str):
            return Path(v)
        return v
```

## Session ID Convention

Session IDs follow this format: `YYYYMMDD_HHMMSS`

Example: `20241121_143022` (November 21, 2024, 14:30:22)

Advantages:
- Chronologically sortable
- Unique per second
- Human-readable
- Filesystem-safe
- No special characters

## Error Handling Strategy

### Explicit Errors
```python
# ✓ Good - explicit error with context
if not config_path.exists():
    raise FileNotFoundError(f"Config not found: {config_path}")
    
# ✗ Avoid - generic error
if not config_path.exists():
    raise Exception("File error")
```

### Error Logging
```python
# ✓ Good - log error with context
logger.error(f"Operation failed: {e}")
output_mgr.log_operation_complete("op", status="error")

# ✗ Avoid - silent failures
try:
    pass
except Exception:
    pass
```

## Scalability Considerations

### Single Machine
- Sequential operation execution
- File-based result storage
- Local logging

### Multi-Machine
- Distributed training coordination
- Checkpoint synchronization
- Centralized result aggregation

### Large Scale
- Result pagination
- Archived sessions
- Cleanup policies

## Testing Architecture

### Unit Tests
- Test individual modules in isolation
- Use pytest fixtures
- Mock external dependencies

### Integration Tests
- Test workflows across modules
- Use real CoGames missions (small)
- Verify output organization

### Test Organization
```
daf_output/evaluations/tests/
├── cogames/        # CoGames framework tests
├── daf/            # DAF module tests
└── test_report.json # Aggregated results
```

## Documentation Architecture

### Hierarchy
1. Quick Start (5 min) - What to do now
2. Architecture (10 min) - How it works
3. API Reference (reference) - What functions do
4. Examples (code) - Working samples
5. Troubleshooting (guide) - Common issues

### File Organization
```
daf/
├── README.md              # Module overview
├── GETTING_STARTED.md     # Beginner guide
├── IMPROVEMENTS_SUMMARY.md # What's new
└── docs/
    ├── OUTPUT_ORGANIZATION.md # Output structure
    ├── LOGGING.md            # Logging guide
    └── OUTPUT_AND_LOGGING_INDEX.md # Full index
```

## Development Workflow

### When Adding Features

1. **Design**
   - Follow existing patterns
   - Plan module responsibilities
   - Consider output organization

2. **Implement**
   - Use OutputManager for outputs
   - Use DAFLogger for logging
   - Add type hints and docstrings

3. **Test**
   - Write unit tests
   - Add integration tests
   - Verify output organization

4. **Document**
   - Add docstrings
   - Update relevant .md files
   - Add example if needed

5. **Review**
   - Check: types, docs, tests, patterns
   - Run linter: `ruff check daf/src/`
   - Run tests: `pytest daf/tests/ -v`

## Quality Standards

### Code Quality
- Type hints: Required
- Docstrings: Required for public APIs
- Linting: Zero errors (ruff)
- Tests: All modules tested

### Documentation Quality
- Clarity: Understand in one read
- Examples: Working code samples
- Completeness: All features documented
- Organization: Logical hierarchy

### Backward Compatibility
- No breaking API changes
- New features opt-in
- Default behavior preserved
- Migration path provided

---

**Status**: Production Ready ✅
**Last Updated**: November 21, 2024



