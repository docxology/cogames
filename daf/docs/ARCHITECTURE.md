# DAF Architecture and Design

Comprehensive overview of the Distributed Agent Framework design, patterns, and implementation details.

## Architecture: Sidecar Utility Pattern

**DAF is a sidecar utility and extension for CoGames.**

### Sidecar Principles

1. **Separation**: DAF lives in `daf/` folder, separate from core `src/cogames/`
2. **Extension, Not Replacement**: DAF extends CoGames functionality, never duplicates or replaces core methods
3. **Invocation Pattern**: All DAF functions invoke underlying CoGames methods
4. **Optional Dependency**: Core CoGames works without DAF; DAF requires CoGames
5. **Clear Boundaries**: All DAF functions use `daf_` prefix to distinguish from core methods

## Design Philosophy

DAF follows these core principles:

1. **Leverage Existing CoGames Infrastructure**: Wrap and extend cogames methods rather than reimplementing
2. **Clear Separation of Concerns**: Each module handles a specific aspect (training, sweeps, comparison, deployment)
3. **Modular and Composable**: Thin orchestrators chain modules into complete workflows
4. **Explicit Prefix Convention**: All DAF functions use `daf_` prefix for clarity
5. **Professional Documentation**: "Show don't tell" - clear examples, not verbose explanations
6. **TDD Approach**: Comprehensive tests alongside implementation
7. **Sidecar Pattern**: DAF extends CoGames, never replaces it

## Module Organization

### Layer 1: Configuration (`config.py`)

**Purpose**: Define all configuration schemas using Pydantic models

**Key Classes**:

- `DAFConfig` - Global settings for all DAF operations
- `DAFSweepConfig` - Hyperparameter sweep definitions
- `DAFDeploymentConfig` - Deployment pipeline settings (with tournament/season support)
- `DAFComparisonConfig` - Policy comparison setup
- `DAFPipelineConfig` - Multi-stage workflow definitions
- `DAFTournamentConfig` - *[NEW 2.1]* Tournament server and season settings
- `DAFVariantConfig` - *[NEW 2.1]* Mission variant configuration

**Design Pattern**:

- Pydantic models for validation and type safety
- YAML/JSON loading and saving
- Environment variable support via ConfigDict
- Follows `MettaGridConfig` patterns

### Layer 2: Validation (`environment_checks.py`)

**Purpose**: Verify environment is suitable before operations

**Key Components**:

- `EnvironmentCheckResult` - Container for check results
- `daf_check_cuda_availability()` - GPU/CUDA verification
- `daf_check_disk_space()` - Storage validation
- `daf_check_dependencies()` - Package availability
- `daf_check_mission_configs()` - Mission loadability
- `daf_check_auth()` - *[NEW 2.1]* Authentication token verification
- `daf_check_environment()` - Comprehensive check

**Design Pattern**:

- Accumulate results without early termination
- Generate both human-readable and structured output
- Use Rich console for professional formatting
- Non-blocking warnings with blocking errors

### Layer 3: Operation Modules

#### `distributed_training.py`

**Purpose**: Wrap cogames.train.train() with distributed coordination

**Key Components**:

- `DistributedTrainingResult` - Training outcome container
- `daf_launch_distributed_training()` - Multi-node orchestration
- `daf_create_training_cluster()` - Resource initialization
- `daf_aggregate_training_stats()` - Metrics collection
- `daf_get_training_status()` - Status tracking

**Design Pattern**:

- Single-node defaults to cogames.train.train() directly
- Multi-node defers to specific backends (Ray, Dask, torch.distributed)
- Checkpoint discovery via mettagrid.policy.loader
- Statistical aggregation across workers

#### `sweeps.py`

**Purpose**: Hyperparameter search with multiple strategies

**Key Components**:

- `SweepTrialResult` - Single trial outcome
- `SweepResult` - Complete sweep results container
- `daf_grid_search()` - Cartesian product of parameters
- `daf_random_search()` - Uniform random sampling
- `daf_launch_sweep()` - Sweep execution orchestrator
- `daf_sweep_best_config()` - Extract best configuration from results
- `daf_sweep_status()` - Load sweep status from disk
- `daf_variant_sweep()` - *[NEW 2.1]* Sweep over mission variants

**Design Pattern**:

- Grid/random search configuration generation
- Uses cogames.evaluate.evaluate() for assessment
- Collects per-mission and aggregated metrics
- Tracks trial-level hyperparameters and results
- JSON persistence with timestamp metadata

#### `comparison.py`

**Purpose**: Policy head-to-head evaluation with statistics

**Key Components**:

- `PolicyComparisonResult` - Two-policy comparison result
- `ComparisonReport` - Multi-policy comparison container
- `daf_compare_policies()` - Run comparison
- `daf_policy_ablation()` - Component ablation studies
- `daf_benchmark_suite()` - Standardized benchmarks
- `daf_compare_with_vor()` - *[NEW 2.1]* VOR-based policy comparison via `cogames.pickup`

**Design Pattern**:

- Uses scipy.stats for statistical tests (t-test, effect size)
- Per-mission and aggregated performance tracking
- Pairwise statistical significance testing
- Mission set presets from cogames.cogs_vs_clips.evals
- JSON export with complete results

#### `visualization.py`

**Purpose**: Plotting and report generation

**Key Components**:

- `daf_plot_training_curves()` - Training progress visualization
- `daf_plot_policy_comparison()` - Comparison plots
- `daf_plot_sweep_results()` - Sweep progress and best config
- `daf_export_comparison_html()` - Interactive HTML reports
- `daf_generate_leaderboard()` - Policy ranking tables

**Design Pattern**:

- Matplotlib for static plots (extending scripts/run_evaluation.py patterns)
- HTML generation without external template engines
- Optional if matplotlib not available (non-blocking)
- Professional styling with consistent aesthetics
- All plots saved as PNG with DPI 150

#### `deployment.py`

**Purpose**: Policy packaging, validation, and deployment

**Key Components**:

- `DeploymentResult` - Deployment operation outcome
- `daf_package_policy()` - Create portable bundles
- `daf_validate_deployment()` - Smoke testing
- `daf_deploy_policy()` - Upload to endpoint
- `daf_monitor_deployment()` - Performance tracking
- `daf_rollback_deployment()` - Version rollback
- `daf_submit_policy()` - *[NEW 2.1]* Tournament submission via `cogames.cli.submit`

**Design Pattern**:

- Tarball packaging with metadata.json
- Uses patterns from cogames.cli.submit
- Isolated validation using cogames.evaluate.evaluate()
- HTTP-based deployment (with httpx)
- Semantic versioning support

### Layer 4: Orchestrators (`orchestrators.py`)

**Purpose**: Chain modules into complete workflows

**Key Functions**:

- `daf_run_training_pipeline()` - Training workflow
- `daf_run_sweep_pipeline()` - Sweep workflow
- `daf_run_comparison_pipeline()` - Comparison workflow
- `daf_run_benchmark_pipeline()` - Benchmark workflow

**Design Pattern**:

- Stage-based pipeline with error handling
- Collect results from each stage
- Stop-on-failure or continue options
- Non-blocking warning vs. blocking errors
- Total execution time tracking
- Comprehensive result reporting

### Layer 5: v2.1 Integration Modules

#### `auth_integration.py` *[NEW 2.1]*

**Purpose**: OAuth2 authentication wrapper for DAF workflows

**Key Functions**: `daf_login()`, `daf_check_auth_status()`, `daf_get_auth_token()`

**Wraps**: `cogames.auth.BaseCLIAuthenticator`

#### `scoring_analysis.py` *[NEW 2.1]*

**Purpose**: Scoring utilities for VOR and weighted metrics

**Key Functions**: `daf_compute_vor()`, `daf_compute_weighted_score()`, `daf_allocate_agents()`, `daf_validate_proportions()`

**Wraps**: `metta_alo.scoring`

#### `policy_analysis.py` *[NEW 2.1]*

**Purpose**: Policy framework discovery and architecture analysis

**Key Functions**: `daf_list_available_policies()`, `daf_analyze_policy()`, `daf_compare_policy_architectures()`

**Wraps**: `cogames.policy.*`

### Layer 6: CLI Integration

**Location**: Extended in `src/cogames/main.py`

**Commands Added**:

```
cogames daf-check                    # Environment validation
cogames daf-train                    # Distributed training
cogames daf-sweep                    # Hyperparameter sweep
cogames daf-compare                  # Policy comparison
cogames daf-deploy                   # Policy deployment
cogames daf-validate                 # Pre-deployment validation
cogames daf-pipeline                 # Workflow orchestration
```

## Data Flow

### Sweep Workflow

```
DAFSweepConfig
    ↓
daf_launch_sweep()
    ├→ Generate configs (grid/random/bayesian)
    ├→ Load missions
    └→ For each config:
        ├→ Create PolicySpec
        ├→ Run cogames.evaluate.evaluate()
        ├→ Extract metrics
        └→ Store SweepTrialResult
    ↓
SweepResult
    ├→ Save JSON
    ├→ daf_plot_sweep_results()
    └→ daf_sweep_best_config()
```

### Comparison Workflow

```
PolicySpec list + Missions
    ↓
daf_compare_policies()
    ├→ Run cogames.evaluate.evaluate()
    ├→ Extract per-mission rewards
    ├→ Compute aggregates
    └→ Run scipy.stats tests
    ↓
ComparisonReport
    ├→ Save JSON
    ├→ daf_plot_policy_comparison()
    ├→ daf_export_comparison_html()
    └→ daf_generate_leaderboard()
```

### Training Pipeline

```
daf_run_training_pipeline()
    ├→ Stage 1: daf_check_environment()
    ├→ Stage 2: daf_launch_distributed_training()
    │   └→ Wraps cogames.train.train()
    ├→ Stage 3: (Optional) cogames.evaluate.evaluate()
    └→ PipelineResult with checkpoint path
```

## Integration Points with CoGames

**DAF integrates with CoGames as a sidecar utility - it invokes CoGames methods rather than duplicating them.**

### Direct Dependencies (CoGames Methods Invoked)

- **cogames.train.train()** - Core training implementation (wrapped by `daf_launch_distributed_training()`)
- **cogames.evaluate.evaluate()** - Evaluation engine for all assessments (used by sweeps and comparisons)
- **cogames.game** - Mission config loading/saving
- **cogames.cli.mission** - Mission discovery and loading (used throughout DAF)
- **cogames.device.resolve_training_device()** - Device resolution (CUDA/CPU) (used by environment checks)
- **cogames.auth** - Authentication for deployment
- **mettagrid** - Core simulation and policy infrastructure
- **mettagrid.policy.loader** - Policy initialization and checkpoint discovery

### Sidecar Integration Pattern

Each DAF module wraps or extends CoGames functionality:

- **distributed_training.py**: Wraps `cogames.train.train()` with distributed coordination
- **sweeps.py**: Uses `cogames.evaluate.evaluate()` for hyperparameter assessment
- **comparison.py**: Uses `cogames.evaluate.evaluate()` for policy comparison
- **environment_checks.py**: Uses `cogames.device.resolve_training_device()` for device validation
- **mission_analysis.py**: Uses `cogames.cli.mission.get_mission()` for discovery
- **deployment.py**: Uses `cogames.auth`, `cogames.cli.submit` for tournament submissions
- **auth_integration.py**: Wraps `cogames.auth.BaseCLIAuthenticator` for OAuth2
- **scoring_analysis.py**: Wraps `metta_alo.scoring` for VOR and weighted metrics
- **policy_analysis.py**: Wraps `cogames.policy.*` for architecture discovery

### Patterns Borrowed (Not Duplicated)

- **Pydantic models** for configuration (from MettaGridConfig)
- **Rich console** for formatted output (from all CLI commands)
- **Typer** for CLI argument handling (from main.py)
- **JSON/YAML persistence** (from game.py)
- **ThreadPoolExecutor** for parallelization (from scripts/run_evaluation.py)
- **Matplotlib plotting** (from scripts/run_evaluation.py)
- **Statistical testing** patterns (from comparison literature)

## Testing Strategy

### Test Hierarchy

1. **Unit Tests** (`test_daf_*.py`)
   - Config loading and validation
   - Individual function behavior
   - Error handling and edge cases

2. **Integration Tests** (within unit test files)
   - Full workflow execution
   - Data persistence and recovery
   - Module interaction

3. **Functional Tests** (planned for CI/CD)
   - End-to-end pipeline execution
   - Real policy evaluation
   - Performance benchmarks

### Test Coverage

- Configuration management: Full
- Environment checks: All checks + error paths
- Sweeps: Grid/random search, result analysis
- Comparison: Statistical tests, HTML generation
- Deployment: Packaging, validation, submission, error handling
- Orchestrators: Pipeline stage execution
- Auth Integration: Login, token, status checking
- Scoring Analysis: VOR, weighted scores, allocation
- Policy Analysis: Discovery, architecture analysis

## Performance Considerations

### Parallelization

- **Sweeps**: Parallel trial execution via ThreadPoolExecutor
- **Comparisons**: Policies evaluated sequentially per mission
- **Training**: Uses pufferlib's vectorized environments
- **Distributed**: Multi-node via Ray/Dask/torch.distributed

### Memory Management

- Checkpoint limiting (`max_checkpoints_to_keep`)
- Disk space validation before operations
- Lazy loading of mission configs
- Result streaming to JSON (not accumulated in memory)

### Scalability

- Grid search: O(n^k) where n=grid_points, k=params
- Random search: O(num_trials)
- Comparison: O(num_policies × num_missions × episodes)
- Multi-node: Linear scaling with worker count (ideal case)

## Error Handling Philosophy

**Non-Blocking vs. Blocking Errors**:

- **Warnings** (non-blocking):
  - Optional package unavailable → Skip feature
  - CUDA not available → Fall back to CPU
  - Visualization fails → Continue without plots
  
- **Errors** (blocking):
  - Required package missing → Exit with error
  - Mission config invalid → Cannot proceed
  - Training divergence detected → Stop training

**Result Reporting**:

- Accumulate all errors for comprehensive reporting
- Never early-exit on first error if possible
- Distinguish transient vs. permanent failures

## Future Extensibility

### Planned Enhancements

1. **Bayesian Optimization** - Full implementation with GPy/skopt
2. **Ray Integration** - Multi-machine distributed training
3. **Dask Integration** - Distributed array-based parallelization
4. **Live Dashboard** - Real-time training/sweep monitoring
5. **Experiment Tracking** - WandB/MLflow integration
6. **Advanced Ablations** - Component importance analysis
7. **Pareto Front** - Multi-objective optimization

### Extension Points

- **Custom sweep strategies**: Inherit from base class
- **Custom visualization**: Add to visualization.py
- **Custom metrics**: Extend comparison result handling
- **Custom backends**: Implement distributed_training variants

## Code Organization Best Practices

1. **Immutability**: Configuration objects are read-only after creation
2. **Composition**: Use dataclass containers for structured results
3. **Functional Style**: Pure functions preferred over classes where possible
4. **Explicit Typing**: Full type hints for all function signatures
5. **Docstrings**: Module-level, class-level, and function-level documentation
6. **Logging**: Use standard Python logging module throughout
7. **Rich Output**: Use Rich console for all user-facing output

## Security Considerations

1. **Authentication**: Use existing cogames.auth infrastructure
2. **File Paths**: Validate and sanitize all file operations
3. **HTTP**: Use HTTPS for deployment endpoints
4. **Secrets**: Never log authentication tokens or API keys
5. **Validation**: Always validate configuration before execution

## Maintenance Guidelines

1. **Keep modules focused**: Each module has single responsibility
2. **Update tests alongside code**: TDD discipline
3. **Document breaking changes**: Maintain backward compatibility
4. **Version external dependencies**: Lock versions in requirements
5. **Regular refactoring**: Simplify and reduce technical debt
