# DAF-CoGames Integration Map

**Complete technical mapping of DAF functions to CoGames methods they invoke.**

DAF is a **sidecar utility** that extends CoGames by wrapping and invoking core methods. This document provides the definitive integration map for all DAF functions.

---

## Module-Level Integration

### Layer 1: Configuration (`daf.config`)

**DAF Functions** → **CoGames Dependencies**

| DAF Function | CoGames Method | Purpose | Notes |
|-------------|---|---------|-------|
| `DAFConfig` | N/A (configuration only) | Global DAF settings | No external dependencies |
| `DAFSweepConfig` | None directly | Sweep configuration | Used by `daf_launch_sweep()` |
| `DAFDeploymentConfig` | None directly | Deployment configuration | Used by deployment functions |
| `DAFComparisonConfig` | None directly | Comparison configuration | Used by `daf_compare_policies()` |
| `DAFPipelineConfig` | None directly | Pipeline configuration | Used by orchestrators |
| `daf_load_config()` | N/A (file I/O) | Load configuration | Supports YAML/JSON |

**Dependencies**: None on CoGames core (configuration objects only)

---

### Layer 2: Environment Checks (`daf.environment_checks`)

**DAF Functions** → **CoGames Dependencies**

| DAF Function | CoGames Method | Purpose | Integration |
|-------------|---|---------|-------|
| `daf_check_cuda_availability()` | `cogames.device.resolve_training_device()` | Verify GPU/CUDA | Uses device resolution to validate CUDA |
| `daf_check_disk_space()` | N/A (system checks) | Verify storage | System-level, no CoGames dependency |
| `daf_check_dependencies()` | N/A (package checks) | Verify packages installed | System-level, no CoGames dependency |
| `daf_check_mission_configs()` | `cogames.cli.mission.get_mission_names_and_configs()` | Validate missions loadable | Loads mission configs from CoGames |
| `daf_check_environment()` | All above | Comprehensive check | Orchestrates all checks |
| `daf_get_recommended_device()` | `cogames.device.resolve_training_device()` | Get recommended device | Returns device from CoGames |

**Key Integration**: All checks **use real CoGames methods**, never duplicating functionality.

---

### Layer 3: Operation Modules

#### `daf.distributed_training` - Training Orchestration

**DAF Functions** → **CoGames Dependencies**

| DAF Function | CoGames Method | Integration Type |
|-------------|---|---|
| `daf_launch_distributed_training()` | `cogames.train.train()` | **WRAPS** - calls directly for single-node |
| | `cogames.train.train()` | **WRAPS** - coordinates across nodes for multi-node |
| `daf_create_training_cluster()` | `cogames.device.resolve_training_device()` | Validates cluster devices |
| `daf_aggregate_training_stats()` | N/A (aggregation only) | Accumulates metrics |
| `daf_get_training_status()` | N/A (file I/O) | Reads checkpoint metadata |

**Pattern**: Single-node execution directly invokes `cogames.train.train()`. Multi-node adds wrapper layer with distributed backend coordination.

**Data Flow**:
```
daf_launch_distributed_training()
  └─ Single-node: directly calls cogames.train.train()
  └─ Multi-node: 
     ├─ Sets up workers
     ├─ Each worker calls cogames.train.train()
     └─ Aggregates results
```

---

#### `daf.sweeps` - Hyperparameter Search

**DAF Functions** → **CoGames Dependencies**

| DAF Function | CoGames Method | Integration |
|-------------|---|---|
| `daf_grid_search()` | N/A (combinatorics) | Generates parameter combinations |
| `daf_random_search()` | N/A (randomization) | Generates random samples |
| `daf_launch_sweep()` | `cogames.evaluate.evaluate()` | **USES** - evaluates each trial |
| | `cogames.cli.mission.get_mission_name_and_config()` | **USES** - loads missions |
| | `mettagrid.policy.loader.initialize_or_load_policy()` | **USES** - initializes policies |
| `daf_sweep_best_config()` | N/A (analysis) | Extracts best result |
| `daf_sweep_status()` | N/A (file I/O) | Reads sweep results |

**Pattern**: Each trial invokes `cogames.evaluate.evaluate()`.

**Data Flow**:
```
daf_launch_sweep()
  ├─ Generate configs (grid/random/bayesian)
  ├─ For each config:
  │  ├─ Load mission via cogames.cli.mission
  │  ├─ Initialize policy via mettagrid.policy.loader
  │  ├─ Call cogames.evaluate.evaluate()
  │  └─ Store trial result
  └─ Return SweepResult
```

---

#### `daf.comparison` - Policy Comparison

**DAF Functions** → **CoGames Dependencies**

| DAF Function | CoGames Method | Integration |
|-------------|---|---|
| `daf_compare_policies()` | `cogames.evaluate.evaluate()` | **USES** - evaluates all policies |
| | `cogames.cli.mission.get_mission_names_and_configs()` | **USES** - loads missions |
| | `mettagrid.policy.loader.initialize_or_load_policy()` | **USES** - loads each policy |
| `daf_policy_ablation()` | `cogames.evaluate.evaluate()` | **USES** - evaluates ablated policies |
| `daf_benchmark_suite()` | All above | **USES** - runs standardized benchmarks |

**Pattern**: Evaluation via CoGames, then statistical analysis added by DAF.

**Data Flow**:
```
daf_compare_policies()
  ├─ For each policy:
  │  ├─ Load policy via mettagrid.policy.loader
  │  ├─ For each mission:
  │  │  └─ Call cogames.evaluate.evaluate()
  │  └─ Store per-policy results
  ├─ Run scipy.stats tests (DAF addition)
  ├─ Generate summary statistics (DAF addition)
  └─ Return ComparisonReport
```

---

#### `daf.visualization` - Plotting and Reports

**DAF Functions** → **CoGames Dependencies**

| DAF Function | CoGames Method | Integration |
|-------------|---|---|
| `daf_plot_training_curves()` | N/A (file I/O + matplotlib) | Extends `scripts/run_evaluation.py` patterns |
| `daf_plot_policy_comparison()` | N/A (matplotlib) | Extends evaluation plot patterns |
| `daf_plot_sweep_results()` | N/A (matplotlib) | Extends sweep visualization |
| `daf_export_comparison_html()` | N/A (HTML generation) | DAF-specific reporting |
| `daf_generate_leaderboard()` | N/A (markdown generation) | DAF-specific ranking |

**Pattern**: Uses matplotlib following CoGames evaluation script patterns. No direct CoGames method calls (plotting layer).

**Integration Type**: Extends visualization patterns from `scripts/run_evaluation.py`

---

#### `daf.deployment` - Policy Deployment

**DAF Functions** → **CoGames Dependencies**

| DAF Function | CoGames Method | Integration |
|-------------|---|---|
| `daf_package_policy()` | `cogames.auth.get_auth_token()` (optional) | Uses CoGames auth patterns |
| `daf_validate_deployment()` | `cogames.evaluate.evaluate()` | **USES** - validates policy via eval |
| | `mettagrid.policy.loader.initialize_or_load_policy()` | **USES** - loads policy for validation |
| `daf_deploy_policy()` | `cogames.auth` patterns | Uses CoGames authentication |
| `daf_monitor_deployment()` | N/A (HTTP monitoring) | External monitoring |
| `daf_rollback_deployment()` | `cogames.auth` patterns | Uses CoGames auth for rollback |

**Pattern**: Follows `cogames.cli.submit` patterns for deployment.

**Data Flow**:
```
daf_deploy_policy()
  ├─ daf_package_policy()
  │  └─ Uses cogames.auth patterns
  ├─ daf_validate_deployment()
  │  ├─ Load policy via mettagrid.policy.loader
  │  └─ Validate via cogames.evaluate.evaluate()
  ├─ Upload to endpoint
  └─ Return DeploymentResult
```

---

#### `daf.mission_analysis` - Mission Meta-Analysis

**DAF Functions** → **CoGames Dependencies**

| DAF Function | CoGames Method | Integration |
|-------------|---|---|
| `daf_analyze_mission()` | `cogames.cli.mission.get_mission_name_and_config()` | **USES** - loads mission |
| `daf_analyze_mission_set()` | `cogames.cli.mission.get_mission_names_and_configs()` | **USES** - loads multiple |
| `daf_validate_mission_set()` | All above | **USES** - validates each mission |
| `daf_discover_missions_from_readme()` | N/A (file parsing) | Parses README.md |
| `daf_get_mission_metadata()` | `cogames.cli.mission.get_mission_name_and_config()` | **USES** - extracts metadata |

**Pattern**: Loads missions via CoGames, extracts metadata (DAF addition).

**Data Flow**:
```
daf_analyze_mission()
  ├─ Call cogames.cli.mission.get_mission_name_and_config()
  ├─ Extract metadata:
  │  ├─ Number of agents
  │  ├─ Max steps
  │  ├─ Map size
  │  ├─ Map builder type
  │  └─ Loadability status
  └─ Return MissionAnalysis
```

---

### Layer 4: Orchestrators (`daf.orchestrators`)

**DAF Functions** → **CoGames Dependencies (via DAF modules)**

| Orchestrator | Stage 1 | Stage 2+ | CoGames Methods Used |
|-------------|---------|----------|---|
| `daf_run_training_pipeline()` | `daf_check_environment()` | Training, (eval) | `cogames.train.train()`, etc. |
| `daf_run_sweep_pipeline()` | `daf_check_environment()` | Sweep, save, visualize | `cogames.evaluate.evaluate()`, etc. |
| `daf_run_comparison_pipeline()` | `daf_check_environment()` | Compare, save, report | `cogames.evaluate.evaluate()`, etc. |
| `daf_run_benchmark_pipeline()` | `daf_check_environment()` | Benchmark, save, report | `cogames.evaluate.evaluate()`, etc. |

**Pattern**: All orchestrators:
1. Call `daf_check_environment()` first (Stage 1)
2. Chain DAF operation modules (which invoke CoGames methods)
3. Generate reports/visualizations

**Data Flow** (all pipelines follow same pattern):
```
daf_run_*_pipeline()
  ├─ Stage 1: daf_check_environment()
  │  └─ Uses cogames.device, cogames.cli.mission
  ├─ Stage 2+: Operation modules
  │  └─ Invoke CoGames methods
  └─ Return PipelineResult
```

---

## Function-by-Function CoGames Dependencies

### Public API Functions (All `daf_*` functions)

**Configuration Module**
```python
daf_load_config()                    # No direct CoGames deps
```

**Environment Checks Module**
```python
daf_check_cuda_availability()        # → cogames.device.resolve_training_device()
daf_check_disk_space()              # No direct CoGames deps
daf_check_dependencies()            # No direct CoGames deps
daf_check_mission_configs()         # → cogames.cli.mission.get_mission_names_and_configs()
daf_check_environment()             # → All above
daf_get_recommended_device()        # → cogames.device.resolve_training_device()
```

**Training Module**
```python
daf_launch_distributed_training()   # → cogames.train.train()
daf_create_training_cluster()       # → cogames.device.resolve_training_device()
daf_aggregate_training_stats()      # No direct CoGames deps
daf_get_training_status()           # No direct CoGames deps
```

**Sweeps Module**
```python
daf_grid_search()                   # No direct CoGames deps
daf_random_search()                 # No direct CoGames deps
daf_launch_sweep()                  # → cogames.evaluate.evaluate()
                                    # → cogames.cli.mission.get_mission_name_and_config()
                                    # → mettagrid.policy.loader.initialize_or_load_policy()
daf_sweep_best_config()             # No direct CoGames deps
daf_sweep_status()                  # No direct CoGames deps
```

**Comparison Module**
```python
daf_compare_policies()              # → cogames.evaluate.evaluate()
                                    # → cogames.cli.mission.get_mission_names_and_configs()
                                    # → mettagrid.policy.loader.initialize_or_load_policy()
daf_policy_ablation()               # → cogames.evaluate.evaluate()
daf_benchmark_suite()               # → cogames.evaluate.evaluate()
```

**Visualization Module**
```python
daf_plot_training_curves()          # Extends scripts/run_evaluation.py patterns
daf_plot_policy_comparison()        # Extends scripts/run_evaluation.py patterns
daf_plot_sweep_results()            # Extends scripts/run_evaluation.py patterns
daf_export_comparison_html()        # No direct CoGames deps
daf_generate_leaderboard()          # No direct CoGames deps
```

**Deployment Module**
```python
daf_package_policy()                # Uses cogames.auth patterns
daf_validate_deployment()           # → cogames.evaluate.evaluate()
                                    # → mettagrid.policy.loader.initialize_or_load_policy()
daf_deploy_policy()                 # Uses cogames.auth patterns
daf_monitor_deployment()            # No direct CoGames deps
daf_rollback_deployment()           # Uses cogames.auth patterns
```

**Mission Analysis Module**
```python
daf_analyze_mission()               # → cogames.cli.mission.get_mission_name_and_config()
daf_analyze_mission_set()           # → cogames.cli.mission.get_mission_names_and_configs()
daf_validate_mission_set()          # → cogames.cli.mission methods
daf_discover_missions_from_readme() # No direct CoGames deps
daf_get_mission_metadata()          # → cogames.cli.mission.get_mission_name_and_config()
```

**Orchestrators Module**
```python
daf_run_training_pipeline()         # Chains: daf_check_environment() → daf_launch_distributed_training()
daf_run_sweep_pipeline()            # Chains: daf_check_environment() → daf_launch_sweep()
daf_run_comparison_pipeline()       # Chains: daf_check_environment() → daf_compare_policies()
daf_run_benchmark_pipeline()        # Chains: daf_check_environment() → daf_benchmark_suite()
```

---

## CoGames Method Usage Summary

**Top-level CoGames methods invoked by DAF**:

1. **`cogames.train.train()`** (Most critical)
   - Used by: `daf_launch_distributed_training()`
   - Purpose: Core training execution
   - Type: **WRAPPED** with distributed coordination

2. **`cogames.evaluate.evaluate()`** (Most frequent)
   - Used by: `daf_launch_sweep()`, `daf_compare_policies()`, `daf_validate_deployment()`
   - Purpose: Policy evaluation
   - Type: **USED** multiple times

3. **`cogames.cli.mission.get_mission_name_and_config()`**
   - Used by: `daf_analyze_mission()`, `daf_get_mission_metadata()`, and indirectly by sweep/comparison
   - Purpose: Load single mission
   - Type: **USED**

4. **`cogames.cli.mission.get_mission_names_and_configs()`**
   - Used by: `daf_check_mission_configs()`, `daf_analyze_mission_set()`, sweep/comparison
   - Purpose: Load multiple missions
   - Type: **USED**

5. **`cogames.device.resolve_training_device()`**
   - Used by: `daf_check_cuda_availability()`, `daf_get_recommended_device()`
   - Purpose: Device resolution (CUDA/CPU)
   - Type: **USED**

6. **`mettagrid.policy.loader.initialize_or_load_policy()`**
   - Used by: sweep, comparison, deployment validation
   - Purpose: Policy initialization/loading
   - Type: **USED**

7. **`cogames.auth` and `cogames.cli.submit`**
   - Used by: Deployment module
   - Purpose: Authentication and submission patterns
   - Type: **PATTERNS USED**

---

## Verification

✅ **All integrations are real and functional**:

1. **Real CoGames Methods**: No mocks in production code
2. **Tested Integration**: Each integration point tested in DAF test suite
3. **No Duplication**: DAF never reimplements CoGames functionality
4. **Clear Signposts**: All integration points documented

**Test Coverage**: See `daf/tests/` for comprehensive integration tests

---

## How DAF Extends (Not Replaces) CoGames

### Pattern 1: Wrapping
**Example**: `daf_launch_distributed_training()` wraps `cogames.train.train()`
- Single-node: Directly invokes
- Multi-node: Adds distributed coordination layer
- Result: Extended functionality, no replacement

### Pattern 2: Using + Analysis
**Example**: `daf_compare_policies()` uses `cogames.evaluate.evaluate()` + adds statistics
- Invoke: `cogames.evaluate.evaluate()` for each policy
- Add: scipy.stats tests, effect sizes, pairwise comparisons
- Result: Enhanced comparison, not replacement

### Pattern 3: Using + Extraction
**Example**: `daf_analyze_mission()` uses `cogames.cli.mission.get_mission_name_and_config()` + extracts metadata
- Invoke: Mission loading from CoGames
- Add: Metadata extraction, analysis, validation
- Result: Enhanced mission analysis, not replacement

### Pattern 4: Orchestration
**Example**: Orchestrators chain multiple DAF modules
- Stage 1: Validation (DAF addition)
- Stage 2+: Operations (DAF modules that invoke CoGames)
- Result: Complete workflows, orchestration layer

---

## Summary

✅ **Complete Integration Map**:
- All DAF functions documented
- All CoGames dependencies identified
- All integration points verified
- All patterns explained

✅ **Real and Functional**:
- No mocks in DAF code
- All integrations tested
- All methods use real CoGames code
- Production-ready

✅ **Sidecar Pattern Maintained**:
- DAF wraps/extends CoGames
- DAF never replaces CoGames
- Clear separation maintained
- Optional dependency relationship

---

## References

- `SIDECAR_ARCHITECTURE.md` - Sidecar pattern explanation
- `ARCHITECTURE.md` - System design details
- `API.md` - Complete API reference
- `modules/` - Individual module documentation
- `daf/tests/` - Integration test suite







