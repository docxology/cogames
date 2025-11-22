# DAF Source Modules: Agent & Policy Support

DAF source modules provide distributed training, evaluation, and analysis infrastructure for agent policies. All modules wrap underlying CoGames methods and follow the sidecar utility pattern.

## Policy Interface Requirements

DAF modules work with policies implementing these interfaces:

### MultiAgentPolicy (Training & Sweeps)

For batch processing during training:

```python
from mettagrid.policy.policy import MultiAgentPolicy
import numpy as np

class MyPolicy(MultiAgentPolicy):
    def __init__(self, env_interface, seed=42):
        super().__init__(env_interface)
        self.seed = seed
    
    def step_batch(self, observations: np.ndarray, actions: np.ndarray) -> None:
        """Process batch of observations.
        
        Args:
            observations: Shape (num_agents, num_tokens, 3)
            actions: Shape (num_agents,) - write action IDs here
        """
        num_agents = observations.shape[0]
        for agent_id in range(num_agents):
            actions[agent_id] = self.compute_action(observations[agent_id])
```

### AgentPolicy (Play & Evaluation)

For single-agent step-by-step control:

```python
from mettagrid.policy.policy import AgentPolicy
from mettagrid.types import Action, AgentObservation

class MySingleAgentPolicy(AgentPolicy):
    def step(self, obs: AgentObservation) -> Action:
        """Select action for single agent."""
        action_name = self.select_action(obs)
        return Action(name=action_name)
```

## Core Modules

### config.py

Configuration management using Pydantic models for all DAF operations.

**Key Classes:**
- `DAFConfig` - Global settings (output, resources, logging)
- `DAFSweepConfig` - Hyperparameter sweep configuration
- `DAFComparisonConfig` - Multi-policy comparison setup
- `DAFTrainingConfig` - Training orchestration settings
- `DAFDeploymentConfig` - Deployment pipeline configuration

**Example:**
```python
from daf.config import DAFSweepConfig

cfg = DAFSweepConfig.from_yaml("sweep_config.yaml")
cfg.policy_class_path = "lstm"
cfg.validate()
```

### sweeps.py

Hyperparameter search with grid, random, and Bayesian strategies.

**Key Functions:**
- `daf_launch_sweep()` - Execute hyperparameter sweep
- `daf_sweep_best_config()` - Get best configuration
- `daf_sweep_analysis()` - Analyze sweep results

**Supported Strategies:**
- `grid` - Exhaustive grid search
- `random` - Random sampling
- `bayesian` - Bayesian optimization

**Example:**
```python
from daf import sweeps, config

cfg = config.DAFSweepConfig.from_yaml("sweep.yaml")
results = sweeps.daf_launch_sweep(cfg)
best = sweeps.daf_sweep_best_config(results)
print(f"Best learning rate: {best.hyperparameters['learning_rate']}")
```

### comparison.py

Statistical policy comparison and benchmarking.

**Key Functions:**
- `daf_compare_policies()` - Multi-policy comparison
- `daf_benchmark_suite()` - Standardized benchmarks
- `daf_policy_ablation()` - Component ablation studies

**Output:**
- Summary statistics per policy/mission
- Pairwise significance tests
- HTML reports with visualizations

**Example:**
```python
from daf import comparison
from mettagrid.policy.policy import PolicySpec

results = comparison.daf_compare_policies(
    policies=[
        PolicySpec(class_path="lstm"),
        PolicySpec(class_path="baseline"),
        PolicySpec(class_path="random"),
    ],
    missions=["training_facility_1", "assembler_2"],
    episodes_per_mission=10
)
print(results.summary_statistics)
```

### distributed_training.py

Multi-node, multi-GPU training orchestration.

**Key Functions:**
- `daf_launch_distributed_training()` - Start distributed training
- `daf_monitor_training_job()` - Monitor job progress
- `daf_collect_training_results()` - Gather results

**Features:**
- Automatic GPU distribution
- Checkpoint synchronization
- Memory monitoring
- Failure recovery

**Example:**
```python
from daf import distributed_training

job = distributed_training.daf_launch_distributed_training(
    env_cfg=env_cfg,
    policy_class_path="lstm",
    num_steps=1_000_000,
    num_machines=4,
)
checkpoint = job.wait_for_completion()
```

### mission_analysis.py

Per-mission performance analysis and failure mode detection.

**Key Functions:**
- `daf_analyze_mission()` - Analyze single mission
- `daf_analyze_mission_set()` - Batch analysis
- `daf_discover_missions_from_readme()` - Auto-discovery
- `daf_get_mission_metadata()` - Retrieve mission info
- `daf_validate_mission_set()` - Validate missions exist

**Analysis Output:**
- Success rate
- Mean/variance reward
- Failure modes
- Resource usage

**Example:**
```python
from daf import mission_analysis

analysis = mission_analysis.daf_analyze_mission(
    mission_name="assembler_2"
)
print(f"Success rate: {analysis.success_rate}")
print(f"Mean reward: {analysis.mean_reward}")
```

### environment_checks.py

Pre-flight validation and device optimization.

**Key Functions:**
- `daf_check_environment()` - Full environment validation
- `daf_check_cuda()` - CUDA availability and version
- `daf_check_disk_space()` - Disk availability
- `daf_check_dependencies()` - Package verification
- `daf_check_mission_configs()` - Mission availability
- `daf_get_recommended_device()` - Device selection

**Checks:**
- CUDA/GPU availability
- Disk space for checkpoints
- Required package imports
- Mission configuration files
- Memory availability

**Example:**
```python
from daf import environment_checks

result = environment_checks.daf_check_environment(
    missions=["training_facility_1"],
    check_cuda=True,
    check_disk=True
)

if result.is_healthy():
    device = environment_checks.daf_get_recommended_device()
    print(f"Using device: {device}")
else:
    print("Fix issues:", result.warnings)
```

### visualization.py

HTML reports and performance dashboards.

**Key Functions:**
- `daf_export_comparison_html()` - Generate comparison report
- `daf_plot_training_curves()` - Plot training progress
- `daf_plot_policy_comparison()` - Compare policy performance

**Output:**
- Interactive HTML reports
- Performance curves
- Statistical summaries
- Agent behavior analysis

**Example:**
```python
from daf import visualization, comparison

results = comparison.daf_compare_policies(...)
visualization.daf_export_comparison_html(
    results,
    output_path="comparison_report.html"
)
```

### deployment.py

Policy packaging and deployment pipeline.

**Key Functions:**
- `daf_package_policy()` - Create deployment package
- `daf_validate_deployment()` - Validate package
- `daf_deploy_policy()` - Submit to deployment endpoint
- `daf_rollback_deployment()` - Revert deployment

**Package Contents:**
- Policy weights/checkpoint
- Metadata and configuration
- Documentation and requirements
- Version information

**Example:**
```python
from daf import deployment

result = deployment.daf_package_policy(
    policy_class_path="lstm",
    weights_path="checkpoints/best_model.pt",
    output_dir="deployment_packages"
)
```

### logging_config.py

Structured logging with metrics tracking and operation monitoring.

**Key Classes:**
- `DAFLogger` - Structured logger with context
- `OperationTracker` - Track operation metrics
- `MetricsCollector` - Collect and aggregate metrics

**Features:**
- Operation tracking
- Metrics aggregation
- Session metadata
- Rich console output

**Example:**
```python
from daf.logging_config import create_daf_logger

logger = create_daf_logger("my_experiment")

with logger.track_operation("training"):
    logger.info("Starting training...")
    # Your code here
    
logger.print_metrics_summary()
```

### output_manager.py

Centralized output organization with structured subfolders.

**Key Classes:**
- `OutputDirectories` - Manage output folder structure
- `OutputManager` - Centralized output coordination

**Output Structure:**
```
daf_output/
├── sweeps/          # Sweep results by session
├── comparisons/     # Comparison reports
├── training/        # Training runs
├── deployment/      # Deployment packages
├── evaluations/     # Test results
├── visualizations/  # Generated plots/HTML
├── logs/            # Session logs
├── reports/         # Generated reports
└── artifacts/       # Reusable artifacts
```

**Example:**
```python
from daf.output_manager import get_output_manager

output_mgr = get_output_manager(base_dir="./daf_output")
output_mgr.print_output_info()

output_mgr.save_json_results(
    results,
    operation="sweep",
    filename="results",
)
```

### orchestrators/ Module

High-level workflow orchestration (see `orchestrators/AGENTS.md`).

## Integration Patterns

### Pattern 1: Environment Validation

Always validate environment before operations:

```python
from daf import environment_checks

result = environment_checks.daf_check_environment()
assert result.is_healthy(), "Environment check failed"
```

### Pattern 2: Configuration & Execution

Use configuration for reproducibility:

```python
from daf.config import DAFSweepConfig
from daf import sweeps

cfg = DAFSweepConfig.from_yaml("sweep.yaml")
cfg.seed = 42  # Deterministic
results = sweeps.daf_launch_sweep(cfg)
```

### Pattern 3: Result Analysis & Export

Analyze results and generate reports:

```python
from daf import comparison, visualization

results = comparison.daf_compare_policies(...)
visualization.daf_export_comparison_html(results)
```

## Best Practices

1. **Always check environment first**
   ```python
   from daf import environment_checks
   result = environment_checks.daf_check_environment()
   assert result.is_healthy()
   ```

2. **Use configuration files for reproducibility**
   ```python
   from daf.config import DAFSweepConfig
   cfg = DAFSweepConfig.from_yaml("config.yaml")
   ```

3. **Support deterministic seeding**
   ```python
   cfg.seed = 42
   results = daf_launch_sweep(cfg)
   ```

4. **Minimize policy global state**
   ```python
   # GOOD: Instance variables
   class MyPolicy(MultiAgentPolicy):
       def __init__(self, env_interface):
           self.network = MyNetwork()
   
   # AVOID: Global state
   global_network = None  # Don't do this!
   ```

5. **Validate inputs before execution**
   ```python
   from pathlib import Path
   ckpt = Path("checkpoints/model.pt")
   assert ckpt.exists(), f"Checkpoint not found: {ckpt}"
   ```

## Troubleshooting

### Policy Import Errors

```python
# Error: ModuleNotFoundError
# Solution: Ensure policy module is in PYTHONPATH
import sys
sys.path.insert(0, "/path/to/policy/package")
```

### Checkpoint Loading Issues

```python
# Error: Could not load checkpoint
# Solution: Verify path and format
from pathlib import Path
ckpt = Path("checkpoints/model.pt")
if not ckpt.exists():
    raise FileNotFoundError(f"Not found: {ckpt}")
```

### Memory Issues During Sweeps

```python
# Solution: Reduce parallel jobs or episodes
from daf.config import DAFConfig
cfg = DAFConfig(max_parallel_jobs=2)

# Or reduce episodes per trial
sweep_cfg.episodes_per_trial = 1
```

## See Also

- `README.md` - Module overview and quick start
- `orchestrators/AGENTS.md` - Workflow orchestration
- `../AGENTS.md` - Top-level agent architecture
- `../docs/API.md` - Complete API reference
- `../docs/ARCHITECTURE.md` - System architecture



