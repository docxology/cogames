# DAF Orchestrators: Agent & Policy Workflow Orchestration

Orchestrators chain DAF modules into complete multi-stage workflows for agent development. Each orchestrator is a thin wrapper that sequences operations and handles stage transitions.

## Policy Workflow Patterns

Orchestrators enable workflows where policies flow through multiple stages:

```
Policy → Training → Evaluation → Sweeps → Comparison → Deployment
```

Each stage invokes underlying CoGames methods via DAF modules.

## Policy Interface Requirements

Orchestrators work with policies implementing these interfaces:

### MultiAgentPolicy (Training Stages)

Required for training and sweep stages:

```python
from mettagrid.policy.policy import MultiAgentPolicy

class MyPolicy(MultiAgentPolicy):
    def __init__(self, env_interface, seed=42):
        super().__init__(env_interface)
        self.seed = seed
    
    def step_batch(self, observations, actions):
        """Batch processing for training."""
        # Policy implementation
```

### AgentPolicy (Evaluation Stages)

Required for comparison and evaluation stages:

```python
from mettagrid.policy.policy import AgentPolicy

class MySingleAgentPolicy(AgentPolicy):
    def step(self, obs):
        """Single-agent step for evaluation."""
        # Policy implementation
```

## Core Orchestrators

### Training Pipeline

Executes a single training run with optional evaluation.

**Stages:**
1. Environment validation
2. Training execution
3. Checkpoint saving
4. (Optional) Evaluation on test missions

**Usage:**
```python
from daf.orchestrators import daf_run_training_pipeline

result = daf_run_training_pipeline(
    policy_class_path="lstm",
    missions=["training_facility_1"],
    num_training_steps=1_000_000,
    eval_missions=["assembler_2"],
    eval_episodes=10,
)

print(f"Training checkpoint: {result.outputs['final_checkpoint']}")
```

**Output:**
- Training checkpoint
- Evaluation results (if enabled)
- Session metadata

### Sweep Pipeline

Executes hyperparameter search with optional comparison.

**Stages:**
1. Environment validation
2. Sweep execution
3. Best config extraction
4. (Optional) Comparison of best configs

**Usage:**
```python
from daf.orchestrators import daf_run_sweep_pipeline
from daf.config import DAFSweepConfig

cfg = DAFSweepConfig.from_yaml("sweep.yaml")

result = daf_run_sweep_pipeline(
    config=cfg,
    compare_best=True,
    comparison_episodes=10,
)

best = result.outputs['best_config']
print(f"Optimal learning rate: {best.hyperparameters['learning_rate']}")
```

**Output:**
- Sweep results with all trials
- Best configuration
- Comparison results (if enabled)

### Comparison Pipeline

Evaluates multiple policies with statistical analysis.

**Stages:**
1. Environment validation
2. Policy comparison
3. Statistical significance testing
4. Report generation

**Usage:**
```python
from daf.orchestrators import daf_run_comparison_pipeline
from mettagrid.policy.policy import PolicySpec

policies = [
    PolicySpec(class_path="lstm", data_path="ckpt/v1.pt"),
    PolicySpec(class_path="lstm", data_path="ckpt/v2.pt"),
    PolicySpec(class_path="baseline"),
]

result = daf_run_comparison_pipeline(
    policies=policies,
    missions=["training_facility_1", "assembler_2"],
    episodes_per_mission=5,
)

print(result.outputs['summary_statistics'])
```

**Output:**
- Per-policy, per-mission scores
- Statistical significance tests
- HTML comparison report

### Benchmark Pipeline

Complete workflow from training through deployment.

**Stages:**
1. Environment validation
2. Training (if policy not provided)
3. Evaluation across test missions
4. Comparison with baselines (optional)
5. Deployment package creation

**Usage:**
```python
from daf.orchestrators import daf_run_benchmark_pipeline

result = daf_run_benchmark_pipeline(
    policies=["lstm"],
    missions=["training_facility_1", "assembler_2"],
    episodes_per_mission=10,
    create_deployment_package=True,
)

if result.is_success():
    print(f"Deployment package: {result.outputs['deployment_package']}")
```

**Output:**
- Evaluation results
- Performance comparison
- Deployment package (optional)

## Pipeline Result Objects

All orchestrators return `PipelineResult` with consistent interface:

```python
@dataclass
class PipelineResult:
    pipeline_name: str              # Name of executed pipeline
    status: str                     # "success", "failed", "partial"
    stages_completed: list[str]     # List of successful stages
    stages_failed: list[str]        # List of failed stages
    errors: list[str]               # Error messages
    outputs: dict[str, Any]         # Stage outputs
    total_time_seconds: float       # Total execution time
    timestamp: datetime             # Execution timestamp
    
    def is_success(self) -> bool:
        """Check if pipeline succeeded."""
        return self.status == "success"
```

## Workflow Composition

### Multi-Stage Pipeline

Chain multiple pipeline steps:

```python
from daf.orchestrators import (
    daf_run_training_pipeline,
    daf_run_sweep_pipeline,
    daf_run_comparison_pipeline,
)

# Stage 1: Train initial policy
train_result = daf_run_training_pipeline(
    policy_class_path="lstm",
    missions=["training_facility_1"],
    num_training_steps=1_000_000,
)

# Stage 2: Optimize hyperparameters
sweep_cfg = DAFSweepConfig(
    policy_class_path="lstm",
    policy_data_path=train_result.outputs['final_checkpoint'],
    missions=["training_facility_2"],
    search_space={
        'learning_rate': [0.0001, 0.001, 0.01],
    }
)

sweep_result = daf_run_sweep_pipeline(config=sweep_cfg)

# Stage 3: Compare with baselines
comparison_result = daf_run_comparison_pipeline(
    policies=[
        PolicySpec(
            class_path="lstm",
            data_path=sweep_result.outputs['best_checkpoint']
        ),
        PolicySpec(class_path="baseline"),
    ],
    missions=["training_facility_1", "assembler_2"],
    episodes_per_mission=10,
)
```

### Conditional Workflows

Handle pipeline failures gracefully:

```python
result = daf_run_benchmark_pipeline(...)

if result.is_success():
    print("All stages succeeded")
    checkpoint = result.outputs['final_checkpoint']
elif result.status == "partial":
    print(f"Partial success. Completed: {result.stages_completed}")
    print(f"Failed stages: {result.stages_failed}")
else:
    print(f"Pipeline failed: {result.errors}")
```

## Integration with Custom Policies

### Example: Custom Policy Training → Evaluation

```python
from daf.orchestrators import daf_run_training_pipeline
from daf.config import DAFTrainingConfig

# Your custom policy
class MyCustomPolicy(MultiAgentPolicy):
    def __init__(self, env_interface, seed=42):
        super().__init__(env_interface)
        self.network = MyNetwork()
    
    def step_batch(self, observations, actions):
        # Implementation
        pass

# Train via orchestrator
train_cfg = DAFTrainingConfig(
    policy_class_path="mypackage.MyCustomPolicy",
    missions=["training_facility_1"],
    num_training_steps=1_000_000,
)

result = daf_run_training_pipeline(config=train_cfg)
```

## Error Handling

Orchestrators provide detailed error information:

```python
result = daf_run_sweep_pipeline(config=sweep_cfg)

if not result.is_success():
    print(f"Status: {result.status}")
    print(f"Completed stages: {result.stages_completed}")
    print(f"Failed stages: {result.stages_failed}")
    
    for error in result.errors:
        print(f"Error: {error}")
```

## Performance Considerations

### Parallel vs Sequential

By default, orchestrators run stages sequentially:

```python
result = daf_run_sweep_pipeline(
    config=sweep_cfg,
    parallel_stages=False,  # Sequential (default)
)
```

For independent stages, enable parallelization:

```python
result = daf_run_comparison_pipeline(
    policies=policies,
    missions=missions,
    parallel_evaluation=True,  # Parallel policy evaluation
)
```

### Resource Management

Control resource usage:

```python
from daf.config import DAFConfig

cfg = DAFConfig(
    max_parallel_jobs=4,        # Parallel job limit
    gpu_memory_fraction=0.8,    # GPU memory limit
)

result = daf_run_benchmark_pipeline(
    config=cfg,
    ...
)
```

## Best Practices

1. **Always start with environment validation**
   ```python
   from daf import environment_checks
   result = environment_checks.daf_check_environment()
   assert result.is_healthy()
   ```

2. **Use configuration files for reproducibility**
   ```python
   from daf.config import DAFSweepConfig
   cfg = DAFSweepConfig.from_yaml("sweep.yaml")
   result = daf_run_sweep_pipeline(config=cfg)
   ```

3. **Check pipeline status before using outputs**
   ```python
   if result.is_success():
       checkpoint = result.outputs['final_checkpoint']
   else:
       raise RuntimeError(f"Pipeline failed: {result.errors}")
   ```

4. **Monitor execution time and resources**
   ```python
   print(f"Execution time: {result.total_time_seconds}s")
   print(f"Completed stages: {result.stages_completed}")
   ```

5. **Test on small scale before full pipeline**
   ```python
   # Test with 1 episode first
   cfg.episodes_per_trial = 1
   result = daf_run_sweep_pipeline(config=cfg)
   
   # Then increase for full sweep
   cfg.episodes_per_trial = 10
   ```

## Troubleshooting

### Pipeline Fails at Specific Stage

```python
result = daf_run_benchmark_pipeline(...)

if not result.is_success():
    failed_stage = result.stages_failed[0]
    print(f"Failed at: {failed_stage}")
    print(f"Error: {result.errors}")
```

### Memory Issues

```python
# Reduce parallel jobs or episodes
from daf.config import DAFConfig

cfg = DAFConfig(max_parallel_jobs=1)
result = daf_run_sweep_pipeline(config=cfg)
```

### Policy Not Found

```python
# Verify policy is importable
from daf import environment_checks
result = environment_checks.daf_check_dependencies()
assert result.is_healthy()
```

## See Also

- `README.md` - Orchestrator overview and usage
- `../AGENTS.md` - DAF source modules and policy integration
- `../README.md` - DAF source organization
- `../../AGENTS.md` - Top-level agent architecture
- `../../docs/API.md` - Complete API reference







