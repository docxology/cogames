# DAF Orchestrators

Orchestrators chain DAF modules into complete multi-stage workflows. Each orchestrator is a thin wrapper that sequences operations and handles transitions between stages.

## Orchestrator Modules

```
orchestrators/
├── __init__.py                  # Main orchestrator exports
├── training_pipeline.py         # Training workflow
├── sweep_pipeline.py            # Hyperparameter sweep workflow
├── comparison_pipeline.py       # Policy comparison workflow
└── benchmark_pipeline.py        # End-to-end benchmark workflow
```

## Quick Start

### Training Pipeline

Train a policy with optional evaluation:

```python
from daf.orchestrators import daf_run_training_pipeline

result = daf_run_training_pipeline(
    policy_class_path="lstm",
    missions=["training_facility_1"],
    num_training_steps=1_000_000,
    eval_missions=["assembler_2"],
    eval_episodes=10,
)

if result.is_success():
    print(f"Checkpoint: {result.outputs['final_checkpoint']}")
```

### Sweep Pipeline

Run hyperparameter search:

```python
from daf.orchestrators import daf_run_sweep_pipeline
from daf.config import DAFSweepConfig

cfg = DAFSweepConfig.from_yaml("sweep.yaml")
result = daf_run_sweep_pipeline(config=cfg)

best = result.outputs['best_config']
```

### Comparison Pipeline

Compare multiple policies:

```python
from daf.orchestrators import daf_run_comparison_pipeline
from mettagrid.policy.policy import PolicySpec

policies = [
    PolicySpec(class_path="lstm"),
    PolicySpec(class_path="baseline"),
]

result = daf_run_comparison_pipeline(
    policies=policies,
    missions=["training_facility_1", "assembler_2"],
    episodes_per_mission=5,
)

if result.is_success():
    print(result.outputs['summary_statistics'])
```

### Benchmark Pipeline

Complete workflow from training through deployment:

```python
from daf.orchestrators import daf_run_benchmark_pipeline

result = daf_run_benchmark_pipeline(
    policies=["lstm"],
    missions=["training_facility_1"],
    episodes_per_mission=10,
)

print(f"Status: {result.status}")
```

## Pipeline Stages

Each orchestrator executes stages in sequence:

### Training Pipeline Stages

```
1. Environment Check
   └─ Validate CUDA, disk, dependencies
   
2. Training
   └─ Train policy using cogames.train.train()
   
3. Checkpoint Save
   └─ Save trained weights
   
4. Evaluation (optional)
   └─ Evaluate on test missions
```

### Sweep Pipeline Stages

```
1. Environment Check
   └─ Validate environment
   
2. Sweep Execution
   └─ Run hyperparameter search
   
3. Best Config Extraction
   └─ Identify best performing configuration
   
4. Comparison (optional)
   └─ Compare best configs
```

### Comparison Pipeline Stages

```
1. Environment Check
   └─ Validate environment
   
2. Policy Comparison
   └─ Evaluate all policies
   
3. Statistical Analysis
   └─ Compute significance tests
   
4. Report Generation
   └─ Create HTML report
```

### Benchmark Pipeline Stages

```
1. Environment Check
   └─ Validate environment
   
2. Training (if needed)
   └─ Train policies
   
3. Evaluation
   └─ Evaluate across missions
   
4. Comparison (optional)
   └─ Compare with baselines
   
5. Deployment (optional)
   └─ Create deployment package
```

## Result Objects

All orchestrators return `PipelineResult`:

```python
result = daf_run_training_pipeline(...)

# Check overall status
if result.is_success():
    print("Pipeline succeeded")

# Examine stages
print(f"Completed: {result.stages_completed}")
print(f"Failed: {result.stages_failed}")

# Access stage outputs
checkpoint = result.outputs['final_checkpoint']

# Check timing
print(f"Total time: {result.total_time_seconds}s")
```

## Workflow Examples

### Example 1: Simple Training

```python
from daf.orchestrators import daf_run_training_pipeline

result = daf_run_training_pipeline(
    policy_class_path="lstm",
    missions=["training_facility_1"],
    num_training_steps=100_000,
)

checkpoint = result.outputs['final_checkpoint']
```

### Example 2: Training with Evaluation

```python
result = daf_run_training_pipeline(
    policy_class_path="lstm",
    missions=["training_facility_1"],
    num_training_steps=1_000_000,
    eval_missions=["training_facility_2", "assembler_2"],
    eval_episodes=10,
)

eval_results = result.outputs['evaluation_results']
```

### Example 3: Hyperparameter Optimization

```python
from daf.orchestrators import daf_run_sweep_pipeline
from daf.config import DAFSweepConfig

cfg = DAFSweepConfig(
    policy_class_path="lstm",
    missions=["training_facility_1"],
    strategy="grid",
    search_space={
        'learning_rate': [0.0001, 0.001, 0.01],
        'hidden_size': [64, 128, 256],
    },
    episodes_per_trial=5,
)

result = daf_run_sweep_pipeline(
    config=cfg,
    compare_best=True,
)

best_config = result.outputs['best_config']
comparison = result.outputs['comparison_results']
```

### Example 4: Multi-Policy Comparison

```python
from daf.orchestrators import daf_run_comparison_pipeline
from mettagrid.policy.policy import PolicySpec

policies = [
    PolicySpec(class_path="lstm", data_path="ckpt/lstm_v1.pt"),
    PolicySpec(class_path="lstm", data_path="ckpt/lstm_v2.pt"),
    PolicySpec(class_path="stateless"),
    PolicySpec(class_path="baseline"),
]

result = daf_run_comparison_pipeline(
    policies=policies,
    missions=[
        "training_facility_1",
        "training_facility_2",
        "assembler_2",
    ],
    episodes_per_mission=10,
)

stats = result.outputs['summary_statistics']
print(f"Winner: {stats['best_policy']}")
```

### Example 5: Chained Multi-Stage Workflow

```python
from daf.orchestrators import (
    daf_run_training_pipeline,
    daf_run_sweep_pipeline,
    daf_run_comparison_pipeline,
)
from daf.config import DAFSweepConfig
from mettagrid.policy.policy import PolicySpec

# Stage 1: Train initial policy
print("Stage 1: Training...")
train_result = daf_run_training_pipeline(
    policy_class_path="lstm",
    missions=["training_facility_1"],
    num_training_steps=1_000_000,
)
assert train_result.is_success()

# Stage 2: Optimize with hyperparameter sweep
print("Stage 2: Hyperparameter sweep...")
sweep_cfg = DAFSweepConfig(
    policy_class_path="lstm",
    policy_data_path=train_result.outputs['final_checkpoint'],
    missions=["training_facility_2"],
    search_space={'learning_rate': [0.0001, 0.001]},
)

sweep_result = daf_run_sweep_pipeline(config=sweep_cfg)
assert sweep_result.is_success()

# Stage 3: Compare against baselines
print("Stage 3: Comparison...")
comparison_result = daf_run_comparison_pipeline(
    policies=[
        PolicySpec(
            class_path="lstm",
            data_path=sweep_result.outputs['best_checkpoint']
        ),
        PolicySpec(class_path="baseline"),
    ],
    missions=["assembler_2"],
    episodes_per_mission=5,
)

if comparison_result.is_success():
    print("Complete workflow succeeded!")
    stats = comparison_result.outputs['summary_statistics']
    print(f"Final winner: {stats['best_policy']}")
```

## Error Handling

### Check Pipeline Success

```python
result = daf_run_training_pipeline(...)

if result.is_success():
    checkpoint = result.outputs['final_checkpoint']
elif result.status == "partial":
    print(f"Partial success at: {result.stages_completed[-1]}")
else:
    print(f"Failed: {result.errors}")
```

### Identify Failed Stages

```python
if not result.is_success():
    print(f"Failed stages: {result.stages_failed}")
    for error in result.errors:
        print(f"  - {error}")
```

### Continue on Failure

```python
result = daf_run_comparison_pipeline(...)

# Extract what succeeded even if pipeline partially failed
if result.stages_completed:
    completed = result.stages_completed[-1]
    print(f"Latest completed stage: {completed}")
    
    if 'summary_statistics' in result.outputs:
        print(result.outputs['summary_statistics'])
```

## Performance & Resources

### Resource Limits

```python
from daf.config import DAFConfig

cfg = DAFConfig(
    max_parallel_jobs=2,        # Limit parallel jobs
    gpu_memory_fraction=0.5,    # Limit GPU memory
)

result = daf_run_sweep_pipeline(
    config=sweep_cfg,
)
```

### Execution Monitoring

```python
result = daf_run_training_pipeline(...)

print(f"Total time: {result.total_time_seconds}s")
print(f"Timestamp: {result.timestamp}")
print(f"Stages: {result.stages_completed}")
```

## Testing

### Test Single Stage

```python
from daf.orchestrators import daf_run_training_pipeline

# Quick test with small scale
result = daf_run_training_pipeline(
    policy_class_path="lstm",
    missions=["training_facility_1"],
    num_training_steps=100,  # Tiny for testing
)

assert result.is_success()
```

### Test Full Pipeline

```bash
# Run orchestrator tests
python -m pytest daf/tests/test_orchestrators.py -v
```

## See Also

- `AGENTS.md` - Detailed agent and policy support
- `../README.md` - DAF source modules
- `../AGENTS.md` - Policy interfaces and integration
- `../../AGENTS.md` - Top-level agent architecture
- `../../docs/API.md` - Complete API documentation






