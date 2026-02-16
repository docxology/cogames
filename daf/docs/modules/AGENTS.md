# DAF Module: Agent & Policy Support

Per-module documentation for agent and policy integration with DAF source modules.

## Policy Interface Requirements

DAF modules work with policies implementing these interfaces:

### MultiAgentPolicy (Training & Sweeps)

Required for batch processing:

```python
from mettagrid.policy.policy import MultiAgentPolicy

class MyPolicy(MultiAgentPolicy):
    def step_batch(self, observations, actions):
        """Process batch of observations."""
        for i in range(observations.shape[0]):
            actions[i] = self.compute_action(observations[i])
```

### AgentPolicy (Evaluation & Play)

Required for single-agent evaluation:

```python
from mettagrid.policy.policy import AgentPolicy

class MySingleAgentPolicy(AgentPolicy):
    def step(self, obs):
        """Select action for single agent."""
        return self.select_action(obs)
```

## Module Policy Support

### comparison.py

Compares multiple policies via `daf_compare_policies()`:

```python
from daf.src.comparison import daf_compare_policies
from mettagrid.policy.policy import PolicySpec

results = daf_compare_policies(
    policies=[
        PolicySpec(class_path="lstm"),
        PolicySpec(class_path="baseline"),
    ],
    missions=["training_facility_1"],
    episodes_per_mission=5,
)
```

**Policy Requirements:**
- Must implement `AgentPolicy` for evaluation
- Can use built-in shortcuts (`lstm`, `baseline`, `random`)
- Supports checkpoint loading via `data_path`

### deployment.py

Packages policies for deployment via `daf_package_policy()`:

```python
from daf.src.deployment import daf_package_policy

result = daf_package_policy(
    policy_class_path="mypackage.MyPolicy",
    weights_path="checkpoints/model.pt",
    output_dir="packages/",
)
```

**Policy Requirements:**
- Must be importable from `policy_class_path`
- Checkpoint format must match policy expectations
- Policy must work with CoGames submission API

### distributed_training.py

Orchestrates multi-machine training:

```python
from daf.src.distributed_training import daf_launch_distributed_training

job = daf_launch_distributed_training(
    policy_class_path="lstm",
    mission_name="training_facility_1",
    num_machines=4,
    steps_per_machine=250_000,
)
```

**Policy Requirements:**
- Must implement `MultiAgentPolicy` for training
- Must support deterministic seeding
- Should minimize global state for distributed execution

### mission_analysis.py

Analyzes policy performance per mission:

```python
from daf.src.mission_analysis import daf_analyze_mission_performance

analysis = daf_analyze_mission_performance(
    policy_class_path="lstm",
    mission_name="assembler_2",
    episodes=20,
)
```

**Policy Requirements:**
- Must implement `AgentPolicy` for evaluation
- Performance metrics extracted from `cogames.evaluate`

### visualization.py

Generates reports (no direct policy requirements):

```python
from daf.src.visualization import daf_export_comparison_html

daf_export_comparison_html(
    comparison_results,
    output_path="report.html",
)
```

## Built-in Policy Shortcuts

DAF supports CoGames policy shortcuts:

| Shortcut | Full Class Path |
|----------|----------------|
| `lstm` | `cogames.policy.lstm.LSTMPolicy` |
| `stateless` | `cogames.policy.stateless.StatelessPolicy` |
| `baseline` | `cogames.policy.scripted_agent.baseline_agent.BaselinePolicy` |
| `starter` | `cogames.policy.scripted_agent.starter_agent.StarterPolicy` |
| `random` | `cogames.policy.random.RandomPolicy` |
| `noop` | `cogames.policy.noop.NoopPolicy` |

## Best Practices

1. **Use PolicySpec for flexibility**:
   ```python
   from mettagrid.policy.policy import PolicySpec
   
   spec = PolicySpec(
       class_path="lstm",
       data_path="checkpoints/model.pt",
   )
   ```

2. **Validate checkpoints exist**:
   ```python
   from pathlib import Path
   
   ckpt = Path("checkpoints/model.pt")
   assert ckpt.exists(), f"Checkpoint not found: {ckpt}"
   ```

3. **Support deterministic seeding**:
   ```python
   class MyPolicy(MultiAgentPolicy):
       def __init__(self, env_interface, seed=42):
           self.rng = np.random.RandomState(seed)
   ```

## See Also

- [README.md](README.md) - Module index
- [../AGENTS.md](../AGENTS.md) - Full policy developer guide
- [../API.md](../API.md) - Complete API reference
- [../../AGENTS.md](../../AGENTS.md) - Top-level agent architecture

