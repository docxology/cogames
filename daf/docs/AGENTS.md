# DAF Developer Guide for Custom Policies

**DAF is a sidecar utility that extends CoGames.** This guide explains how to use DAF tools with your custom policies. DAF wraps and extends CoGames functionality, working seamlessly with any policy that implements the `MultiAgentPolicy` or `AgentPolicy` interface from mettagrid.

## DAF as Sidecar Utility

DAF extends CoGames by:
- Wrapping `cogames.train.train()` for distributed training
- Using `cogames.evaluate.evaluate()` for sweeps and comparisons
- Using `cogames.cli.mission` for mission discovery
- Using `cogames.device` for device resolution

All DAF functions invoke underlying CoGames methods - DAF never duplicates core functionality.

## Policy Compatibility

DAF works with all policies compatible with cogames:

1. **Built-in policies** (shorthand names):
   - `lstm` - LSTM-based policy
   - `baseline` - Baseline scripted agent
   - `starter` - Starter policy template

2. **Custom policies** (full class paths):
   - Must implement `MettagridPolicy` or `TrainablePolicy` interface
   - Must be importable from your Python environment
   - Can have optional checkpoint/weights files

## Specifying Policies in DAF

### Using Policy Shortcuts

CoGames provides shortcuts for common policies:

```python
from daf import sweeps

# Use shorthand
sweep_cfg = DAFSweepConfig(
    policy_class_path="lstm",  # Equivalent to "cogames.policy.lstm.LSTMPolicy"
    ...
)
```

### Using Full Class Paths

For custom policies:

```python
from daf import sweeps

sweep_cfg = DAFSweepConfig(
    policy_class_path="mypackage.policies.MyCustomPolicy",
    ...
)
```

### Including Weights/Checkpoints

Specify pre-trained weights:

```python
from mettagrid.policy.policy import PolicySpec

policy = PolicySpec(
    class_path="cogames.policy.lstm.LSTMPolicy",
    data_path="path/to/checkpoint.pt"  # Optional weights file
)
```

## Policy Interfaces

### MultiAgentPolicy (Required for Training/Sweeps)

For batch processing during training:

```python
from mettagrid.policy.policy import MultiAgentPolicy
import numpy as np

class MyPolicy(MultiAgentPolicy):
    def __init__(self, env_interface, policy_env_info):
        super().__init__(env_interface)
        # Initialize policy components
        self.network = MyNetwork(...)

    def step_batch(self, raw_observations: np.ndarray, raw_actions: np.ndarray) -> None:
        """Process batch of observations and write actions to output buffer.
        
        Args:
            raw_observations: Shape (num_cogs, num_tokens, 3)
            raw_actions: Shape (num_cogs,) - write action IDs here
        """
        num_cogs = raw_observations.shape[0]
        for agent_id in range(num_cogs):
            obs = raw_observations[agent_id]
            action_id = self.select_action(obs)
            raw_actions[agent_id] = action_id

    def agent_policy(self, agent_id: int):
        """Return single-agent policy for play/eval."""
        return self  # Or return agent-specific policy variant
```

### AgentPolicy (For Play/Evaluate Only)

For single-agent policy interface:

```python
from mettagrid.policy.policy import AgentPolicy

class MySingleAgentPolicy(AgentPolicy):
    def step(self, obs: AgentObservation) -> Action:
        """Select action for single agent."""
        # Your policy logic here
        action_name = self.select_action_name(obs)
        return Action(name=action_name)
```

## Using DAF with Custom Policies

### Hyperparameter Sweeps

Sweep over policy hyperparameters:

```yaml
# sweep_custom_policy.yaml
name: "custom_policy_sweep"
missions: ["training_facility_1"]
policy_class_path: "mypackage.MyCustomPolicy"
strategy: "grid"
search_space:
  learning_rate: [0.0001, 0.001]
  hidden_size: [64, 128]
  dropout_rate: [0.1, 0.2]
episodes_per_trial: 3
```

Then launch:

```python
from daf import config, sweeps

cfg = config.DAFSweepConfig.from_yaml("sweep_custom_policy.yaml")
results = sweeps.daf_launch_sweep(cfg)
best = sweeps.daf_sweep_best_config(results)
print(f"Best config: {best.hyperparameters}")
```

### Policy Comparison

Compare your policy against baselines:

```python
from daf import comparison
from mettagrid.policy.policy import PolicySpec

results = comparison.daf_compare_policies(
    policies=[
        PolicySpec(class_path="mypackage.MyCustomPolicy"),
        PolicySpec(class_path="cogames.policy.lstm.LSTMPolicy"),
        PolicySpec(class_path="cogames.policy.scripted_agent.baseline_agent.BaselinePolicy"),
    ],
    missions=["training_facility_1", "assembler_2"],
    episodes_per_mission=5
)

print(results.summary_statistics)
print(results.pairwise_significance_tests)
```

### Training with DAF

Use DAF environment checks and orchestration with your policy:

```python
from daf import environment_checks, orchestrators

# Check environment first
env_result = environment_checks.daf_check_environment(
    missions=["training_facility_1"]
)

if not env_result.is_healthy():
    raise RuntimeError("Environment check failed")

# Train using DAF pipeline
pipeline_result = orchestrators.daf_run_training_pipeline(
    policy_class_path="mypackage.MyTrainablePolicy",
    missions=["training_facility_1"],
    num_training_steps=1_000_000,
    eval_missions=["assembler_2"],
    eval_episodes=10
)

print(f"Final checkpoint: {pipeline_result.final_checkpoint}")
```

## Integrating with Existing Workflows

### Adding DAF to Your Training Script

```python
import torch
from pathlib import Path
from daf import environment_checks, config
from cogames.train import train
from mettagrid import MettaGridConfig
from mypackage import MyTrainablePolicy

# Step 1: Check environment
env_result = environment_checks.daf_check_environment(
    checkout_dir=Path("./checkpoints")
)

if not env_result.is_healthy():
    exit(1)

# Step 2: Load mission and create environment
from cogames.cli.mission import get_mission_name_and_config
mission_name, env_cfg, _ = get_mission_name_and_config(None, "training_facility_1")

# Step 3: Train using cogames infrastructure
device = environment_checks.daf_get_recommended_device()
train(
    env_cfg=env_cfg,
    policy_class_path="mypackage.MyTrainablePolicy",
    device=device,
    num_steps=1_000_000,
    checkpoints_path=Path("./checkpoints"),
)
```

### Comparing Multiple Training Runs

```python
from daf import comparison, visualization
from mettagrid.policy.policy import PolicySpec

# Compare outputs from multiple training runs
policies = [
    PolicySpec(class_path="lstm", data_path="run1/checkpoint.pt"),
    PolicySpec(class_path="lstm", data_path="run2/checkpoint.pt"),
    PolicySpec(class_path="lstm", data_path="run3/checkpoint.pt"),
]

comparison_result = comparison.daf_compare_policies(
    policies=policies,
    missions=["training_facility_1", "assembler_2"],
    episodes_per_mission=10
)

# Generate comparison report
visualization.daf_export_comparison_html(
    comparison_result,
    output_path="comparison_report.html"
)
```

## Best Practices

### 1. Make Policies Stateless-Compatible

DAF may run policies in different processes/machines. Minimize global state:

```python
# GOOD: State in instance variables
class MyPolicy(MultiAgentPolicy):
    def __init__(self, policy_env_info):
        super().__init__(policy_env_info)
        self.network = MyNetwork()
        self.rng = np.random.RandomState(42)

# AVOID: Global state
global_network = None  # Don't do this!
```

### 2. Support Deterministic Seeding

Make policies reproducible:

```python
class MyPolicy(MultiAgentPolicy):
    def __init__(self, policy_env_info, seed=42):
        super().__init__(policy_env_info)
        self.rng = np.random.RandomState(seed)
        torch.manual_seed(seed)
```

### 3. Handle Optional Checkpoints Gracefully

Support both random init and pre-trained weights:

```python
class MyPolicy(TrainablePolicy):
    def __init__(self, policy_env_info, checkpoint_path=None):
        super().__init__(policy_env_info)
        self.network = MyNetwork(...)
        if checkpoint_path:
            self.network.load_state_dict(torch.load(checkpoint_path))

    @classmethod
    def load(cls, path, env=None):
        return cls(env, checkpoint_path=path)
```

### 4. Profile Before Sweeping

Test a single run before launching large sweeps:

```python
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator.rollout import Rollout
from cogames.cli.mission import get_mission_name_and_config

# Quick sanity check
_, env_cfg, _ = get_mission_name_and_config(None, "training_facility_1")
policy_env_info = PolicyEnvInterface.from_mg_cfg(env_cfg)
policy = MyPolicy(policy_env_info)
agent_policies = [policy.agent_policy(i) for i in range(env_cfg.game.num_agents)]

rollout = Rollout(env_cfg, agent_policies, render_mode="none", seed=42)
rollout.run_until_done()
print(f"First run reward: {sum(rollout._sim.episode_rewards)}")
```

### 5. Use Mission Variants for Robustness

Test across difficulty variants:

```python
from daf import comparison

# Test across different difficulties
missions = [
    "training_facility_1",  # Base
    "training_facility_1:hard",  # Hard variant
    "training_facility_1:energy_starved",  # Limited energy
]

results = comparison.daf_compare_policies(
    policies=[PolicySpec(class_path="mypackage.MyPolicy")],
    missions=missions,
    episodes_per_mission=5
)
```

## Troubleshooting

### Policy Not Found

```python
# Error: ModuleNotFoundError: No module named 'mypackage'
# Solution: Ensure your policy module is in PYTHONPATH
import sys
sys.path.insert(0, "/path/to/your/package")
```

### Checkpoint Not Loading

```python
# Error: Could not load checkpoint
# Solution: Verify path and format
from pathlib import Path

ckpt = Path("path/to/checkpoint.pt")
if not ckpt.exists():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

weights = torch.load(ckpt, map_location="cpu")
print(f"Checkpoint keys: {weights.keys()}")
```

### Out of Memory During Sweeps

```python
# Solution: Reduce parallel jobs or episode count
from daf.config import DAFConfig

config = DAFConfig(
    max_parallel_jobs=2,  # Reduce from default 4
)

# Or reduce episodes in sweep config
sweep_cfg.episodes_per_trial = 1
```

## See Also

- `README.md` - DAF overview and quick start
- `API.md` - Complete API reference
- `/daf/examples/` - Example configurations
- `TECHNICAL_MANUAL.md` - CoGames sensor and action specifications

