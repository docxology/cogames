# DAF Integration with CoGames Cursorrules

## Core Integration Principle

**DAF is a non-duplicating wrapper around CoGames.**

- DAF adds: orchestration, analysis, reporting, distributed training
- DAF delegates to CoGames: policy management, mission loading, environment interaction
- Never reimplement CoGames functionality in DAF

## Policy Interfaces

DAF works with two CoGames policy interfaces:

### 1. MultiAgentPolicy (for training)

Used by sweeps and distributed training:

```python
from mettagrid.policy.policy import MultiAgentPolicy
import numpy as np

class MyPolicy(MultiAgentPolicy):
    def __init__(self, env_interface, seed=42):
        super().__init__(env_interface)
        self.seed = seed
    
    def step_batch(
        self,
        raw_observations: np.ndarray,
        raw_actions: np.ndarray,
    ) -> None:
        """Process batch of observations for all agents.
        
        Args:
            raw_observations: Shape (num_agents, num_tokens, 3)
            raw_actions: Shape (num_agents,) - write action IDs here
        """
        num_agents = raw_observations.shape[0]
        for agent_id in range(num_agents):
            obs = raw_observations[agent_id]
            action_id = self.select_action(obs)
            raw_actions[agent_id] = action_id
```

### 2. AgentPolicy (for evaluation/comparison)

Used by evaluation and policy comparison:

```python
from mettagrid.policy.policy import AgentPolicy
from mettagrid.types import Action, AgentObservation

class MySingleAgentPolicy(AgentPolicy):
    def step(self, obs: AgentObservation) -> Action:
        """Select action for single agent."""
        action_name = self.select_action_name(obs)
        return Action(name=action_name)
```

## Policy Loading

Always use CoGames PolicySpec for consistency:

```python
from mettagrid.policy.policy import PolicySpec

# Built-in policy
spec = PolicySpec(class_path="lstm")

# Custom policy
spec = PolicySpec(class_path="mypackage.CustomPolicy")

# Policy with weights
spec = PolicySpec(
    class_path="lstm",
    data_path="checkpoints/model.pt"
)

# Load policy
policy = spec.load()
```

**DAF Usage**:

```python
# In sweeps
from daf.config import DAFSweepConfig

config = DAFSweepConfig(
    policy_class_path="lstm",  # String path, not loaded yet
)

# DAF loads policies internally when needed
```

## Mission Configuration

Always use CoGames mission loading:

```python
from cogames.game import load_mission_config

# Load mission configuration
mission_config = load_mission_config("training_facility_1")

# Use config to create mission
game = Game.from_config(mission_config)
```

**DAF Integration**:

```python
# In configuration
config = DAFSweepConfig(
    missions=["training_facility_1", "assembler_2"],
)

# DAF loads missions internally
for mission_name in config.missions:
    mission_config = load_mission_config(mission_name)
    # Use mission
```

## Environment Interaction

All environment stepping must go through CoGames Game class:

```python
# ✓ Good - use CoGames Game
game = Game.from_config(mission_config)
obs = game.reset()
action = policy.step(obs)
next_obs, reward, done = game.step(action)

# ✗ Bad - don't reimplement environment
# (don't create custom environments)
```

## PolicySpec Pattern in DAF

### Creating PolicySpec from String

```python
from mettagrid.policy.policy import PolicySpec

# From configuration
config = DAFSweepConfig(
    policy_class_path="lstm",
    policy_data_path="checkpoints/model.pt",  # Optional
)

# Create spec
spec = PolicySpec(
    class_path=config.policy_class_path,
    data_path=config.policy_data_path,
)

# Load policy
policy = spec.load()
```

### Multiple Policies for Comparison

```python
from mettagrid.policy.policy import PolicySpec

policies = [
    PolicySpec(class_path="lstm"),
    PolicySpec(class_path="baseline"),
    PolicySpec(class_path="random"),
]

# Load all
loaded_policies = [spec.load() for spec in policies]
```

## Mission Lists

DAF accepts mission names as strings:

```python
# Configuration
config = DAFComparisonConfig(
    missions=[
        "training_facility_1",
        "assembler_2",
        "hello_world.open_world",
    ]
)

# DAF resolves names internally
for mission_name in config.missions:
    mission_config = load_mission_config(mission_name)
```

## Evaluation Pattern

DAF uses CoGames evaluate() function:

```python
from cogames.evaluate import evaluate

# Get evaluation results
results = evaluate(
    policy=policy,
    mission=mission_config,
    num_episodes=10,
    max_steps=1000,
)

# DAF processes and saves results
output_mgr.save_json_results(
    results.to_dict(),
    operation="evaluation",
    filename="results"
)
```

## Training Integration

DAF wraps CoGames training:

```python
from cogames.train import train

# Configure training
train_config = TrainingConfig(...)

# Run training via CoGames
checkpoint = train(
    policy=policy,
    config=train_config,
    # ... other args
)

# DAF saves and organizes checkpoint
output_mgr.save_json_results(
    {"checkpoint": str(checkpoint)},
    operation="training",
    filename="results"
)
```

## Device Management

Respect CoGames device configuration:

```python
import os
from cogames.device import get_device

# Get CoGames device
device = get_device()

# Or from environment
device = os.getenv("COGAMES_DEVICE", "cpu")

# Use with policies
policy = spec.load()
policy = policy.to(device)
```

**In DAF Configuration**:

```python
config = DAFConfig(
    enable_gpu=True,
    gpu_memory_fraction=0.8,
)

# DAF respects these settings for device allocation
```

## Verbose/Logging Integration

Coordinate with CoGames verbose settings:

```python
import os

# Check CoGames verbose
cogames_verbose = os.getenv("COGAMES_VERBOSE", "false").lower() == "true"

# Set DAF verbose
daf_config = DAFConfig(
    verbose=cogames_verbose,
)

# Use both
logger = create_daf_logger("operation", verbose=daf_config.verbose)
```

## Seeding for Reproducibility

Use consistent seeding across CoGames and DAF:

```python
import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    """Set seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CoGames also sets its own seed

# In DAF operations
config = DAFSweepConfig(
    seed=42,
)

set_seed(config.seed)
```

## Error Handling Integration

Preserve CoGames error context:

```python
try:
    # CoGames operation
    game = Game.from_config(mission_config)
    obs = game.reset()
except CoGamesError as e:
    # Log DAF context, preserve CoGames error
    logger.error(f"Game initialization failed: {e}")
    raise
```

## Dependency Management

### Required CoGames Modules

```python
# Mission loading
from cogames.game import load_mission_config

# Game creation
from cogames.game import Game

# Training
from cogames.train import train

# Evaluation
from cogames.evaluate import evaluate

# Policies
from mettagrid.policy.policy import PolicySpec, MultiAgentPolicy, AgentPolicy

# Device
from cogames.device import get_device
```

### Optional Integrations

```python
# Advanced features (if available)
from cogames.export import export_model  # May not exist
from cogames.benchmark import benchmark  # May not exist
```

## Version Compatibility

DAF is compatible with:
- **CoGames**: Main branch (tracked in requirements)
- **mettagrid**: Latest stable
- **PyTorch**: 2.0+
- **Python**: 3.9+

Check versions:
```bash
pip show cogames mettagrid torch
```

## No-Duplication Checklist

When adding DAF features, ensure:

- [ ] Not reimplementing policy loading (use PolicySpec)
- [ ] Not creating custom environments (use CoGames Game)
- [ ] Not duplicating mission definitions
- [ ] Not reimplementing training (use cogames.train)
- [ ] Not duplicating evaluation logic (use cogames.evaluate)
- [ ] Using CoGames error types (not custom)
- [ ] Respecting CoGames configuration
- [ ] Not duplicating device management

## Integration Testing

Test DAF → CoGames integration:

```python
def test_daf_sweep_uses_cogames_policies():
    """Ensure sweeps properly load CoGames policies."""
    from mettagrid.policy.policy import PolicySpec
    from daf import sweeps
    
    config = DAFSweepConfig(
        policy_class_path="lstm",
        missions=["training_facility_1"],
        num_trials=2,
    )
    
    results = sweeps.daf_launch_sweep(config)
    
    # Verify used actual CoGames policies
    assert results.num_trials == 2
    assert results.best_score > 0
```

## Common Integration Patterns

### Pattern 1: Sweep with Custom Policy

```python
from daf import sweeps
from daf.config import DAFSweepConfig

config = DAFSweepConfig(
    policy_class_path="my.custom.PolicyClass",
    policy_data_path="optional_weights.pt",
    missions=["training_facility_1"],
)

results = sweeps.daf_launch_sweep(config)
```

### Pattern 2: Compare Built-in vs Custom

```python
from daf import comparison

results = comparison.daf_compare_policies(
    policies=["lstm", "baseline", "my.custom.Policy"],
    missions=["training_facility_1", "assembler_2"],
)
```

### Pattern 3: Train and Deploy

```python
from cogames.train import train
from daf.output_manager import get_output_manager
from daf.deployment import package_policy

# Train using CoGames
checkpoint = train(policy, config)

# Package using DAF
output_mgr = get_output_manager()
output_mgr.save_json_results(
    {"checkpoint": str(checkpoint)},
    operation="training",
    filename="results"
)
```

## Troubleshooting Integration

### Policy not loading?
```python
# Check PolicySpec is correct
from mettagrid.policy.policy import PolicySpec
spec = PolicySpec(class_path="lstm")
policy = spec.load()  # Should work
```

### Mission not found?
```python
# Check mission exists
from cogames.game import load_mission_config
config = load_mission_config("mission_name")  # Should not raise
```

### Incompatible policy interface?
```python
# Check policy implements required interface
from mettagrid.policy.policy import MultiAgentPolicy
assert isinstance(policy, MultiAgentPolicy)
```

---

**Status**: Production Ready ✅
**Last Updated**: November 21, 2024







