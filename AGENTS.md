# CoGames Agent Architecture

CoGames provides multiple policy interfaces and implementations for agents in the Cogs vs Clips cooperative game environment.

## Agent Types

### 1. Built-in Policies

CoGames includes ready-to-use policy implementations:

| Policy | Type | Use Case |
|--------|------|----------|
| `lstm` | Stateful Neural Network | Production training with history tracking |
| `stateless` | Feedforward Neural Network | Fast baseline; no temporal context |
| `baseline` | Scripted Agent | Rule-based coordination; deterministic behavior |
| `starter` | Template Policy | Starting point for custom implementations |
| `random` | Random Actions | Benchmark comparison; testing |
| `noop` | No-operation | Environment testing; baseline |

### 2. Policy Interfaces

All policies must implement one of two mettagrid interfaces:

#### `MultiAgentPolicy` (Training & Sweeps)
For batch processing multiple agents simultaneously:

```python
from mettagrid.policy.policy import MultiAgentPolicy
import numpy as np

class MyPolicy(MultiAgentPolicy):
    def step_batch(self, raw_observations: np.ndarray, raw_actions: np.ndarray) -> None:
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

**Best for**: Training via `cogames train`, sweeps via DAF, distributed evaluation

#### `AgentPolicy` (Play & Evaluation)
For single-agent step-by-step control:

```python
from mettagrid.policy.policy import AgentPolicy
from mettagrid.types import Action, AgentObservation

class MySingleAgentPolicy(AgentPolicy):
    def step(self, obs: AgentObservation) -> Action:
        """Select action for single agent."""
        action_name = self.select_action_name(obs)
        return Action(name=action_name)
```

**Best for**: Interactive play via `cogames play`, evaluation via `cogames eval`

### 3. Scripted Agents

Located in `src/cogames/policy/scripted_agent/`, these agents use programmatic decision-making:

- **BaselinePolicy**: Coordinates resource gathering and crafting
- **HeuristicAgent**: Task-prioritization and station optimization
- **RandomPolicy**: Uniformly random action selection

### 4. Nim-based Agents

Located in `src/cogames/policy/nim_agents/`, high-performance agents implemented in Nim:

- **RandomAgents**: Fast random policy baseline
- **ThinkyAgents**: Deliberative planning agents
- **RaceCarAgents**: Speed-optimized agents

## Policy Specifications

Use `PolicySpec` to precisely specify policy configurations:

```python
from mettagrid.policy.policy import PolicySpec

# Built-in policy
policy = PolicySpec(class_path="lstm")

# Custom policy with weights
policy = PolicySpec(
    class_path="mypackage.MyPolicy",
    data_path="checkpoints/model_v2.pt"
)

# Load multiple checkpoints for comparison
policies = [
    PolicySpec(class_path="lstm", data_path=f"run{i}/checkpoint.pt")
    for i in range(3)
]
```

## Agent Coordination

Agents coordinate through:

1. **Movement**: Navigation around the map and stations
2. **Emotes**: Non-verbal communication signals (❤️, 🔄, 💯, etc.)
3. **Shared State**: Observing other agents' positions and emotes
4. **Resource Management**: Competing for and sharing limited resources

## Usage Patterns

### Training a Policy
```bash
cogames train -m training_facility_1 -p stateless --steps 100000
```

### Evaluating a Policy
```bash
cogames eval -m assembler_1 -p lstm:checkpoints/model.pt --episodes 10
```

### Playing with a Policy
```bash
cogames play -m training_facility_1 -p lstm --render gui
```

### Policy Submission
```bash
cogames login
cogames submit -p lstm:checkpoints/final_model.pt -n "My LSTM Policy"
```

## Extending Agents

To create custom policies:

1. Implement `MultiAgentPolicy` for training
2. Optionally implement `AgentPolicy` for interactive play
3. Support deterministic seeding for reproducibility
4. Minimize global state for distributed execution

See [DAF Developer Guide](daf/docs/AGENTS.md) for detailed custom policy tutorials.

## Policy Registry

Run `cogames policies` to list all available policies:

```bash
$ cogames policies
Built-in Policies:
  lstm          - LSTM-based policy
  stateless     - Feedforward neural network
  baseline      - Scripted baseline agent
  starter       - Policy template
  random        - Random action policy
  noop          - No-operation policy
```

## See Also

- [DAF Developer Guide](daf/docs/AGENTS.md) - Custom policy tutorials
- [CoGames README](README.md) - Overview and quick start
- [Mission Documentation](MISSION.md) - Game mechanics and observations
- [Technical Manual](TECHNICAL_MANUAL.md) - Sensor and action specifications

