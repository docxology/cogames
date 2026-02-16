# DAF Quick Start Guide

**Develop, evaluate, and compare CoGames policies using real simulation runs.**

## Installation

```bash
cd cogames
uv sync --all-extras
source .venv/bin/activate
```

## Quick Commands

### 1. Full Evaluation Suite (Recommended)

Run real CoGames evaluations with comprehensive visualizations:

```bash
# Quick evaluation (3 episodes, ~15 seconds)
./daf/scripts/run_full_suite.sh --quick

# Standard evaluation with sweep
./daf/scripts/run_full_suite.sh

# Custom policies and missions
./daf/scripts/run_full_suite.sh --policies lstm baseline random --missions training_facility.harvest

# Skip hyperparameter sweep
./daf/scripts/run_full_suite.sh --quick --no-sweep
```

### 2. Unit Tests

Run DAF module tests (mocked, fast):

```bash
./daf/run_tests.sh
```

### 3. Individual DAF Modules

```python
# Policy Comparison
from daf.src.comparison import daf_compare_policies
from mettagrid.policy.policy import PolicySpec
from cogames.cli.mission import get_mission

# Load mission
name, env_cfg, _ = get_mission("training_facility.harvest")

# Compare policies
report = daf_compare_policies(
    policies=[PolicySpec(class_path="baseline"), PolicySpec(class_path="random")],
    missions=[(name, env_cfg)],
    episodes_per_mission=5,
)
print(report.summary_statistics)
```

```python
# Hyperparameter Sweep
from daf.src.sweeps import daf_launch_sweep
from daf.src.config import DAFSweepConfig

cfg = DAFSweepConfig(
    name="my_sweep",
    policy_class_path="baseline",
    missions=["training_facility.harvest"],
    strategy="grid",
    search_space={"learning_rate": [0.001, 0.01]},
    episodes_per_trial=3,
)
results = daf_launch_sweep(cfg)
print(f"Best: {results.get_best_trial().hyperparameters}")
```

## Output Structure

```
daf_output/full_suite/suite_YYYYMMDD_HHMMSS/
├── comparisons/
│   ├── report.html                  # Interactive comparison
│   ├── policy_rewards_comparison.png
│   ├── performance_by_mission.png
│   └── leaderboard.json
├── sweeps/baseline/
│   ├── sweep_progress.png           # Trial performance
│   ├── heatmap.png                  # Parameter heatmap
│   └── sweep_results.json
├── dashboard/
│   └── dashboard.html               # Summary dashboard
└── SUITE_SUMMARY.json
```

## Available Missions

```bash
# List all missions
python3 -c "from cogames.cogs_vs_clips.missions import MISSIONS; [print(m.full_name()) for m in MISSIONS[:10]]"
```

Common missions:
- `training_facility.harvest`
- `training_facility.open_world`
- `hello_world.hello_world_unclip`
- `machina_1.open_world`

## Available Policies

| Policy | Description |
|--------|-------------|
| `baseline` | Scripted coordination agent |
| `random` | Random action selection |
| `lstm` | LSTM-based neural policy |
| `stateless` | Feedforward neural network |

## Workflow: Develop → Evaluate → Compare

1. **Develop** your policy implementing `MultiAgentPolicy`:

```python
from mettagrid.policy.policy import MultiAgentPolicy

class MyPolicy(MultiAgentPolicy):
    def step_batch(self, observations, actions):
        for i in range(len(observations)):
            actions[i] = self.compute_action(observations[i])
```

2. **Evaluate** against baselines:

```bash
./daf/scripts/run_full_suite.sh --policies mypackage.MyPolicy baseline --quick
```

3. **Compare** with statistical analysis:

Open `daf_output/full_suite/suite_*/comparisons/report.html`

## See Also

- `daf/examples/README.md` - Configuration examples
- `daf/AGENTS.md` - Policy interfaces
- `TECHNICAL_MANUAL.md` - Game mechanics
