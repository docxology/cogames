# DAF Comparison Module

**DAF sidecar utility: Policy comparison by invoking `cogames.evaluate.evaluate()`.**

## Overview

Head-to-head policy evaluation with statistical significance testing, performance scoring, and comprehensive metrics analysis.

## CoGames Integration

**Primary Method**: `cogames.evaluate.evaluate()`

Each policy comparison:
1. Invokes evaluation once with all policies
2. Extracts per-policy, per-mission rewards and agent metrics
3. Computes performance scores when raw rewards are zero
4. Performs statistical analysis (t-tests, effect sizes)
5. Generates reports with detailed metrics

## Key Functions

- `daf_compare_policies()` - Multi-policy comparison with performance scoring
- `daf_benchmark_suite()` - Standardized benchmarks
- `daf_policy_ablation()` - Ablation studies

## Performance Score Computation

When environments return zero rewards (common for missions without explicit reward functions), DAF computes a **composite performance score** from agent metrics.

### Score Formula

```python
score = 0.0

# Resource gathering (most important for game success)
score += carbon.gained * 1.0
score += silicon.gained * 1.0
score += oxygen.gained * 1.0
score += germanium.gained * 1.0
score += energy.gained * 1.0

# Inventory diversity (indicates good resource management)
score += inventory.diversity * 10.0

# Successful actions (indicates active, productive behavior)
score += action.move.success * 0.1

# Penalties for ineffective behavior
score -= action.failed * 0.05
score -= status.max_steps_without_motion * 0.1
```

### Usage

```python
from daf.comparison import daf_compare_policies
from mettagrid.policy.policy import PolicySpec

report = daf_compare_policies(
    policies=[PolicySpec(class_path="baseline"), PolicySpec(class_path="random")],
    missions=[("hello_world.hello_world_unclip", env_cfg)],
    episodes_per_mission=5,
    use_performance_score=True,  # Enabled by default
)

# Results show meaningful scores even when raw rewards are zero
print(report.summary_statistics)
# {'baseline': {'avg_reward': 6168.60, 'std_dev': 0.0}, 
#  'random': {'avg_reward': 1206.42, 'std_dev': 0.0}}
```

## Detailed Metrics

The `ComparisonReport` includes detailed agent metrics for comprehensive analysis.

### Accessing Detailed Metrics

```python
# After running comparison
report = daf_compare_policies(...)

# Access detailed metrics per policy per mission
baseline_metrics = report.policy_detailed_metrics['baseline']['hello_world.hello_world_unclip']

print(baseline_metrics)
# {
#   'action.move.success': 2607.5,
#   'action.move.failed': 2341.0,
#   'carbon.gained': 50.0,
#   'silicon.gained': 150.0,
#   'energy.gained': 5525.0,
#   'inventory.diversity': 25.0,
#   ...
# }
```

### Metric Categories

| Category | Metrics |
|----------|---------|
| **Resources Gained** | carbon.gained, silicon.gained, oxygen.gained, germanium.gained |
| **Resources Held** | carbon.amount, silicon.amount, oxygen.amount, germanium.amount |
| **Energy** | energy.amount, energy.gained, energy.lost |
| **Actions** | action.move.success, action.move.failed, action.noop.success, action.change_vibe.success, action.failed |
| **Inventory** | inventory.diversity, inventory.diversity.ge.2, .ge.3, .ge.4, .ge.5 |
| **Status** | status.max_steps_without_motion |

### JSON Export

When saving comparison results, detailed metrics are included:

```python
report.save_json("comparison_results.json")
```

Output includes:
```json
{
  "summary_statistics": {...},
  "pairwise_comparisons": {...},
  "detailed_metrics": {
    "baseline": {
      "hello_world.hello_world_unclip": {
        "carbon.gained": 50.0,
        "energy.gained": 5525.0,
        ...
      }
    },
    "random": {...}
  }
}
```

## Statistical Analysis

### Pairwise Comparisons

For each pair of policies, DAF computes:
- **T-test** (two-sample independent)
- **P-value** for statistical significance
- **Cohen's d** effect size
- **Winner** determination

### Handling Zero-Variance Data

When performance scores are identical across episodes (common with aggregated metrics), DAF handles the edge case gracefully:
- If means differ, difference is considered significant (p=0.0)
- If means are identical, p=1.0 (no difference)

## Visualization

See `daf/docs/modules/visualization.md` for:
- `daf_plot_policy_comparison()` - Basic comparison plots
- `daf_plot_detailed_metrics_comparison()` - Detailed metrics bar charts
- `daf_export_comparison_html()` - HTML report with metrics tables

## Example Output

Running the full suite produces:

```
Policy Comparison Report
Policies: baseline, random
Episodes per mission: 5

Summary Statistics:
  baseline: avg=6168.60, std=0.00
  random: avg=1206.42, std=0.00

Pairwise Comparison:
  baseline vs random: p=0.0000 (significant), winner=baseline
```

## Testing

See `daf/tests/test_comparison.py` for coverage.

## See Also

- `daf/docs/API.md` - Complete API reference
- `daf/docs/INTEGRATION_MAP.md` - Full integration details
- `daf/docs/modules/visualization.md` - Visualization functions
