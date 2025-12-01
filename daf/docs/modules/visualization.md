# DAF Visualization Module

**DAF sidecar utility: Visualization extending patterns from `scripts/run_evaluation.py`.**

## Overview

Training curves, policy comparisons, detailed agent metrics, and benchmark visualizations using matplotlib.

## CoGames Integration

Extends patterns from `scripts/run_evaluation.py` for consistent visualization style.

## Key Functions

### Basic Visualization

- `daf_plot_training_curves()` - Training progress from checkpoints
- `daf_plot_policy_comparison()` - Basic comparison bar charts
- `daf_plot_sweep_results()` - Sweep progress and best configuration
- `daf_export_comparison_html()` - Interactive HTML reports
- `daf_generate_leaderboard()` - Policy rankings in Markdown

### Detailed Metrics Visualization

- `daf_plot_detailed_metrics_comparison()` - Comprehensive agent metrics
- `daf_plot_sweep_heatmap()` - Performance heatmap across parameters
- `daf_plot_sweep_parallel_coordinates()` - Hyperparameter exploration
- `daf_generate_summary_dashboard()` - Combined dashboard

## Detailed Metrics Visualization

The `daf_plot_detailed_metrics_comparison()` function generates comprehensive plots for all agent metrics captured during evaluation.

### Usage

```python
from daf.visualization import daf_plot_detailed_metrics_comparison
from daf.comparison import daf_compare_policies

# Run comparison (populates policy_detailed_metrics)
report = daf_compare_policies(policies, missions)

# Generate detailed metrics plots
daf_plot_detailed_metrics_comparison(
    report,
    output_dir="comparisons/"
)
```

### Generated Files

| File | Description |
|------|-------------|
| `metrics_resources_gained.png` | Bar chart: carbon, silicon, oxygen, germanium gained |
| `metrics_resources_held.png` | Bar chart: current resource amounts |
| `metrics_energy.png` | Bar chart: energy amount, gained, lost |
| `metrics_actions.png` | Bar chart: move success/fail, noop, change_vibe |
| `metrics_inventory.png` | Bar chart: inventory diversity metrics |
| `metrics_radar.png` | Radar chart: key metrics comparison |
| `action_distribution.png` | Pie charts: action type distribution per policy |

### Metric Categories

The function automatically groups metrics into categories:

```python
metric_categories = {
    "Resources Gained": ["carbon.gained", "silicon.gained", "oxygen.gained", "germanium.gained"],
    "Resources Held": ["carbon.amount", "silicon.amount", "oxygen.amount", "germanium.amount"],
    "Energy": ["energy.amount", "energy.gained", "energy.lost"],
    "Actions": ["action.move.success", "action.move.failed", "action.noop.success", "action.change_vibe.success"],
    "Inventory": ["inventory.diversity", "inventory.diversity.ge.2", "inventory.diversity.ge.3", ...],
}
```

## HTML Report with Detailed Metrics

The `daf_export_comparison_html()` function includes detailed metrics tables when `policy_detailed_metrics` is available.

### HTML Sections

1. **Summary Statistics** - Overall rankings with highlighting
2. **Pairwise Comparisons** - Statistical significance tests
3. **Detailed Agent Metrics** - Tables for each category:
   - Resources (carbon, silicon, oxygen, germanium)
   - Energy (amount, gained, lost)
   - Actions (move, noop, change_vibe)
   - Inventory (diversity metrics)
   - Status (max_steps_without_motion)

### Winner Highlighting

Best values in each metric are highlighted:
- Higher is better: most metrics
- Lower is better: `action.failed`, `energy.lost`, `status.max_steps_without_motion`

## Sweep Visualization

### Sweep Progress

```python
from daf.visualization import daf_plot_sweep_results

daf_plot_sweep_results(sweep_result, output_dir="sweeps/")
```

**Generated Files:**
- `sweep_progress.png` - Trial performance with trend line
- `best_configuration.png` - Best hyperparameter values
- `hyperparameter_importance.png` - Correlation with performance

### Parameter Heatmap

```python
from daf.visualization import daf_plot_sweep_heatmap

daf_plot_sweep_heatmap(
    sweep_result,
    hp_x="learning_rate",
    hp_y="hidden_size",
    output_path="sweeps/heatmap.png"
)
```

### Parallel Coordinates

```python
from daf.visualization import daf_plot_sweep_parallel_coordinates

daf_plot_sweep_parallel_coordinates(
    sweep_result,
    output_path="sweeps/parallel.png",
    top_n=10  # Only show top 10 trials
)
```

## Dashboard Generation

Combine all visualizations into a single dashboard:

```python
from daf.visualization import daf_generate_summary_dashboard

files = daf_generate_summary_dashboard(
    sweep_result=sweep_result,
    comparison_report=comparison_report,
    output_dir="dashboard/"
)
```

**Generated Structure:**
```
dashboard/
├── dashboard.html
├── comparisons/
│   └── policy_rewards_comparison.png
└── sweeps/
    ├── best_configuration.png
    ├── param_heatmap.png
    ├── parallel_coords.png
    └── sweep_progress.png
```

## Color Palettes

DAF uses professional color palettes:

```python
PALETTE_COGAMES = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#f39c12", "#1abc9c", "#34495e", "#e67e22"]
```

## Testing

See `daf/tests/test_visualization.py` for coverage.

## See Also

- `daf/docs/API.md` - Complete API reference
- `daf/docs/INTEGRATION_MAP.md` - Full integration details
- `daf/docs/modules/comparison.md` - Comparison module
