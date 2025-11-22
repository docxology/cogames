# DAF Visualization Module

**DAF sidecar utility: Visualization extending patterns from `scripts/run_evaluation.py`.**

## Overview

Training curves, policy comparisons, and benchmark visualizations using matplotlib.

## CoGames Integration

Extends patterns from `scripts/run_evaluation.py` for consistent visualization style.

## Key Functions

- `daf_plot_training_curves()` - Training progress
- `daf_plot_policy_comparison()` - Comparison plots
- `daf_plot_sweep_results()` - Sweep progress
- `daf_export_comparison_html()` - HTML reports
- `daf_generate_leaderboard()` - Policy rankings

## Testing

See `daf/tests/test_visualization.py` for coverage.

## See Also

- `daf/docs/INTEGRATION_MAP.md` - Full integration details
