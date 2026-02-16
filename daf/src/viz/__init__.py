"""DAF visualization: plots, dashboards, and GIF rendering.

Re-exports the primary visualization functions so consumers can do
``from daf.src.viz import daf_plot_policy_comparison``.
"""

# Lazy import to avoid pulling in matplotlib at package load
from __future__ import annotations

__all__ = [
    # visualization.py
    "daf_plot_training_curves",
    "daf_plot_learning_dynamics",
    "daf_plot_policy_comparison",
    "daf_plot_detailed_metrics_comparison",
    "daf_plot_policy_radar",
    "daf_plot_sweep_results",
    "daf_plot_sweep_heatmap",
    "daf_plot_sweep_parallel_coordinates",
    "daf_export_comparison_html",
    "daf_generate_leaderboard",
    "daf_generate_summary_dashboard",
    # gif_generator.py
    "generate_gif_from_replay",
    "generate_gif_from_simulation",
    "generate_frames_only",
    "load_replay",
    "get_mettascope_url",
]


def __getattr__(name: str):
    """Lazy import of visualization and gif_generator attributes."""
    import importlib

    _VIZ_NAMES = {
        "daf_plot_training_curves",
        "daf_plot_learning_dynamics",
        "daf_plot_policy_comparison",
        "daf_plot_detailed_metrics_comparison",
        "daf_plot_policy_radar",
        "daf_plot_sweep_results",
        "daf_plot_sweep_heatmap",
        "daf_plot_sweep_parallel_coordinates",
        "daf_export_comparison_html",
        "daf_generate_leaderboard",
        "daf_generate_summary_dashboard",
    }
    _GIF_NAMES = {
        "generate_gif_from_replay",
        "generate_gif_from_simulation",
        "generate_frames_only",
        "load_replay",
        "get_mettascope_url",
    }

    if name in _VIZ_NAMES:
        mod = importlib.import_module("daf.src.viz.visualization")
        return getattr(mod, name)
    if name in _GIF_NAMES:
        mod = importlib.import_module("daf.src.viz.gif_generator")
        return getattr(mod, name)
    raise AttributeError(f"module 'daf.src.viz' has no attribute '{name}'")
