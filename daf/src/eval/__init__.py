"""DAF evaluation: comparison, sweeps, mission & scoring analysis.

Submodules:
    comparison       Multi-policy comparison via cogames.evaluate
    sweeps           Hyperparameter sweep orchestration
    mission_analysis Mission performance analysis
    scoring_analysis VOR computation and scoring
    policy_analysis  Policy discovery and architecture analysis
"""

from __future__ import annotations

__all__ = [
    # comparison
    "daf_compare_policies",
    "daf_policy_ablation",
    "daf_benchmark_suite",
    "daf_compare_with_vor",
    "ComparisonReport",
    "PolicyComparisonResult",
    # sweeps
    "daf_launch_sweep",
    "daf_grid_search",
    "daf_random_search",
    "daf_sweep_best_config",
    "daf_sweep_status",
    "daf_variant_sweep",
    "SweepResult",
    "SweepTrialResult",
    # mission_analysis
    "daf_analyze_mission_performance",
    "daf_list_all_sites",
    "daf_list_all_variants",
    # scoring_analysis
    "daf_compute_vor",
    "daf_compute_weighted_score",
    "daf_allocate_agents",
    "daf_validate_proportions",
    "daf_scoring_summary",
    # policy_analysis
    "daf_list_available_policies",
    "daf_analyze_policy",
    "daf_compare_policy_architectures",
]

_MODULE_MAP = {
    "daf_compare_policies": "daf.src.eval.comparison",
    "daf_policy_ablation": "daf.src.eval.comparison",
    "daf_benchmark_suite": "daf.src.eval.comparison",
    "daf_compare_with_vor": "daf.src.eval.comparison",
    "ComparisonReport": "daf.src.eval.comparison",
    "PolicyComparisonResult": "daf.src.eval.comparison",
    "daf_launch_sweep": "daf.src.eval.sweeps",
    "daf_grid_search": "daf.src.eval.sweeps",
    "daf_random_search": "daf.src.eval.sweeps",
    "daf_sweep_best_config": "daf.src.eval.sweeps",
    "daf_sweep_status": "daf.src.eval.sweeps",
    "daf_variant_sweep": "daf.src.eval.sweeps",
    "SweepResult": "daf.src.eval.sweeps",
    "SweepTrialResult": "daf.src.eval.sweeps",
    "daf_analyze_mission_performance": "daf.src.eval.mission_analysis",
    "daf_list_all_sites": "daf.src.eval.mission_analysis",
    "daf_list_all_variants": "daf.src.eval.mission_analysis",
    "daf_compute_vor": "daf.src.eval.scoring_analysis",
    "daf_compute_weighted_score": "daf.src.eval.scoring_analysis",
    "daf_allocate_agents": "daf.src.eval.scoring_analysis",
    "daf_validate_proportions": "daf.src.eval.scoring_analysis",
    "daf_scoring_summary": "daf.src.eval.scoring_analysis",
    "daf_list_available_policies": "daf.src.eval.policy_analysis",
    "daf_analyze_policy": "daf.src.eval.policy_analysis",
    "daf_compare_policy_architectures": "daf.src.eval.policy_analysis",
}


def __getattr__(name: str):
    """Lazy import of eval submodule attributes."""
    if name in _MODULE_MAP:
        import importlib
        mod = importlib.import_module(_MODULE_MAP[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'daf.src.eval' has no attribute '{name}'")
