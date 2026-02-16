"""DAF (Distributed Agent Framework) module for CoGames.

Provides distributed training, evaluation, and analysis infrastructure
for agent policies in the Cogs vs Clips cooperative game environment.

Subpackages:
    daf.src.core          Configuration, logging, output management
    daf.src.eval          Policy comparison, sweeps, mission/scoring/policy analysis
    daf.src.train         Distributed training, deployment, authentication
    daf.src.viz           Visualization dashboards, GIF rendering
    daf.src.testing       Test runner, report generation
    daf.src.orchestrators Pipeline orchestration

Usage:
    from daf.src.core import DAFConfig, DAFSweepConfig
    from daf.src.eval import daf_compare_policies, daf_launch_sweep
    from daf.src.train import daf_launch_distributed_training, daf_deploy_policy
    from daf.src.viz import daf_plot_policy_comparison
    from daf.src.testing import TestRunner
    from daf.src.orchestrators import daf_run_training_pipeline

Note:
    Most modules require cogames and mettagrid to be installed.
    Import individual subpackages as needed to avoid loading all dependencies.
"""

from __future__ import annotations

_SUBMODULE_MAPPING = {
    "core": "daf.src.core",
    "eval": "daf.src.eval",
    "train": "daf.src.train",
    "viz": "daf.src.viz",
    "testing": "daf.src.testing",
    "orchestrators": "daf.src.orchestrators",
    "environment_checks": "daf.src.environment_checks",
}


def __getattr__(name: str):
    """Lazy import of subpackages."""
    if name in _SUBMODULE_MAPPING:
        import importlib

        module = importlib.import_module(_SUBMODULE_MAPPING[name])
        globals()[name] = module
        return module

    raise AttributeError(f"module 'daf' has no attribute '{name}'")


def __dir__():
    """List available attributes."""
    return list(_SUBMODULE_MAPPING.keys()) + ["__version__"]


__version__ = "2.2.0"

__all__ = [
    "core",
    "eval",
    "train",
    "viz",
    "testing",
    "orchestrators",
    "environment_checks",
    "__version__",
]
