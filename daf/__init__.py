"""DAF (Distributed Agent Framework) module for CoGames.

Provides distributed training, evaluation, and analysis infrastructure
for agent policies in the Cogs vs Clips cooperative game environment.

Usage:
    from daf import sweeps, comparison, environment_checks
    from daf.config import DAFSweepConfig, DAFConfig
    from daf.orchestrators import daf_run_training_pipeline

Note:
    Most modules require cogames and mettagrid to be installed.
    Import individual modules as needed to avoid loading all dependencies.
"""

from __future__ import annotations

# Lazy imports using module-level __getattr__ for cleaner import experience
# This avoids loading all heavy dependencies at import time

_SUBMODULE_MAPPING = {
    # Core modules
    "sweeps": "daf.src.sweeps",
    "comparison": "daf.src.comparison",
    "deployment": "daf.src.deployment",
    "distributed_training": "daf.src.distributed_training",
    "environment_checks": "daf.src.environment_checks",
    "mission_analysis": "daf.src.mission_analysis",
    "visualization": "daf.src.visualization",
    # Infrastructure modules
    "config": "daf.src.config",
    "output_manager": "daf.src.output_manager",
    "output_utils": "daf.src.output_utils",
    "logging_config": "daf.src.logging_config",
    "test_runner": "daf.src.test_runner",
    "orchestrators": "daf.src.orchestrators",
}


def __getattr__(name: str):
    """Lazy import of submodules."""
    if name in _SUBMODULE_MAPPING:
        import importlib

        module = importlib.import_module(_SUBMODULE_MAPPING[name])
        globals()[name] = module
        return module

    raise AttributeError(f"module 'daf' has no attribute '{name}'")


def __dir__():
    """List available attributes."""
    return list(_SUBMODULE_MAPPING.keys()) + ["__version__"]


__version__ = "2.0.0"

__all__ = [
    # Modules (accessed via daf.sweeps, daf.comparison, etc.)
    "sweeps",
    "comparison",
    "deployment",
    "distributed_training",
    "environment_checks",
    "mission_analysis",
    "visualization",
    "config",
    "output_manager",
    "output_utils",
    "logging_config",
    "test_runner",
    "orchestrators",
    # Version
    "__version__",
]
