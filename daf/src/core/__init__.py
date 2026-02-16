"""DAF core infrastructure: configuration, logging, and output management.

Submodules:
    config         Pydantic configuration models (DAFConfig, DAFSweepConfig, etc.)
    logging_config Structured logging with Rich integration
    output_manager Centralized output directory management
    output_utils   Output discovery, cleanup, and export utilities
"""

from __future__ import annotations

__all__ = [
    "DAFConfig",
    "DAFSweepConfig",
    "DAFDeploymentConfig",
    "DAFComparisonConfig",
    "DAFPipelineConfig",
    "DAFTournamentConfig",
    "DAFVariantConfig",
    "OutputManager",
    "get_output_manager",
    "DAFLogger",
    "create_daf_logger",
]

_MODULE_MAP = {
    "DAFConfig": "daf.src.core.config",
    "DAFSweepConfig": "daf.src.core.config",
    "DAFDeploymentConfig": "daf.src.core.config",
    "DAFComparisonConfig": "daf.src.core.config",
    "DAFPipelineConfig": "daf.src.core.config",
    "DAFTournamentConfig": "daf.src.core.config",
    "DAFVariantConfig": "daf.src.core.config",
    "OutputManager": "daf.src.core.output_manager",
    "get_output_manager": "daf.src.core.output_manager",
    "DAFLogger": "daf.src.core.logging_config",
    "create_daf_logger": "daf.src.core.logging_config",
}


def __getattr__(name: str):
    """Lazy import of core submodule attributes."""
    if name in _MODULE_MAP:
        import importlib
        mod = importlib.import_module(_MODULE_MAP[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'daf.src.core' has no attribute '{name}'")
