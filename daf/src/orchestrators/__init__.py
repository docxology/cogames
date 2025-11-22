"""DAF orchestrators for workflow management.

DAF sidecar utility: Chains DAF modules into complete workflows.

Each orchestrator (training, sweep, comparison, benchmark) invokes underlying
CoGames methods via DAF modules. Orchestrators are thin - they just sequence
stages and handle stage transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

__all__ = [
    "PipelineResult",
    "daf_run_training_pipeline",
    "daf_run_sweep_pipeline",
    "daf_run_comparison_pipeline",
    "daf_run_benchmark_pipeline",
]


@dataclass
class PipelineResult:
    """Result from orchestrated pipeline execution."""

    pipeline_name: str
    status: str  # "success", "failed", "partial"
    stages_completed: list[str]
    stages_failed: list[str]
    errors: list[str]
    outputs: dict[str, Any]
    total_time_seconds: float = 0.0
    timestamp: datetime = None

    def __post_init__(self) -> None:
        """Set default timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def is_success(self) -> bool:
        """Check if pipeline completed successfully.

        Returns:
            True if all stages completed without errors
        """
        return self.status == "success" and len(self.stages_failed) == 0


# Import pipeline functions AFTER defining PipelineResult to avoid circular imports
from daf.src.orchestrators.benchmark_pipeline import benchmark_pipeline  # noqa: E402
from daf.src.orchestrators.comparison_pipeline import comparison_pipeline  # noqa: E402
from daf.src.orchestrators.sweep_pipeline import sweep_pipeline  # noqa: E402
from daf.src.orchestrators.training_pipeline import training_pipeline  # noqa: E402

# DAF naming convention: daf_run_* prefix for pipeline orchestrators
daf_run_training_pipeline = training_pipeline
daf_run_sweep_pipeline = sweep_pipeline
daf_run_comparison_pipeline = comparison_pipeline
daf_run_benchmark_pipeline = benchmark_pipeline

