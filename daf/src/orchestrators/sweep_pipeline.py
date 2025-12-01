"""Sweep pipeline orchestrator.

Chains environment validation → hyperparameter sweep → results saving into a workflow.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from daf.src.orchestrators import PipelineResult

if TYPE_CHECKING:
    from daf.src.config import DAFSweepConfig


def sweep_pipeline(
    sweep_config: DAFSweepConfig,
    save_results: bool = True,
    output_dir: Optional[Path | str] = None,
    **kwargs: Any,
) -> PipelineResult:
    """Execute hyperparameter sweep pipeline with environment validation.

    Stages:
        1. environment_check - Validate environment and missions
        2. sweep - Execute hyperparameter search
        3. save_results - Save sweep results (if enabled)
        4. visualization - Generate reports (if enabled)

    Args:
        sweep_config: DAFSweepConfig with sweep parameters
        save_results: Whether to save results to disk
        output_dir: Directory for output files
        **kwargs: Additional parameters

    Returns:
        PipelineResult with status, stages, and outputs
    """
    start_time = time.time()
    stages_completed: list[str] = []
    stages_failed: list[str] = []
    errors: list[str] = []
    outputs: dict[str, Any] = {}

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Environment Check
    try:
        from daf.src.environment_checks import daf_check_environment

        env_result = daf_check_environment(missions=sweep_config.missions)
        outputs["environment_check"] = {
            "healthy": env_result.is_healthy(),
            "warnings": env_result.warnings if hasattr(env_result, "warnings") else [],
        }

        if not env_result.is_healthy():
            stages_failed.append("environment_check")
            errors.append(f"Environment check failed: {env_result.warnings}")
            return PipelineResult(
                pipeline_name="sweep",
                status="failed",
                stages_completed=stages_completed,
                stages_failed=stages_failed,
                errors=errors,
                outputs=outputs,
                total_time_seconds=time.time() - start_time,
            )

        stages_completed.append("environment_check")

    except Exception as e:
        stages_failed.append("environment_check")
        errors.append(f"Environment check error: {e}")
        return PipelineResult(
            pipeline_name="sweep",
            status="failed",
            stages_completed=stages_completed,
            stages_failed=stages_failed,
            errors=errors,
            outputs=outputs,
            total_time_seconds=time.time() - start_time,
        )

    # Stage 2: Sweep
    try:
        from daf.src.sweeps import daf_launch_sweep

        sweep_results = daf_launch_sweep(sweep_config)
        outputs["sweep"] = {
            "name": sweep_config.name,
            "num_trials": sweep_config.num_trials,
            "strategy": sweep_config.strategy,
            "results": sweep_results,
        }
        stages_completed.append("sweep")

    except Exception as e:
        stages_failed.append("sweep")
        errors.append(f"Sweep error: {e}")
        return PipelineResult(
            pipeline_name="sweep",
            status="partial",
            stages_completed=stages_completed,
            stages_failed=stages_failed,
            errors=errors,
            outputs=outputs,
            total_time_seconds=time.time() - start_time,
        )

    # Stage 3: Save Results
    if save_results and output_dir:
        try:
            import json

            results_path = output_dir / "sweep_results.json"
            with open(results_path, "w") as f:
                json.dump(
                    {
                        "name": sweep_config.name,
                        "config": sweep_config.model_dump()
                        if hasattr(sweep_config, "model_dump")
                        else {},
                    },
                    f,
                    indent=2,
                    default=str,
                )

            outputs["save_results"] = {"path": str(results_path)}
            stages_completed.append("save_results")

        except Exception as e:
            stages_failed.append("save_results")
            errors.append(f"Save results error: {e}")

    total_time = time.time() - start_time
    status = "success" if not stages_failed else "partial"

    return PipelineResult(
        pipeline_name="sweep",
        status=status,
        stages_completed=stages_completed,
        stages_failed=stages_failed,
        errors=errors,
        outputs=outputs,
        total_time_seconds=total_time,
    )

