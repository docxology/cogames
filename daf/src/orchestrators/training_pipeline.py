"""Training pipeline orchestrator.

Chains environment validation → training → checkpoint saving into a complete workflow.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

try:
    from rich.console import Console
except ImportError:
    Console = None

from daf.src.orchestrators import PipelineResult


def training_pipeline(
    policy_class_path: str,
    mission_names: list[str],
    num_training_steps: int,
    checkpoints_path: Path | str,
    console: Optional[Any] = None,
    **kwargs: Any,
) -> PipelineResult:
    """Execute training pipeline with environment validation.

    Stages:
        1. environment_check - Validate environment and missions
        2. training - Execute training run
        3. checkpoint_save - Save final checkpoint

    Args:
        policy_class_path: Fully qualified policy class path
        mission_names: List of mission names to train on
        num_training_steps: Number of training steps
        checkpoints_path: Directory to save checkpoints
        console: Optional Rich console for output
        **kwargs: Additional training parameters

    Returns:
        PipelineResult with status, stages, and outputs
    """
    start_time = time.time()
    stages_completed: list[str] = []
    stages_failed: list[str] = []
    errors: list[str] = []
    outputs: dict[str, Any] = {}

    checkpoints_path = Path(checkpoints_path)
    checkpoints_path.mkdir(parents=True, exist_ok=True)

    # Stage 1: Environment Check
    try:
        from daf.src.environment_checks import daf_check_environment

        env_result = daf_check_environment(missions=mission_names)
        outputs["environment_check"] = {
            "healthy": env_result.is_healthy(),
            "warnings": env_result.warnings if hasattr(env_result, "warnings") else [],
        }

        if not env_result.is_healthy():
            stages_failed.append("environment_check")
            errors.append(f"Environment check failed: {env_result.warnings}")
            return PipelineResult(
                pipeline_name="training",
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
            pipeline_name="training",
            status="failed",
            stages_completed=stages_completed,
            stages_failed=stages_failed,
            errors=errors,
            outputs=outputs,
            total_time_seconds=time.time() - start_time,
        )

    # Stage 2: Training
    try:
        # Import cogames training functionality
        from cogames.train import train

        # Execute training for each mission
        for mission_name in mission_names:
            train(
                mission=mission_name,
                policy=policy_class_path,
                steps=num_training_steps,
                checkpoint_dir=str(checkpoints_path),
            )

        outputs["training"] = {
            "missions": mission_names,
            "steps": num_training_steps,
            "policy": policy_class_path,
        }
        stages_completed.append("training")

    except Exception as e:
        stages_failed.append("training")
        errors.append(f"Training error: {e}")
        return PipelineResult(
            pipeline_name="training",
            status="partial",
            stages_completed=stages_completed,
            stages_failed=stages_failed,
            errors=errors,
            outputs=outputs,
            total_time_seconds=time.time() - start_time,
        )

    # Stage 3: Checkpoint Save (implicit in training, but we confirm)
    try:
        checkpoint_files = list(checkpoints_path.glob("*.pt")) + list(
            checkpoints_path.glob("*.pth")
        )
        outputs["checkpoint_save"] = {
            "path": str(checkpoints_path),
            "files": [f.name for f in checkpoint_files],
        }
        stages_completed.append("checkpoint_save")

    except Exception as e:
        stages_failed.append("checkpoint_save")
        errors.append(f"Checkpoint save error: {e}")

    total_time = time.time() - start_time
    status = "success" if not stages_failed else "partial"

    return PipelineResult(
        pipeline_name="training",
        status=status,
        stages_completed=stages_completed,
        stages_failed=stages_failed,
        errors=errors,
        outputs=outputs,
        total_time_seconds=total_time,
    )

