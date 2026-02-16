"""Comparison pipeline orchestrator.

Chains environment validation → policy comparison → reporting into a workflow.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from daf.src.orchestrators import PipelineResult


def comparison_pipeline(
    policy_specs: list[Any],
    missions: list[tuple[str, Any]],
    episodes_per_mission: int = 5,
    generate_html_report: bool = True,
    generate_leaderboard: bool = True,
    output_dir: Optional[Path | str] = None,
    **kwargs: Any,
) -> PipelineResult:
    """Execute policy comparison pipeline with environment validation.

    Stages:
        1. environment_check - Validate environment and missions
        2. comparison - Run policy comparisons
        3. save_results - Save comparison results
        4. html_report - Generate HTML report (if enabled)
        5. leaderboard - Generate leaderboard (if enabled)

    Args:
        policy_specs: List of PolicySpec objects to compare
        missions: List of (mission_name, env_config) tuples
        episodes_per_mission: Number of episodes per mission per policy
        generate_html_report: Whether to generate HTML report
        generate_leaderboard: Whether to generate leaderboard
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

    mission_names = [m[0] if isinstance(m, tuple) else m for m in missions]

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
                pipeline_name="comparison",
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
            pipeline_name="comparison",
            status="failed",
            stages_completed=stages_completed,
            stages_failed=stages_failed,
            errors=errors,
            outputs=outputs,
            total_time_seconds=time.time() - start_time,
        )

    # Stage 2: Comparison
    try:
        from daf.src.eval.comparison import daf_compare_policies

        comparison_results = daf_compare_policies(
            policies=policy_specs,
            missions=mission_names,
            episodes_per_mission=episodes_per_mission,
        )
        outputs["comparison"] = {
            "num_policies": len(policy_specs),
            "num_missions": len(missions),
            "episodes_per_mission": episodes_per_mission,
            "results": comparison_results,
        }
        stages_completed.append("comparison")

    except Exception as e:
        stages_failed.append("comparison")
        errors.append(f"Comparison error: {e}")
        return PipelineResult(
            pipeline_name="comparison",
            status="partial",
            stages_completed=stages_completed,
            stages_failed=stages_failed,
            errors=errors,
            outputs=outputs,
            total_time_seconds=time.time() - start_time,
        )

    # Stage 3: Save Results
    if output_dir:
        try:
            import json

            results_path = output_dir / "comparison_results.json"
            with open(results_path, "w") as f:
                json.dump(
                    {
                        "num_policies": len(policy_specs),
                        "missions": mission_names,
                        "episodes_per_mission": episodes_per_mission,
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

    # Stage 4: HTML Report
    if generate_html_report and output_dir:
        try:
            from daf.src.viz.visualization import daf_export_comparison_html

            html_path = output_dir / "comparison_report.html"
            daf_export_comparison_html(
                outputs.get("comparison", {}).get("results"),
                output_path=str(html_path),
            )
            outputs["html_report"] = {"path": str(html_path)}
            stages_completed.append("html_report")

        except Exception as e:
            stages_failed.append("html_report")
            errors.append(f"HTML report error: {e}")

    # Stage 5: Leaderboard
    if generate_leaderboard and output_dir:
        try:
            leaderboard_path = output_dir / "leaderboard.json"
            import json

            with open(leaderboard_path, "w") as f:
                json.dump({"policies": len(policy_specs)}, f, indent=2)

            outputs["leaderboard"] = {"path": str(leaderboard_path)}
            stages_completed.append("leaderboard")

        except Exception as e:
            stages_failed.append("leaderboard")
            errors.append(f"Leaderboard error: {e}")

    total_time = time.time() - start_time
    status = "success" if not stages_failed else "partial"

    return PipelineResult(
        pipeline_name="comparison",
        status=status,
        stages_completed=stages_completed,
        stages_failed=stages_failed,
        errors=errors,
        outputs=outputs,
        total_time_seconds=total_time,
    )

