"""Benchmark pipeline orchestrator.

Chains environment validation → benchmarking → reporting into a workflow.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from daf.src.orchestrators import PipelineResult


# Standard benchmark mission sets
BENCHMARK_SUITES = {
    "standard": [
        "training_facility_1",
        "training_facility_2",
        "assembler_2",
    ],
    "minimal": [
        "training_facility_1",
    ],
    "full": [
        "training_facility_1",
        "training_facility_2",
        "assembler_1",
        "assembler_2",
        "assembler_3",
    ],
}


def benchmark_pipeline(
    policy_specs: list[Any],
    benchmark_name: str = "standard",
    output_dir: Optional[Path | str] = None,
    episodes_per_mission: int = 5,
    **kwargs: Any,
) -> PipelineResult:
    """Execute benchmark pipeline with environment validation.

    Stages:
        1. environment_check - Validate environment and missions
        2. benchmark - Run benchmark evaluations
        3. save_results - Save benchmark results
        4. report - Generate benchmark report

    Args:
        policy_specs: List of PolicySpec objects to benchmark
        benchmark_name: Name of benchmark suite ("standard", "minimal", "full")
        output_dir: Directory for output files
        episodes_per_mission: Number of episodes per mission
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

    # Get benchmark missions
    missions = BENCHMARK_SUITES.get(benchmark_name, BENCHMARK_SUITES["standard"])

    # Stage 1: Environment Check
    try:
        from daf.src.environment_checks import daf_check_environment

        env_result = daf_check_environment(missions=missions)
        outputs["environment_check"] = {
            "healthy": env_result.is_healthy(),
            "warnings": env_result.warnings if hasattr(env_result, "warnings") else [],
            "benchmark_suite": benchmark_name,
            "missions": missions,
        }

        if not env_result.is_healthy():
            stages_failed.append("environment_check")
            errors.append(f"Environment check failed: {env_result.warnings}")
            return PipelineResult(
                pipeline_name="benchmark",
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
            pipeline_name="benchmark",
            status="failed",
            stages_completed=stages_completed,
            stages_failed=stages_failed,
            errors=errors,
            outputs=outputs,
            total_time_seconds=time.time() - start_time,
        )

    # Stage 2: Benchmark
    try:
        from daf.src.comparison import daf_compare_policies

        benchmark_results = daf_compare_policies(
            policies=policy_specs,
            missions=missions,
            episodes_per_mission=episodes_per_mission,
        )
        outputs["benchmark"] = {
            "benchmark_name": benchmark_name,
            "num_policies": len(policy_specs),
            "num_missions": len(missions),
            "episodes_per_mission": episodes_per_mission,
            "results": benchmark_results,
        }
        stages_completed.append("benchmark")

    except Exception as e:
        stages_failed.append("benchmark")
        errors.append(f"Benchmark error: {e}")
        return PipelineResult(
            pipeline_name="benchmark",
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

            results_path = output_dir / "benchmark_results.json"
            with open(results_path, "w") as f:
                json.dump(
                    {
                        "benchmark_name": benchmark_name,
                        "num_policies": len(policy_specs),
                        "missions": missions,
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

    # Stage 4: Report
    if output_dir:
        try:
            report_path = output_dir / "benchmark_report.md"
            with open(report_path, "w") as f:
                f.write(f"# Benchmark Report: {benchmark_name}\n\n")
                f.write(f"- Policies evaluated: {len(policy_specs)}\n")
                f.write(f"- Missions: {', '.join(missions)}\n")
                f.write(f"- Episodes per mission: {episodes_per_mission}\n")

            outputs["report"] = {"path": str(report_path)}
            stages_completed.append("report")

        except Exception as e:
            stages_failed.append("report")
            errors.append(f"Report error: {e}")

    total_time = time.time() - start_time
    status = "success" if not stages_failed else "partial"

    return PipelineResult(
        pipeline_name="benchmark",
        status=status,
        stages_completed=stages_completed,
        stages_failed=stages_failed,
        errors=errors,
        outputs=outputs,
        total_time_seconds=total_time,
    )

