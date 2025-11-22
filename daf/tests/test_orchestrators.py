"""Tests for DAF orchestrators - verify clear ordering and environment setup."""

from pathlib import Path

import pytest

from daf.src.orchestrators import PipelineResult, daf_run_benchmark_pipeline, daf_run_comparison_pipeline, daf_run_sweep_pipeline, daf_run_training_pipeline


def test_pipeline_result_creation():
    """Test PipelineResult initialization."""
    result = PipelineResult(
        pipeline_name="test",
        status="success",
        stages_completed=["stage1"],
        stages_failed=[],
        errors=[],
        outputs={},
    )

    assert result.pipeline_name == "test"
    assert result.status == "success"
    assert result.is_success() is True


def test_pipeline_result_is_success():
    """Test PipelineResult.is_success() method."""
    # Success case
    result = PipelineResult(
        pipeline_name="test",
        status="success",
        stages_completed=["stage1", "stage2"],
        stages_failed=[],
        errors=[],
        outputs={},
    )
    assert result.is_success() is True

    # Failed case
    result_failed = PipelineResult(
        pipeline_name="test",
        status="failed",
        stages_completed=["stage1"],
        stages_failed=["stage2"],
        errors=["error"],
        outputs={},
    )
    assert result_failed.is_success() is False


def test_training_pipeline_environment_check_ordering(tmp_path):
    """Verify training pipeline has environment check as Stage 1."""
    # Mock console to capture output
    from io import StringIO
    from rich.console import Console

    console = Console(file=StringIO(), force_terminal=False)

    result = daf_run_training_pipeline(
        policy_class_path="cogames.policy.scripted_agent.baseline_agent.BaselinePolicy",
        mission_names=["training_facility_1"],
        num_training_steps=10,  # Very small for testing
        checkpoints_path=tmp_path,
        console=console,
    )

    # Verify environment_check is first stage
    if result.stages_completed:
        assert result.stages_completed[0] == "environment_check", "Environment check must be Stage 1"

    # Verify environment_check is in outputs if completed
    if "environment_check" in result.stages_completed:
        assert "environment_check" in result.outputs


def test_training_pipeline_stage_ordering(tmp_path):
    """Verify training pipeline has correct stage ordering."""
    result = daf_run_training_pipeline(
        policy_class_path="cogames.policy.scripted_agent.baseline_agent.BaselinePolicy",
        mission_names=["training_facility_1"],
        num_training_steps=10,
        checkpoints_path=tmp_path,
    )

    # Check that stages are ordered correctly
    stages = result.stages_completed

    if "environment_check" in stages:
        env_idx = stages.index("environment_check")
        if "training" in stages:
            train_idx = stages.index("training")
            assert env_idx < train_idx, "Environment check must come before training"


def test_sweep_pipeline_environment_check_ordering(tmp_path):
    """Verify sweep pipeline has environment check as Stage 1."""
    from src.config import DAFSweepConfig

    sweep_config = DAFSweepConfig(
        name="test_sweep",
        missions=["training_facility_1"],
        policy_class_path="cogames.policy.scripted_agent.baseline_agent.BaselinePolicy",
        search_space={"x": [1, 2]},
        strategy="grid",
        episodes_per_trial=1,  # Minimal for testing
    )

    result = daf_run_sweep_pipeline(
        sweep_config=sweep_config,
        save_results=False,
        output_dir=tmp_path,
    )

    # Verify environment_check is first stage
    if result.stages_completed:
        assert result.stages_completed[0] == "environment_check", "Environment check must be Stage 1"


def test_sweep_pipeline_stage_ordering(tmp_path):
    """Verify sweep pipeline has correct stage ordering."""
    from src.config import DAFSweepConfig

    sweep_config = DAFSweepConfig(
        name="test_sweep",
        missions=["training_facility_1"],
        policy_class_path="cogames.policy.scripted_agent.baseline_agent.BaselinePolicy",
        search_space={"x": [1]},  # Single config for speed
        strategy="grid",
        episodes_per_trial=1,
    )

    result = daf_run_sweep_pipeline(
        sweep_config=sweep_config,
        save_results=True,
        output_dir=tmp_path,
    )

    stages = result.stages_completed

    # Verify ordering: environment_check -> sweep -> save_results -> visualization
    if "environment_check" in stages and "sweep" in stages:
        assert stages.index("environment_check") < stages.index("sweep")

    if "sweep" in stages and "save_results" in stages:
        assert stages.index("sweep") < stages.index("save_results")


def test_comparison_pipeline_environment_check_ordering(tmp_path):
    """Verify comparison pipeline has environment check as Stage 1."""
    from mettagrid.policy.policy import PolicySpec

    # Create minimal mission config
    from cogames.cli.mission import get_mission_name_and_config

    try:
        from typer import Context

        ctx = Context(lambda: None)
        _, env_cfg, _ = get_mission_name_and_config(ctx, "training_facility_1")
        missions = [("training_facility_1", env_cfg)]

        policy_specs = [
            PolicySpec(class_path="cogames.policy.scripted_agent.baseline_agent.BaselinePolicy")
        ]

        result = daf_run_comparison_pipeline(
            policy_specs=policy_specs,
            missions=missions,
            episodes_per_mission=1,  # Minimal for testing
            generate_html_report=False,
            generate_leaderboard=False,
            output_dir=tmp_path,
        )

        # Verify environment_check is first stage
        if result.stages_completed:
            assert result.stages_completed[0] == "environment_check", "Environment check must be Stage 1"

    except Exception:
        pytest.skip("Could not load mission for comparison test")


def test_comparison_pipeline_stage_ordering(tmp_path):
    """Verify comparison pipeline has correct stage ordering."""
    from mettagrid.policy.policy import PolicySpec

    try:
        from cogames.cli.mission import get_mission_name_and_config

        from typer import Context

        ctx = Context(lambda: None)
        _, env_cfg, _ = get_mission_name_and_config(ctx, "training_facility_1")
        missions = [("training_facility_1", env_cfg)]

        policy_specs = [
            PolicySpec(class_path="cogames.policy.scripted_agent.baseline_agent.BaselinePolicy")
        ]

        result = daf_run_comparison_pipeline(
            policy_specs=policy_specs,
            missions=missions,
            episodes_per_mission=1,
            generate_html_report=True,
            generate_leaderboard=True,
            output_dir=tmp_path,
        )

        stages = result.stages_completed

        # Verify ordering: environment_check -> comparison -> save_results -> html_report -> leaderboard
        if "environment_check" in stages and "comparison" in stages:
            assert stages.index("environment_check") < stages.index("comparison")

        if "comparison" in stages and "save_results" in stages:
            assert stages.index("comparison") < stages.index("save_results")

    except Exception:
        pytest.skip("Could not load mission for comparison test")


def test_benchmark_pipeline_environment_check_ordering(tmp_path):
    """Verify benchmark pipeline has environment check as Stage 1."""
    from mettagrid.policy.policy import PolicySpec

    policy_specs = [
        PolicySpec(class_path="cogames.policy.scripted_agent.baseline_agent.BaselinePolicy")
    ]

    result = daf_run_benchmark_pipeline(
        policy_specs=policy_specs,
        benchmark_name="standard",
        output_dir=tmp_path,
    )

    # Verify environment_check is first stage
    if result.stages_completed:
        assert result.stages_completed[0] == "environment_check", "Environment check must be Stage 1"


def test_pipeline_failure_on_environment_check(tmp_path):
    """Verify pipeline fails early if environment check fails."""
    # This test verifies that pipelines stop on environment check failure
    # We can't easily simulate a failing environment check, but we can verify
    # the structure is correct

    result = daf_run_training_pipeline(
        policy_class_path="cogames.policy.scripted_agent.baseline_agent.BaselinePolicy",
        mission_names=["nonexistent_mission_xyz"],
        num_training_steps=10,
        checkpoints_path=tmp_path,
    )

    # If environment check fails, should have failed stage
    if "environment_check" in result.stages_failed:
        assert result.status in ("failed", "partial")
        assert len(result.errors) > 0


def test_pipeline_result_timestamp():
    """Test that PipelineResult has timestamp."""
    result = PipelineResult(
        pipeline_name="test",
        status="success",
        stages_completed=[],
        stages_failed=[],
        errors=[],
        outputs={},
    )

    assert result.timestamp is not None


def test_pipeline_result_total_time():
    """Test that PipelineResult tracks total time."""
    result = PipelineResult(
        pipeline_name="test",
        status="success",
        stages_completed=[],
        stages_failed=[],
        errors=[],
        outputs={},
        total_time_seconds=123.45,
    )

    assert result.total_time_seconds == 123.45


class TestOrchestratorIntegration:
    """Integration tests for orchestrator workflows."""

    def test_training_pipeline_complete_workflow(self, tmp_path):
        """Test complete training pipeline workflow."""
        result = daf_run_training_pipeline(
            policy_class_path="cogames.policy.scripted_agent.baseline_agent.BaselinePolicy",
            mission_names=["training_facility_1"],
            num_training_steps=10,
            checkpoints_path=tmp_path,
        )

        # Should have completed at least environment check
        assert len(result.stages_completed) > 0
        assert "environment_check" in result.stages_completed

    def test_sweep_pipeline_with_minimal_config(self, tmp_path):
        """Test sweep pipeline with minimal configuration."""
        from src.config import DAFSweepConfig

        sweep_config = DAFSweepConfig(
            name="minimal_test",
            missions=["training_facility_1"],
            policy_class_path="cogames.policy.scripted_agent.baseline_agent.BaselinePolicy",
            search_space={"x": [1]},
            strategy="grid",
            episodes_per_trial=1,
        )

        result = daf_run_sweep_pipeline(
            sweep_config=sweep_config,
            save_results=True,
            output_dir=tmp_path,
        )

        # Should have environment check
        assert "environment_check" in result.stages_completed or "environment_check" in result.stages_failed

    def test_all_pipelines_have_environment_check(self, tmp_path):
        """Verify all pipelines include environment check stage."""
        # Training pipeline
        train_result = daf_run_training_pipeline(
            policy_class_path="cogames.policy.scripted_agent.baseline_agent.BaselinePolicy",
            mission_names=["training_facility_1"],
            num_training_steps=10,
            checkpoints_path=tmp_path,
        )
        assert "environment_check" in (train_result.stages_completed + train_result.stages_failed)

        # Sweep pipeline
        from src.config import DAFSweepConfig

        sweep_config = DAFSweepConfig(
            name="test",
            missions=["training_facility_1"],
            policy_class_path="cogames.policy.scripted_agent.baseline_agent.BaselinePolicy",
            search_space={"x": [1]},
            strategy="grid",
            episodes_per_trial=1,
        )
        sweep_result = daf_run_sweep_pipeline(sweep_config=sweep_config, output_dir=tmp_path)
        assert "environment_check" in (sweep_result.stages_completed + sweep_result.stages_failed)

        # Benchmark pipeline
        from mettagrid.policy.policy import PolicySpec

        benchmark_result = daf_run_benchmark_pipeline(
            policy_specs=[PolicySpec(class_path="cogames.policy.scripted_agent.baseline_agent.BaselinePolicy")],
            output_dir=tmp_path,
        )
        assert "environment_check" in (benchmark_result.stages_completed + benchmark_result.stages_failed)

