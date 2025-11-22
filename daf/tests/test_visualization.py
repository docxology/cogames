"""Tests for DAF visualization functionality."""

from pathlib import Path

import pytest

from daf.src.comparison import ComparisonReport
from daf.src.sweeps import SweepResult, SweepTrialResult
from daf.src.visualization import (
    daf_export_comparison_html,
    daf_generate_leaderboard,
    daf_plot_policy_comparison,
    daf_plot_sweep_results,
    daf_plot_training_curves,
)


def test_daf_plot_training_curves_no_matplotlib(tmp_path):
    """Test training curves plotting (may skip if matplotlib unavailable)."""
    try:
        daf_plot_training_curves(
            checkpoint_dir=tmp_path,
            output_dir=tmp_path / "plots",
        )
        # If matplotlib available, should complete without error
    except ImportError:
        pytest.skip("matplotlib not available")


def test_daf_plot_policy_comparison(tmp_path):
    """Test policy comparison plotting."""
    report = ComparisonReport(
        policies=["policy1", "policy2"],
        missions=["mission1"],
        episodes_per_mission=5,
    )

    report.add_policy_results("policy1", {"mission1": [1.0, 2.0]})
    report.add_policy_results("policy2", {"mission1": [2.0, 3.0]})

    try:
        daf_plot_policy_comparison(
            comparison_report=report,
            output_dir=tmp_path / "plots",
        )

        # Check if plots were created
        plot_dir = tmp_path / "plots"
        if plot_dir.exists():
            plot_files = list(plot_dir.glob("*.png"))
            assert len(plot_files) > 0

    except ImportError:
        pytest.skip("matplotlib not available")


def test_daf_plot_sweep_results(tmp_path):
    """Test sweep results plotting."""
    sweep_result = SweepResult("test_sweep", "reward", "maximize")

    for i in range(1, 4):
        trial = SweepTrialResult(
            trial_id=i,
            hyperparameters={"lr": 0.001 * i},
            primary_metric=float(i),
            all_metrics={"reward": float(i)},
            mission_results={"m1": float(i)},
            success=True,
        )
        sweep_result.add_trial(trial)

    try:
        daf_plot_sweep_results(
            sweep_result=sweep_result,
            output_dir=tmp_path / "plots",
        )

        plot_dir = tmp_path / "plots"
        if plot_dir.exists():
            plot_files = list(plot_dir.glob("*.png"))
            assert len(plot_files) > 0

    except ImportError:
        pytest.skip("matplotlib not available")


def test_daf_plot_sweep_results_empty(tmp_path):
    """Test plotting with empty sweep results."""
    sweep_result = SweepResult("empty", "reward", "maximize")

    try:
        daf_plot_sweep_results(
            sweep_result=sweep_result,
            output_dir=tmp_path / "plots",
        )
        # Should handle gracefully
    except ImportError:
        pytest.skip("matplotlib not available")


def test_daf_export_comparison_html(tmp_path):
    """Test HTML report export."""
    report = ComparisonReport(
        policies=["policy1", "policy2"],
        missions=["mission1"],
        episodes_per_mission=5,
    )

    report.add_policy_results("policy1", {"mission1": [1.0, 2.0]})
    report.add_policy_results("policy2", {"mission1": [2.0, 3.0]})
    report.compute_pairwise_comparisons()

    html_path = tmp_path / "comparison.html"
    daf_export_comparison_html(
        comparison_report=report,
        output_path=html_path,
    )

    assert html_path.exists()

    # Check HTML content
    content = html_path.read_text()
    assert "Policy Comparison Report" in content
    assert "policy1" in content or "policy2" in content


def test_daf_export_comparison_html_empty_report(tmp_path):
    """Test HTML export with empty report."""
    report = ComparisonReport(
        policies=[],
        missions=[],
        episodes_per_mission=5,
    )

    html_path = tmp_path / "empty.html"
    daf_export_comparison_html(
        comparison_report=report,
        output_path=html_path,
    )

    assert html_path.exists()


def test_daf_generate_leaderboard(tmp_path):
    """Test leaderboard generation."""
    report = ComparisonReport(
        policies=["policy1", "policy2", "policy3"],
        missions=["mission1"],
        episodes_per_mission=5,
    )

    report.add_policy_results("policy1", {"mission1": [1.0]})
    report.add_policy_results("policy2", {"mission1": [3.0]})
    report.add_policy_results("policy3", {"mission1": [2.0]})

    leaderboard_path = tmp_path / "leaderboard.json"
    markdown = daf_generate_leaderboard(
        comparison_report=report,
        output_path=leaderboard_path,
    )

    assert isinstance(markdown, str)
    assert "Rank" in markdown
    assert "Policy" in markdown

    if leaderboard_path.exists():
        import json

        with open(leaderboard_path, "r") as f:
            data = json.load(f)

        assert "leaderboard" in data
        assert len(data["leaderboard"]) == 3
        # Should be sorted by performance
        assert data["leaderboard"][0]["policy"] == "policy2"  # Highest reward


def test_daf_generate_leaderboard_no_output(tmp_path):
    """Test leaderboard generation without saving."""
    report = ComparisonReport(
        policies=["policy1"],
        missions=["mission1"],
        episodes_per_mission=5,
    )

    report.add_policy_results("policy1", {"mission1": [1.0]})

    markdown = daf_generate_leaderboard(
        comparison_report=report,
        output_path=None,
    )

    assert isinstance(markdown, str)
    assert "policy1" in markdown


class TestVisualizationIntegration:
    """Integration tests for visualization."""

    def test_complete_comparison_visualization_workflow(self, tmp_path):
        """Test complete comparison visualization workflow."""
        report = ComparisonReport(
            policies=["policy1", "policy2"],
            missions=["mission1", "mission2"],
            episodes_per_mission=5,
        )

        report.add_policy_results("policy1", {"mission1": [1.0, 2.0], "mission2": [1.5, 2.5]})
        report.add_policy_results("policy2", {"mission1": [2.0, 3.0], "mission2": [2.5, 3.5]})
        report.compute_pairwise_comparisons()

        # Generate plots
        try:
            daf_plot_policy_comparison(report, output_dir=tmp_path / "plots")
        except ImportError:
            pass  # matplotlib may not be available

        # Generate HTML
        html_path = tmp_path / "report.html"
        daf_export_comparison_html(report, html_path)
        assert html_path.exists()

        # Generate leaderboard
        leaderboard_path = tmp_path / "leaderboard.json"
        daf_generate_leaderboard(report, output_path=leaderboard_path)
        assert leaderboard_path.exists()

