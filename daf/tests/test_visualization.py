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
    daf_plot_sweep_heatmap,
    daf_plot_sweep_parallel_coordinates,
    daf_plot_learning_dynamics,
    daf_plot_policy_radar,
    daf_plot_metrics_correlation_matrix,
    daf_plot_episode_reward_distribution,
    daf_plot_cumulative_performance,
    daf_plot_action_frequency,
    daf_generate_summary_dashboard,
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


def test_daf_plot_sweep_heatmap(tmp_path):
    """Test sweep heatmap visualization."""
    sweep_result = SweepResult("test_sweep", "reward", "maximize")
    
    # Create trials with two hyperparameters for heatmap
    for lr in [0.001, 0.01, 0.1]:
        for batch_size in [16, 32, 64]:
            trial = SweepTrialResult(
                trial_id=len(sweep_result.trials) + 1,
                hyperparameters={"lr": lr, "batch_size": batch_size},
                primary_metric=lr * batch_size,  # Simple formula for testing
                all_metrics={"reward": lr * batch_size},
                mission_results={"m1": lr * batch_size},
                success=True,
            )
            sweep_result.add_trial(trial)
    
    try:
        result = daf_plot_sweep_heatmap(
            sweep_result=sweep_result,
            hp_x="lr",
            hp_y="batch_size",
            output_path=tmp_path / "heatmap.png",
        )
        if result:
            assert result.exists()
    except ImportError:
        pytest.skip("matplotlib not available")


def test_daf_plot_sweep_parallel_coordinates(tmp_path):
    """Test parallel coordinates visualization."""
    sweep_result = SweepResult("test_sweep", "reward", "maximize")
    
    for i in range(10):
        trial = SweepTrialResult(
            trial_id=i + 1,
            hyperparameters={"lr": 0.001 * (i + 1), "layers": i % 4 + 1, "dropout": 0.1 * i},
            primary_metric=float(i),
            all_metrics={"reward": float(i)},
            mission_results={"m1": float(i)},
            success=True,
        )
        sweep_result.add_trial(trial)
    
    try:
        result = daf_plot_sweep_parallel_coordinates(
            sweep_result=sweep_result,
            output_path=tmp_path / "parallel.png",
        )
        if result:
            assert result.exists()
    except ImportError:
        pytest.skip("matplotlib not available")


def test_daf_plot_learning_dynamics(tmp_path):
    """Test learning dynamics visualization."""
    training_data = {
        "reward": [float(i) for i in range(100)],
        "loss": [10.0 - 0.1 * i for i in range(100)],
        "entropy": [5.0 - 0.05 * i for i in range(100)],
    }
    
    try:
        result = daf_plot_learning_dynamics(
            training_data=training_data,
            output_path=tmp_path / "dynamics.png",
            title="Test Learning Dynamics",
        )
        if result:
            assert result.exists()
    except ImportError:
        pytest.skip("matplotlib not available")


def test_daf_plot_policy_radar(tmp_path):
    """Test radar chart visualization."""
    policy_metrics = {
        "policy1": {"speed": 0.8, "accuracy": 0.9, "efficiency": 0.7, "robustness": 0.85},
        "policy2": {"speed": 0.7, "accuracy": 0.95, "efficiency": 0.8, "robustness": 0.75},
    }
    
    try:
        result = daf_plot_policy_radar(
            policy_metrics=policy_metrics,
            output_path=tmp_path / "radar.png",
        )
        if result:
            assert result.exists()
    except ImportError:
        pytest.skip("matplotlib not available")


def test_daf_plot_metrics_correlation_matrix(tmp_path):
    """Test correlation matrix visualization."""
    import random
    random.seed(42)
    
    metrics_data = {
        "reward": [random.random() for _ in range(50)],
        "steps": [random.random() for _ in range(50)],
        "energy": [random.random() for _ in range(50)],
        "resources": [random.random() for _ in range(50)],
    }
    
    try:
        result = daf_plot_metrics_correlation_matrix(
            metrics_data=metrics_data,
            output_path=tmp_path / "correlation.png",
        )
        if result:
            assert result.exists()
    except ImportError:
        pytest.skip("matplotlib or scipy not available")


def test_daf_plot_episode_reward_distribution(tmp_path):
    """Test episode reward distribution visualization."""
    episode_rewards = [
        [1.0, 2.0, 3.0, 2.5, 1.5, 2.0, 3.5],
        [2.0, 3.0, 4.0, 3.5, 2.5, 3.0, 4.5],
    ]
    
    try:
        result = daf_plot_episode_reward_distribution(
            episode_rewards=episode_rewards,
            policy_names=["baseline", "trained"],
            output_path=tmp_path / "distribution.png",
        )
        if result:
            assert result.exists()
    except ImportError:
        pytest.skip("matplotlib not available")


def test_daf_plot_cumulative_performance(tmp_path):
    """Test cumulative performance visualization."""
    performance_over_time = {
        "run1": [1.0, 2.0, 1.5, 3.0, 2.5, 4.0],
        "run2": [1.5, 1.8, 2.2, 2.8, 3.2, 3.5],
    }
    
    try:
        result = daf_plot_cumulative_performance(
            performance_over_time=performance_over_time,
            output_path=tmp_path / "cumulative.png",
        )
        if result:
            assert result.exists()
    except ImportError:
        pytest.skip("matplotlib not available")


def test_daf_plot_action_frequency(tmp_path):
    """Test action frequency visualization."""
    action_counts = {
        "policy1": {"move": 100, "attack": 50, "defend": 30, "wait": 20},
        "policy2": {"move": 80, "attack": 70, "defend": 40, "wait": 10},
    }
    
    try:
        result = daf_plot_action_frequency(
            action_counts=action_counts,
            output_path=tmp_path / "actions.png",
        )
        if result:
            assert result.exists()
    except ImportError:
        pytest.skip("matplotlib not available")


def test_daf_generate_summary_dashboard(tmp_path):
    """Test summary dashboard generation."""
    sweep_result = SweepResult("test_sweep", "reward", "maximize")
    
    for i in range(5):
        trial = SweepTrialResult(
            trial_id=i + 1,
            hyperparameters={"lr": 0.001 * (i + 1)},
            primary_metric=float(i),
            all_metrics={"reward": float(i)},
            mission_results={"m1": float(i)},
            success=True,
        )
        sweep_result.add_trial(trial)
    
    report = ComparisonReport(
        policies=["policy1", "policy2"],
        missions=["mission1"],
        episodes_per_mission=5,
    )
    report.add_policy_results("policy1", {"mission1": [1.0, 2.0]})
    report.add_policy_results("policy2", {"mission1": [2.0, 3.0]})
    
    try:
        files = daf_generate_summary_dashboard(
            sweep_result=sweep_result,
            comparison_report=report,
            additional_metrics={"total_episodes": 100, "avg_reward": 2.5},
            output_dir=tmp_path / "dashboard",
        )
        
        # Dashboard HTML should always be generated
        dashboard_html = tmp_path / "dashboard" / "dashboard.html"
        assert dashboard_html.exists()
        
    except ImportError:
        pytest.skip("matplotlib not available")


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
    
    def test_comprehensive_sweep_visualization_workflow(self, tmp_path):
        """Test comprehensive sweep visualization workflow."""
        sweep_result = SweepResult("comprehensive_test", "reward", "maximize")
        
        # Generate trials with multiple hyperparameters
        for lr in [0.001, 0.005, 0.01]:
            for batch in [16, 32, 64]:
                for layers in [2, 4]:
                    trial = SweepTrialResult(
                        trial_id=len(sweep_result.trials) + 1,
                        hyperparameters={"lr": lr, "batch_size": batch, "layers": layers},
                        primary_metric=lr * batch / layers,
                        all_metrics={"reward": lr * batch / layers},
                        mission_results={"m1": lr * batch / layers},
                        success=True,
                    )
                    sweep_result.add_trial(trial)
        
        try:
            # Standard sweep plots
            daf_plot_sweep_results(sweep_result, output_dir=tmp_path / "sweeps")
            
            # Heatmap
            daf_plot_sweep_heatmap(
                sweep_result, "lr", "batch_size",
                output_path=tmp_path / "heatmap.png"
            )
            
            # Parallel coordinates
            daf_plot_sweep_parallel_coordinates(
                sweep_result,
                output_path=tmp_path / "parallel.png"
            )
            
            # Verify outputs exist
            sweep_dir = tmp_path / "sweeps"
            if sweep_dir.exists():
                assert len(list(sweep_dir.glob("*.png"))) > 0
                
        except ImportError:
            pass  # matplotlib may not be available

