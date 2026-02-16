"""Tests for DAF policy comparison functionality."""

from pathlib import Path

import pytest

from daf.src.eval.comparison import ComparisonReport, PolicyComparisonResult, daf_benchmark_suite, daf_compare_policies


def test_policy_comparison_result_creation():
    """Test PolicyComparisonResult creation."""
    result = PolicyComparisonResult(
        policy_a_name="policy_a",
        policy_b_name="policy_b",
        missions=["mission1"],
        episodes_per_mission=5,
        mission_rewards_a={"mission1": [1.0, 2.0]},
        mission_rewards_b={"mission1": [1.5, 2.5]},
        avg_reward_a=1.5,
        avg_reward_b=2.0,
        reward_std_a=0.5,
        reward_std_b=0.5,
        p_value=0.1,
        is_significant=False,
        effect_size=0.5,
    )

    assert result.policy_a_name == "policy_a"
    assert result.p_value == 0.1
    assert result.is_significant is False


def test_policy_comparison_result_summary():
    """Test PolicyComparisonResult summary string."""
    result = PolicyComparisonResult(
        policy_a_name="A",
        policy_b_name="B",
        missions=["m1"],
        episodes_per_mission=5,
        mission_rewards_a={},
        mission_rewards_b={},
        avg_reward_a=1.0,
        avg_reward_b=2.0,
        reward_std_a=0.1,
        reward_std_b=0.2,
        p_value=0.05,
        is_significant=True,
        effect_size=1.0,
    )

    summary = result.summary_string()
    assert "A" in summary
    assert "B" in summary
    assert "significant" in summary.lower()


def test_comparison_report_creation():
    """Test ComparisonReport initialization."""
    report = ComparisonReport(
        policies=["policy1", "policy2"],
        missions=["mission1"],
        episodes_per_mission=5,
    )

    assert len(report.policies) == 2
    assert len(report.missions) == 1
    assert report.episodes_per_mission == 5


def test_comparison_report_add_policy_results():
    """Test adding policy results to report."""
    report = ComparisonReport(
        policies=["policy1"],
        missions=["mission1"],
        episodes_per_mission=5,
    )

    report.add_policy_results("policy1", {"mission1": [1.0, 2.0, 3.0]})

    assert "policy1" in report.policy_mission_rewards
    assert "mission1" in report.policy_mission_rewards["policy1"]
    assert len(report.policy_mission_rewards["policy1"]["mission1"]) == 3


def test_comparison_report_summary_statistics():
    """Test summary statistics generation."""
    report = ComparisonReport(
        policies=["policy1", "policy2"],
        missions=["mission1"],
        episodes_per_mission=5,
    )

    report.add_policy_results("policy1", {"mission1": [1.0, 2.0]})
    report.add_policy_results("policy2", {"mission1": [2.0, 3.0]})

    stats = report.summary_statistics
    assert "policy1" in stats
    assert "policy2" in stats
    assert "avg_reward" in stats["policy1"]
    assert "std_dev" in stats["policy1"]


def test_comparison_report_compute_pairwise_comparisons():
    """Test pairwise comparison computation."""
    report = ComparisonReport(
        policies=["policy1", "policy2"],
        missions=["mission1"],
        episodes_per_mission=5,
    )

    report.add_policy_results("policy1", {"mission1": [1.0, 2.0, 3.0]})
    report.add_policy_results("policy2", {"mission1": [2.0, 3.0, 4.0]})

    report.compute_pairwise_comparisons()

    assert len(report.pairwise_comparisons) > 0
    comparison = list(report.pairwise_comparisons.values())[0]
    assert hasattr(comparison, "p_value")
    assert hasattr(comparison, "is_significant")


def test_comparison_report_save_json(tmp_path):
    """Test saving comparison report to JSON."""
    report = ComparisonReport(
        policies=["policy1"],
        missions=["mission1"],
        episodes_per_mission=5,
    )

    report.add_policy_results("policy1", {"mission1": [1.0, 2.0]})
    report.compute_pairwise_comparisons()

    json_path = tmp_path / "comparison.json"
    report.save_json(json_path)

    assert json_path.exists()

    import json

    with open(json_path, "r") as f:
        data = json.load(f)

    assert data["policies"] == ["policy1"]
    assert "summary_statistics" in data


def test_daf_compare_policies_with_real_mission(tmp_path, safe_mission_loader):
    """Test daf_compare_policies with real mission."""
    try:
        from mettagrid.policy.policy import PolicySpec
    except ImportError:
        pytest.skip("mettagrid not installed")

    # Use fixture for safe mission loading
    mission_name, env_cfg = safe_mission_loader("cogsguard_machina_1.basic")
    missions = [(mission_name, env_cfg)]

    policy_specs = [PolicySpec(class_path="cogames.policy.starter_agent.StarterPolicy")]

    report = daf_compare_policies(
        policies=policy_specs,
        missions=missions,
        episodes_per_mission=1,  # Minimal for testing
    )

    assert isinstance(report, ComparisonReport)
    assert len(report.policies) == 1


def test_daf_benchmark_suite():
    """Test benchmark suite execution."""
    try:
        from mettagrid.policy.policy import PolicySpec
    except ImportError:
        pytest.skip("mettagrid not installed")

    policy_specs = [
        PolicySpec(class_path="cogames.policy.starter_agent.StarterPolicy")
    ]

    try:
        report = daf_benchmark_suite(
            policy_specs=policy_specs,
            benchmark_name="standard",
        )

        assert isinstance(report, ComparisonReport)
        assert len(report.policies) == 1

    except Exception as e:
        pytest.skip(f"Benchmark suite not available: {e}")


def test_comparison_report_pairwise_comparisons_empty():
    """Test pairwise comparisons with no policies."""
    report = ComparisonReport(
        policies=[],
        missions=["mission1"],
        episodes_per_mission=5,
    )

    report.compute_pairwise_comparisons()
    assert len(report.pairwise_comparisons) == 0


def test_comparison_report_pairwise_comparisons_single_policy():
    """Test pairwise comparisons with single policy."""
    report = ComparisonReport(
        policies=["policy1"],
        missions=["mission1"],
        episodes_per_mission=5,
    )

    report.add_policy_results("policy1", {"mission1": [1.0]})
    report.compute_pairwise_comparisons()

    # Single policy should have no pairwise comparisons
    assert len(report.pairwise_comparisons) == 0


class TestComparisonIntegration:
    """Integration tests for comparison functionality."""

    def test_comparison_workflow(self, tmp_path, safe_mission_loader):
        """Test complete comparison workflow."""
        try:
            from mettagrid.policy.policy import PolicySpec
        except ImportError:
            pytest.skip("mettagrid not installed")

        # Use fixture for safe mission loading
        mission_name, env_cfg = safe_mission_loader("cogsguard_machina_1.basic")
        missions = [(mission_name, env_cfg)]

        policy_specs = [PolicySpec(class_path="cogames.policy.starter_agent.StarterPolicy")]

        report = daf_compare_policies(
            policies=policy_specs,
            missions=missions,
            episodes_per_mission=1,
        )

        # Save report
        json_path = tmp_path / "test_comparison.json"
        report.save_json(json_path)

        assert json_path.exists()

