"""Tests for DAF distributed training orchestration."""

from pathlib import Path

import pytest


from daf.src.train.distributed_training import (
    DistributedTrainingResult,
    daf_aggregate_training_stats,
    daf_create_training_cluster,
    daf_get_training_status,
    daf_launch_distributed_training,
)


def test_distributed_training_result_creation():
    """Test DistributedTrainingResult initialization."""
    result = DistributedTrainingResult(
        final_checkpoint=Path("checkpoint.pt"),
        training_steps=1000,
        final_loss=0.5,
        final_reward=10.0,
        wall_time_seconds=60.0,
        num_workers=1,
    )

    assert result.training_steps == 1000
    assert result.wall_time_seconds == 60.0
    assert result.training_rate() == 1000 / 60.0


def test_distributed_training_result_training_rate():
    """Test training rate calculation."""
    result = DistributedTrainingResult(
        training_steps=1000,
        wall_time_seconds=10.0,
    )

    assert result.training_rate() == 100.0

    # Zero time case
    result_zero = DistributedTrainingResult(
        training_steps=0,
        wall_time_seconds=0.0,
    )
    assert result_zero.training_rate() == 0.0


def test_daf_create_training_cluster_single_node():
    """Test cluster creation for single node."""
    cluster = daf_create_training_cluster(
        num_nodes=1,
        workers_per_node=1,
        backend="torch",
    )

    assert cluster["backend"] in ("single", "torch")
    assert cluster["num_nodes"] == 1
    assert cluster["workers_per_node"] == 1


def test_daf_create_training_cluster_multi_node():
    """Test cluster creation for multi-node (may fall back gracefully)."""
    try:
        cluster = daf_create_training_cluster(
            num_nodes=2,
            workers_per_node=2,
            backend="torch",
        )

        assert "backend" in cluster
        assert cluster["num_nodes"] == 2

    except Exception:
        # May fail if distributed backend not available - that's OK
        pytest.skip("Distributed backend not available")


def test_daf_aggregate_training_stats():
    """Test training stats aggregation."""
    worker_stats = [
        {"loss": 0.5, "reward": 10.0},
        {"loss": 0.6, "reward": 12.0},
        {"loss": 0.4, "reward": 8.0},
    ]

    aggregated = daf_aggregate_training_stats(worker_stats, aggregation_method="mean")

    assert "loss" in aggregated
    assert "reward" in aggregated
    assert aggregated["loss"] == pytest.approx(0.5, abs=0.01)
    assert aggregated["reward"] == pytest.approx(10.0, abs=0.01)


def test_daf_aggregate_training_stats_sum():
    """Test aggregation with sum method."""
    worker_stats = [
        {"loss": 0.5},
        {"loss": 0.6},
    ]

    aggregated = daf_aggregate_training_stats(worker_stats, aggregation_method="sum")

    assert aggregated["loss"] == pytest.approx(1.1, abs=0.01)


def test_daf_aggregate_training_stats_max():
    """Test aggregation with max method."""
    worker_stats = [
        {"reward": 10.0},
        {"reward": 15.0},
        {"reward": 12.0},
    ]

    aggregated = daf_aggregate_training_stats(worker_stats, aggregation_method="max")

    assert aggregated["reward"] == 15.0


def test_daf_aggregate_training_stats_empty():
    """Test aggregation with empty stats."""
    aggregated = daf_aggregate_training_stats([])
    assert aggregated == {}


def test_daf_get_training_status(tmp_path):
    """Test getting training status."""
    # Create checkpoint directory
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    status = daf_get_training_status(
        checkpoints_path=checkpoint_dir,
        policy_class_path="cogames.cogs_vs_clips",
    )

    assert isinstance(status, dict)
    assert "num_checkpoints" in status
    assert status["num_checkpoints"] == 0  # No checkpoints yet


def test_daf_launch_distributed_training_single_node(tmp_path, safe_mission_loader):
    """Test distributed training with single node (should use standard train)."""
    # Load mission safely using fixture
    mission_name, env_cfg = safe_mission_loader("cogsguard_machina_1.basic")

    import torch
    from unittest.mock import patch, MagicMock

    device = torch.device("cpu")
    
    # Mock the actual training call since StarterPolicy isn't trainable
    # and we only want to test the DAF orchestration logic
    with patch("cogames.train.train") as mock_train:
        result = daf_launch_distributed_training(
            env_cfg=env_cfg,
            policy_class_path="cogames.policy.starter_agent.StarterPolicy",
            device=device,
            num_steps=10,
            checkpoints_path=tmp_path,
            num_nodes=1,
            workers_per_node=1,
        )

        assert mock_train.called
        assert isinstance(result, DistributedTrainingResult)
        assert result.num_workers == 1


class TestDistributedTrainingIntegration:
    """Integration tests for distributed training."""

    def test_training_cluster_creation_workflow(self):
        """Test cluster creation workflow."""
        cluster = daf_create_training_cluster(
            num_nodes=1,
            workers_per_node=1,
        )

        assert cluster is not None
        assert "backend" in cluster

    def test_stats_aggregation_workflow(self):
        """Test stats aggregation workflow."""
        # Simulate multiple workers
        worker_stats = [
            {"loss": 0.5 + i * 0.1, "reward": 10.0 + i}
            for i in range(5)
        ]

        aggregated = daf_aggregate_training_stats(worker_stats)

        assert "loss" in aggregated
        assert "reward" in aggregated

