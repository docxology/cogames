"""Tests for DAF hyperparameter sweeps."""

import json
from pathlib import Path

import pytest

from daf.src.eval.sweeps import (
    SweepResult,
    SweepTrialResult,
    daf_grid_search,
    daf_random_search,
    daf_sweep_best_config,
    daf_sweep_status,
)


def test_sweep_trial_result():
    """Test SweepTrialResult creation."""
    trial = SweepTrialResult(
        trial_id=1,
        hyperparameters={"lr": 0.001},
        primary_metric=0.5,
        all_metrics={"reward": 0.5},
        mission_results={},
        success=True,
    )

    assert trial.trial_id == 1
    assert trial.hyperparameters["lr"] == 0.001
    assert trial.primary_metric == 0.5
    assert trial.success is True
    assert trial.timestamp is not None


def test_sweep_result_creation():
    """Test SweepResult initialization."""
    result = SweepResult(
        sweep_name="test_sweep",
        objective_metric="reward",
        optimize_direction="maximize",
    )

    assert result.sweep_name == "test_sweep"
    assert result.objective_metric == "reward"
    assert len(result.trials) == 0


def test_sweep_result_add_trial():
    """Test adding trials to sweep result."""
    result = SweepResult("test", "reward", "maximize")

    trial1 = SweepTrialResult(
        trial_id=1,
        hyperparameters={"lr": 0.001},
        primary_metric=0.5,
        all_metrics={},
        mission_results={},
        success=True,
    )

    trial2 = SweepTrialResult(
        trial_id=2,
        hyperparameters={"lr": 0.01},
        primary_metric=0.7,
        all_metrics={},
        mission_results={},
        success=True,
    )

    result.add_trial(trial1)
    result.add_trial(trial2)

    assert len(result.trials) == 2


def test_sweep_result_get_best_trial_maximize():
    """Test getting best trial with maximize objective."""
    result = SweepResult("test", "reward", "maximize")

    for i in range(1, 4):
        trial = SweepTrialResult(
            trial_id=i,
            hyperparameters={"x": i},
            primary_metric=float(i),
            all_metrics={},
            mission_results={},
            success=True,
        )
        result.add_trial(trial)

    best = result.get_best_trial()
    assert best.trial_id == 3
    assert best.primary_metric == 3.0


def test_sweep_result_get_best_trial_minimize():
    """Test getting best trial with minimize objective."""
    result = SweepResult("test", "loss", "minimize")

    for i in range(1, 4):
        trial = SweepTrialResult(
            trial_id=i,
            hyperparameters={"x": i},
            primary_metric=float(i),
            all_metrics={},
            mission_results={},
            success=True,
        )
        result.add_trial(trial)

    best = result.get_best_trial()
    assert best.trial_id == 1
    assert best.primary_metric == 1.0


def test_sweep_result_top_trials():
    """Test getting top N trials."""
    result = SweepResult("test", "reward", "maximize")

    for i in range(1, 6):
        trial = SweepTrialResult(
            trial_id=i,
            hyperparameters={"x": i},
            primary_metric=float(i),
            all_metrics={},
            mission_results={},
            success=True,
        )
        result.add_trial(trial)

    top_3 = result.top_trials(3)
    assert len(top_3) == 3
    assert top_3[0].trial_id == 5  # Best
    assert top_3[1].trial_id == 4
    assert top_3[2].trial_id == 3


def test_sweep_result_save_json(tmp_path):
    """Test saving sweep results to JSON."""
    result = SweepResult("test_sweep", "reward", "maximize")

    for i in range(1, 3):
        trial = SweepTrialResult(
            trial_id=i,
            hyperparameters={"lr": 0.001 * i},
            primary_metric=0.5 + i * 0.1,
            all_metrics={"reward": 0.5 + i * 0.1},
            mission_results={"mission1": 0.5 + i * 0.1},
            success=True,
        )
        result.add_trial(trial)

    result.finalize()

    json_path = tmp_path / "sweep_results.json"
    result.save_json(json_path)

    assert json_path.exists()

    with open(json_path, "r") as f:
        data = json.load(f)

    assert data["sweep_name"] == "test_sweep"
    assert data["num_trials"] == 2
    assert len(data["trials"]) == 2


def test_daf_grid_search():
    """Test grid search configuration generation."""
    search_space = {
        "lr": [0.001, 0.01],
        "batch_size": [32, 64],
    }

    configs = daf_grid_search(search_space)

    assert len(configs) == 4  # 2 x 2 = 4 combinations

    # Check all combinations are present
    lrs = [c["lr"] for c in configs]
    assert 0.001 in lrs
    assert 0.01 in lrs


def test_daf_grid_search_empty():
    """Test grid search with empty space."""
    configs = daf_grid_search({})
    assert len(configs) == 1


def test_daf_random_search():
    """Test random search configuration generation."""
    search_space = {
        "lr": (0.0001, 0.1),
        "hidden_size": (32, 256),
    }

    configs = daf_random_search(search_space, num_samples=10, seed=42)

    assert len(configs) == 10

    # Check values are within bounds
    for config in configs:
        assert 0.0001 <= config["lr"] <= 0.1
        # Hidden size should be integer
        assert isinstance(config["hidden_size"], (int, float))


def test_daf_random_search_reproducibility():
    """Test random search reproducibility with seed."""
    search_space = {"x": (0.0, 1.0)}

    configs1 = daf_random_search(search_space, num_samples=5, seed=42)
    configs2 = daf_random_search(search_space, num_samples=5, seed=42)

    assert configs1 == configs2


def test_daf_sweep_best_config():
    """Test extracting best config from sweep."""
    result = SweepResult("test", "reward", "maximize")

    trial = SweepTrialResult(
        trial_id=1,
        hyperparameters={"lr": 0.001, "batch_size": 32},
        primary_metric=0.5,
        all_metrics={},
        mission_results={},
        success=True,
    )
    result.add_trial(trial)

    best_config = daf_sweep_best_config(result)
    assert best_config == {"lr": 0.001, "batch_size": 32}


def test_daf_sweep_best_config_no_trials():
    """Test best config with no trials."""
    result = SweepResult("test", "reward", "maximize")
    best_config = daf_sweep_best_config(result)
    assert best_config is None


def test_daf_sweep_status(tmp_path):
    """Test sweep status from results file."""
    # Create a mock results JSON
    results_data = {
        "sweep_name": "test_sweep",
        "objective_metric": "reward",
        "optimize_direction": "maximize",
        "num_trials": 10,
        "start_time": "2025-01-01T00:00:00",
        "end_time": "2025-01-01T01:00:00",
        "trials": [
            {
                "trial_id": i,
                "hyperparameters": {"x": i},
                "primary_metric": float(i),
                "all_metrics": {},
                "mission_results": {},
                "success": True,
                "timestamp": "2025-01-01T00:00:00",
            }
            for i in range(1, 6)
        ],
    }

    json_path = tmp_path / "results.json"
    with open(json_path, "w") as f:
        json.dump(results_data, f)

    status = daf_sweep_status(json_path)

    assert status["num_trials"] == 10
    assert status["num_completed"] == 5
    assert status["objective_metric"] == "reward"


def test_daf_sweep_status_file_not_found():
    """Test sweep status with missing file."""
    status = daf_sweep_status("/nonexistent/results.json")
    assert status == {}


class TestSweepIntegration:
    """Integration tests for sweep functionality."""

    def test_sweep_lifecycle(self, tmp_path):
        """Test complete sweep lifecycle."""
        result = SweepResult("test_lifecycle", "reward", "maximize")

        # Add trials
        for i in range(1, 4):
            trial = SweepTrialResult(
                trial_id=i,
                hyperparameters={"lr": 0.001 * i},
                primary_metric=float(i),
                all_metrics={"reward": float(i)},
                mission_results={"m1": float(i)},
                success=True,
            )
            result.add_trial(trial)

        result.finalize()

        # Save results
        json_path = tmp_path / "results.json"
        result.save_json(json_path)

        # Check best configuration
        best = result.get_best_trial()
        assert best.trial_id == 3
        assert best.primary_metric == 3.0

        # Verify file contents
        assert json_path.exists()
        with open(json_path, "r") as f:
            data = json.load(f)
        assert data["num_trials"] == 3

