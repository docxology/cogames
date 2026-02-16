"""Tests for DAF configuration management."""

import json
from pathlib import Path

import pytest
import yaml

from daf.src.core.config import (
    DAFConfig,
    DAFComparisonConfig,
    DAFDeploymentConfig,
    DAFPipelineConfig,
    DAFSweepConfig,
    daf_load_config,
)


def test_daf_config_defaults():
    """Test DAF config with default values."""
    config = DAFConfig()
    assert config.output_dir == Path("./daf_output")
    assert config.max_parallel_jobs == 4
    assert config.enable_gpu is True


def test_daf_config_custom_values():
    """Test DAF config with custom values."""
    config = DAFConfig(
        output_dir="./custom_output",
        max_parallel_jobs=8,
        enable_gpu=False,
    )
    assert config.output_dir == Path("./custom_output")
    assert config.max_parallel_jobs == 8
    assert config.enable_gpu is False


def test_daf_sweep_config_from_dict():
    """Test sweep config creation from dict."""
    data = {
        "name": "test_sweep",
        "missions": ["mission1"],
        "policy_class_path": "cogames.policy.lstm.LSTMPolicy",
        "search_space": {"lr": [0.001, 0.01]},
        "strategy": "grid",
    }
    config = DAFSweepConfig(**data)
    assert config.name == "test_sweep"
    assert config.strategy == "grid"
    assert config.num_trials == 10


def test_daf_sweep_config_yaml_roundtrip(tmp_path):
    """Test sweep config YAML save/load roundtrip."""
    config = DAFSweepConfig(
        name="test_sweep",
        missions=["mission1"],
        policy_class_path="cogames.policy.lstm.LSTMPolicy",
        search_space={"lr": [0.001, 0.01]},
        strategy="random",
        num_trials=20,
    )

    yaml_path = tmp_path / "sweep.yaml"
    config.to_yaml(yaml_path)

    loaded = DAFSweepConfig.from_yaml(yaml_path)
    assert loaded.name == config.name
    assert loaded.strategy == config.strategy
    assert loaded.num_trials == config.num_trials


def test_daf_sweep_config_from_yaml_not_found():
    """Test sweep config loading from non-existent file."""
    with pytest.raises(FileNotFoundError):
        DAFSweepConfig.from_yaml("/nonexistent/sweep.yaml")


def test_daf_deployment_config_from_dict():
    """Test deployment config creation."""
    data = {
        "policy_name": "my_policy",
        "policy_class_path": "cogames.policy.lstm.LSTMPolicy",
        "deployment_endpoint": "https://example.com/api",
        "authentication_server": "https://auth.example.com",
    }
    config = DAFDeploymentConfig(**data)
    assert config.policy_name == "my_policy"
    assert config.version == "1.0.0"
    assert config.enable_rollback is True


def test_daf_comparison_config_from_dict():
    """Test comparison config creation."""
    data = {
        "name": "policy_comparison",
        "policies": ["lstm", "baseline"],
        "missions": ["mission1", "mission2"],
        "episodes_per_mission": 5,
    }
    config = DAFComparisonConfig(**data)
    assert config.name == "policy_comparison"
    assert len(config.policies) == 2
    assert config.generate_html_report is True


def test_daf_pipeline_config_from_dict():
    """Test pipeline config creation."""
    data = {
        "name": "full_pipeline",
        "stages": ["training", "evaluation", "comparison"],
    }
    config = DAFPipelineConfig(**data)
    assert config.name == "full_pipeline"
    assert len(config.stages) == 3


def test_daf_pipeline_config_yaml_roundtrip(tmp_path):
    """Test pipeline config YAML save/load."""
    config = DAFPipelineConfig(
        name="test_pipeline",
        stages=["training", "sweep", "comparison"],
        stop_on_failure=True,
    )

    yaml_path = tmp_path / "pipeline.yaml"
    config.to_yaml(yaml_path)

    loaded = DAFPipelineConfig.from_yaml(yaml_path)
    assert loaded.name == config.name
    assert loaded.stages == config.stages


def test_daf_load_config_yaml(tmp_path):
    """Test loading DAF config from YAML."""
    data = {
        "output_dir": "./test_output",
        "max_parallel_jobs": 6,
        "verbose": True,
    }

    yaml_path = tmp_path / "daf_config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f)

    config = daf_load_config(yaml_path)
    assert config.output_dir == Path("./test_output")
    assert config.max_parallel_jobs == 6
    assert config.verbose is True


def test_daf_load_config_json(tmp_path):
    """Test loading DAF config from JSON."""
    data = {
        "output_dir": "./test_output",
        "max_parallel_jobs": 8,
    }

    json_path = tmp_path / "daf_config.json"
    with open(json_path, "w") as f:
        json.dump(data, f)

    config = daf_load_config(json_path)
    assert config.output_dir == Path("./test_output")
    assert config.max_parallel_jobs == 8


def test_daf_load_config_unsupported_format(tmp_path):
    """Test loading config with unsupported format."""
    txt_path = tmp_path / "config.txt"
    txt_path.write_text("invalid")

    with pytest.raises(ValueError, match="Unsupported config format"):
        daf_load_config(txt_path)


def test_gpu_memory_fraction_validation():
    """Test GPU memory fraction validation."""
    # Valid values
    config = DAFConfig(gpu_memory_fraction=0.5)
    assert config.gpu_memory_fraction == 0.5

    # Invalid values should fail validation
    with pytest.raises(ValueError):
        DAFConfig(gpu_memory_fraction=1.5)

    with pytest.raises(ValueError):
        DAFConfig(gpu_memory_fraction=-0.1)


def test_max_checkpoints_validation():
    """Test max checkpoints to keep validation."""
    config = DAFConfig(max_checkpoints_to_keep=10)
    assert config.max_checkpoints_to_keep == 10

    with pytest.raises(ValueError):
        DAFConfig(max_checkpoints_to_keep=0)


def test_sweep_optimize_direction():
    """Test sweep config optimization direction."""
    config = DAFSweepConfig(
        name="test",
        missions=["m1"],
        policy_class_path="policy",
        search_space={"x": [1, 2]},
        optimize_direction="maximize",
    )
    assert config.optimize_direction == "maximize"

    config2 = DAFSweepConfig(
        name="test",
        missions=["m1"],
        policy_class_path="policy",
        search_space={"x": [1, 2]},
        optimize_direction="minimize",
    )
    assert config2.optimize_direction == "minimize"

