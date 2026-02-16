"""DAF configuration management using Pydantic models.

Provides flexible configuration loading from YAML/JSON files and environment
variables, following patterns established by cogames.game.load_mission_config.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class DAFConfig(BaseModel):
    """Global DAF configuration settings.

    Controls behavior of distributed training, sweeps, and deployments.
    Supports environment variable overrides via Pydantic's ConfigDict.
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    # Output and checkpointing
    output_dir: Path = Field(default=Path("./daf_output"), description="Root output directory for all DAF results")
    checkpoint_dir: Path = Field(default=Path("./daf_output/checkpoints"), description="Directory for policy checkpoints")
    max_checkpoints_to_keep: int = Field(default=5, ge=1, description="Maximum checkpoints to retain per sweep/training")
    organize_by_operation: bool = Field(default=True, description="Organize outputs in subfolders by operation type")

    # Resource management
    max_parallel_jobs: int = Field(default=4, ge=1, description="Maximum parallel evaluation/training jobs")
    enable_gpu: bool = Field(default=True, description="Enable GPU acceleration if available")
    gpu_memory_fraction: float = Field(default=0.8, ge=0.1, le=1.0, description="Fraction of GPU memory to use")

    # Logging and monitoring
    verbose: bool = Field(default=False, description="Enable verbose logging")
    log_to_file: bool = Field(default=True, description="Log outputs to file")
    monitor_system_stats: bool = Field(default=True, description="Monitor CPU/GPU/memory usage")

    # Experiment tracking
    track_experiments: bool = Field(default=True, description="Enable experiment tracking and results logging")
    experiments_log_path: Path = Field(default=Path("./daf_output/experiments.jsonl"), description="Path to experiments log")

    @field_validator("output_dir", "checkpoint_dir", mode="before")
    @classmethod
    def ensure_path(cls, v: Any) -> Path:
        """Convert string paths to Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v


class DAFSweepConfig(BaseModel):
    """Hyperparameter sweep configuration.

    Defines parameter search spaces, optimization strategy, and evaluation metrics.
    Integrates with cogames.evaluate.evaluate() for assessment.
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    name: str = Field(description="Sweep name for tracking and reporting")
    description: Optional[str] = Field(default=None, description="Sweep description and goals")

    # Mission and policy configuration
    missions: list[str] = Field(description="Mission names or paths to evaluate on")
    policy_class_path: str = Field(description="Policy class path (e.g., cogames.policy.lstm.LSTMPolicy)")

    # Sweep parameters
    search_space: Dict[str, Any] = Field(description="Parameter search space (varies by strategy)")
    strategy: Literal["grid", "random", "bayesian"] = Field(default="grid", description="Search strategy")
    num_trials: int = Field(default=10, ge=1, description="Number of trials for random/Bayesian search")

    # Evaluation settings
    episodes_per_trial: int = Field(default=3, ge=1, description="Episodes to run per configuration")
    max_steps_per_episode: int = Field(default=1000, ge=1, description="Max steps per episode")
    seed: int = Field(default=42, ge=0, description="Random seed for reproducibility")

    # Optimization objective
    objective_metric: str = Field(default="avg_reward_per_agent", description="Metric to optimize")
    optimize_direction: Literal["maximize", "minimize"] = Field(default="maximize", description="Optimization direction")

    # Output and checkpointing
    checkpoint_best_n: int = Field(default=3, ge=1, description="Save N best configurations")

    @classmethod
    def from_yaml(cls, path: Path | str) -> DAFSweepConfig:
        """Load sweep configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            DAFSweepConfig instance

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If YAML is invalid or missing required fields
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Sweep config not found: {path}")

        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            if data is None:
                data = {}
            return cls(**data)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML: {e}") from e

    @classmethod
    def from_json(cls, path: Path | str) -> DAFSweepConfig:
        """Load sweep configuration from JSON file.

        Args:
            path: Path to JSON configuration file

        Returns:
            DAFSweepConfig instance

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If JSON is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Sweep config not found: {path}")

        try:
            with open(path, "r") as f:
                data = json.load(f)
            return cls(**data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}") from e

    def to_yaml(self, path: Path | str) -> None:
        """Save configuration to YAML file.

        Args:
            path: Output path for YAML file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.model_dump(mode="python"), f, default_flow_style=False, sort_keys=False)


class DAFDeploymentConfig(BaseModel):
    """Policy deployment configuration.

    Specifies deployment endpoints, validation requirements, and monitoring settings.
    Integrates with cogames.auth for secure authentication and cogames.cli.submit
    for tournament submissions.
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    policy_name: str = Field(description="Human-readable policy name for deployment")
    policy_class_path: str = Field(description="Policy class path for loading")
    weights_path: Optional[str] = Field(default=None, description="Path to policy weights/checkpoint")

    # Deployment target
    deployment_endpoint: str = Field(description="Target deployment endpoint URL")
    authentication_server: str = Field(description="Authentication server URL")

    # Tournament submission settings
    season: Optional[str] = Field(default=None, description="Tournament season name")
    validation_mode: Literal["quick", "thorough", "none"] = Field(
        default="quick", description="Validation mode before deployment"
    )
    setup_script: Optional[str] = Field(default=None, description="Setup script for environment preparation")

    # Validation settings
    validation_missions: list[str] = Field(default_factory=list, description="Missions for validation")
    validation_success_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Min success rate to deploy")

    # Versioning and rollback
    version: str = Field(default="1.0.0", description="Semantic version for this deployment")
    enable_rollback: bool = Field(default=True, description="Allow rollback to previous version")
    keep_previous_versions: int = Field(default=3, ge=1, description="Number of previous versions to keep")

    @classmethod
    def from_yaml(cls, path: Path | str) -> DAFDeploymentConfig:
        """Load deployment configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            DAFDeploymentConfig instance

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If YAML is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Deployment config not found: {path}")

        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            if data is None:
                data = {}
            return cls(**data)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML: {e}") from e


class DAFTournamentConfig(BaseModel):
    """Tournament submission and competition settings.

    Configures tournament server, season, and submission parameters
    for interacting with the CoGames leaderboard.
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    server_url: str = Field(default="https://cogames.ai", description="Tournament server URL")
    season: str = Field(description="Tournament season name")
    submission_name: Optional[str] = Field(default=None, description="Custom submission name")

    # Submission settings
    auto_validate: bool = Field(default=True, description="Validate policy before submission")
    validation_episodes: int = Field(default=5, ge=1, description="Episodes for pre-submission validation")

    @classmethod
    def from_yaml(cls, path: Path | str) -> "DAFTournamentConfig":
        """Load tournament configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tournament config not found: {path}")
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
            return cls(**data)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML: {e}") from e


class DAFVariantConfig(BaseModel):
    """Mission variant configuration for sweeps and evaluation.

    Specifies which variants to apply and difficulty settings for
    CogsGuard missions.
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    variant_names: list[str] = Field(default_factory=list, description="Variant names to apply")
    difficulty: Optional[Literal["easy", "medium", "hard"]] = Field(
        default=None, description="Difficulty level for clips events"
    )
    num_cogs: Optional[int] = Field(default=None, ge=1, le=20, description="Override number of cogs")


class DAFComparisonConfig(BaseModel):
    """Policy comparison configuration.

    Specifies policies to compare, missions to evaluate on, and analysis settings.
    Generates statistical comparisons and visualizations.
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    name: str = Field(description="Comparison name for tracking")
    policies: list[str] = Field(description="Policy class paths or shortcuts to compare")
    policy_weights: list[Optional[str]] = Field(default_factory=list, description="Optional weights for each policy")

    # Evaluation configuration
    missions: list[str] = Field(description="Missions to evaluate on")
    episodes_per_mission: int = Field(default=5, ge=1, description="Episodes per mission per policy")
    max_steps_per_episode: int = Field(default=1000, ge=1, description="Max steps per episode")
    seed: int = Field(default=42, ge=0, description="Random seed")

    # Statistical analysis
    significance_level: float = Field(default=0.05, ge=0.01, le=0.1, description="P-value threshold for significance")
    bootstrap_samples: int = Field(default=1000, ge=100, description="Bootstrap samples for confidence intervals")

    # Output settings
    generate_html_report: bool = Field(default=True, description="Generate interactive HTML report")
    save_raw_results: bool = Field(default=True, description="Save raw results as JSON")

    @classmethod
    def from_yaml(cls, path: Path | str) -> DAFComparisonConfig:
        """Load comparison configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            DAFComparisonConfig instance

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If YAML is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Comparison config not found: {path}")

        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            if data is None:
                data = {}
            return cls(**data)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML: {e}") from e


class DAFPipelineConfig(BaseModel):
    """Workflow orchestration configuration.

    Chains multiple DAF operations (training → evaluation → deployment) into
    automated pipelines with dependency tracking and error handling.
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    name: str = Field(description="Pipeline name")
    description: Optional[str] = Field(default=None, description="Pipeline description")

    # Pipeline stages
    stages: list[str] = Field(description="Ordered list of stages: training, sweep, evaluation, comparison, deployment")

    # Stage configurations (optional, can be provided inline)
    training_config: Optional[Dict[str, Any]] = Field(default=None, description="Training configuration if stage includes training")
    sweep_config: Optional[Dict[str, Any]] = Field(default=None, description="Sweep configuration if stage includes sweep")
    evaluation_config: Optional[Dict[str, Any]] = Field(default=None, description="Evaluation configuration")
    comparison_config: Optional[Dict[str, Any]] = Field(default=None, description="Comparison configuration if stage includes comparison")
    deployment_config: Optional[Dict[str, Any]] = Field(default=None, description="Deployment configuration if stage includes deployment")

    # Execution settings
    stop_on_failure: bool = Field(default=True, description="Stop pipeline if any stage fails")
    parallel_stages: bool = Field(default=False, description="Run independent stages in parallel")
    notify_on_completion: bool = Field(default=False, description="Send notification when pipeline completes")

    def to_yaml(self, path: Path | str) -> None:
        """Save pipeline config to YAML file.

        Args:
            path: Path to save YAML file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Path | str) -> DAFPipelineConfig:
        """Load pipeline configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            DAFPipelineConfig instance

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If YAML is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Pipeline config not found: {path}")

        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            if data is None:
                data = {}
            return cls(**data)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML: {e}") from e


def daf_load_config(config_path: Path | str) -> DAFConfig:
    """Load DAF global configuration from file.

    Args:
        config_path: Path to YAML or JSON configuration file

    Returns:
        DAFConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config format is invalid
    """
    config_path = Path(config_path)

    if config_path.suffix == ".yaml" or config_path.suffix == ".yml":
        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}
        return DAFConfig(**data)
    elif config_path.suffix == ".json":
        with open(config_path, "r") as f:
            data = json.load(f)
        return DAFConfig(**data)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")

