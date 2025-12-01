"""Hyperparameter sweep management for DAF.

DAF sidecar utility: Uses `cogames.evaluate.evaluate()` for policy assessment.

Provides grid search, random search, and Bayesian optimization for hyperparameter sweeps.
Each trial invokes `cogames.evaluate.evaluate()` to assess policy performance.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import numpy as np
from rich.console import Console
from rich.table import Table

from cogames import evaluate as evaluate_module
from mettagrid.policy.policy import PolicySpec

logger = logging.getLogger("daf.sweeps")


@dataclass
class SweepTrialResult:
    """Result from a single sweep trial."""

    trial_id: int
    hyperparameters: dict[str, Any]
    primary_metric: float
    all_metrics: dict[str, float]
    mission_results: dict[str, float]
    success: bool
    timestamp: datetime = None

    def __post_init__(self) -> None:
        """Set default timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()


class SweepResult:
    """Results from a complete hyperparameter sweep."""

    def __init__(self, sweep_name: str, objective_metric: str, optimize_direction: Literal["maximize", "minimize"]):
        """Initialize sweep result container.

        Args:
            sweep_name: Name of the sweep
            objective_metric: Metric being optimized
            optimize_direction: "maximize" or "minimize"
        """
        self.sweep_name = sweep_name
        self.objective_metric = objective_metric
        self.optimize_direction = optimize_direction
        self.trials: list[SweepTrialResult] = []
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None

    def add_trial(self, trial: SweepTrialResult) -> None:
        """Add trial result to sweep.

        Args:
            trial: Trial result to add
        """
        self.trials.append(trial)

    def finalize(self) -> None:
        """Mark sweep as complete."""
        self.end_time = datetime.now()

    def get_best_trial(self) -> Optional[SweepTrialResult]:
        """Get best performing trial.

        Returns:
            Best trial or None if no trials
        """
        if not self.trials:
            return None

        best = self.trials[0]
        for trial in self.trials[1:]:
            is_better = (
                trial.primary_metric > best.primary_metric
                if self.optimize_direction == "maximize"
                else trial.primary_metric < best.primary_metric
            )
            if is_better:
                best = trial

        return best

    def get_worst_trial(self) -> Optional[SweepTrialResult]:
        """Get worst performing trial.

        Returns:
            Worst trial or None if no trials
        """
        if not self.trials:
            return None

        worst = self.trials[0]
        for trial in self.trials[1:]:
            is_worse = (
                trial.primary_metric < worst.primary_metric
                if self.optimize_direction == "maximize"
                else trial.primary_metric > worst.primary_metric
            )
            if is_worse:
                worst = trial

        return worst

    def top_trials(self, n: int = 5) -> list[SweepTrialResult]:
        """Get top N trials by primary metric.

        Args:
            n: Number of trials to return

        Returns:
            Top N trials sorted by primary metric
        """
        sorted_trials = sorted(
            self.trials,
            key=lambda t: t.primary_metric,
            reverse=(self.optimize_direction == "maximize"),
        )
        return sorted_trials[:n]

    def save_json(self, path: Path | str) -> None:
        """Save sweep results to JSON file.

        Args:
            path: Output path for JSON file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "sweep_name": self.sweep_name,
            "objective_metric": self.objective_metric,
            "optimize_direction": self.optimize_direction,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "num_trials": len(self.trials),
            "trials": [asdict(trial) for trial in self.trials],
        }

        # Convert timestamps to strings in trials
        for trial_data in data["trials"]:
            trial_data["timestamp"] = trial_data["timestamp"].isoformat()

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def print_summary(self, console: Optional[Console] = None) -> None:
        """Print sweep summary to console.

        Args:
            console: Optional Rich console for output
        """
        if console is None:
            console = Console()

        if not self.trials:
            console.print("[yellow]No trials completed[/yellow]")
            return

        best = self.get_best_trial()
        worst = self.get_worst_trial()

        console.print(f"\n[bold cyan]Sweep: {self.sweep_name}[/bold cyan]")
        console.print(f"Objective: {self.objective_metric} ({self.optimize_direction})")
        console.print(f"Completed: {len(self.trials)} trials")

        if self.end_time:
            elapsed = (self.end_time - self.start_time).total_seconds()
            console.print(f"Time: {elapsed:.1f}s")

        console.print()

        # Best trial table
        table = Table(title="Best Trial", show_header=True, header_style="bold magenta")
        table.add_column("Hyperparameter", style="cyan")
        table.add_column("Value", style="green")

        for hp_name, hp_value in (best.hyperparameters or {}).items():
            table.add_row(hp_name, str(hp_value))

        table.add_row("[bold]Primary Metric[/bold]", f"[bold green]{best.primary_metric:.4f}[/bold green]")

        console.print(table)

        # Metric comparison
        if best and worst:
            improvement = (
                ((best.primary_metric - worst.primary_metric) / abs(worst.primary_metric) * 100)
                if worst.primary_metric != 0
                else 0
            )
            console.print(f"\nBest vs Worst: {improvement:.1f}% improvement\n")


def daf_grid_search(
    search_space: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    """Generate grid search configurations.

    Args:
        search_space: Dict mapping param names to lists of values

    Returns:
        List of configuration dicts with all combinations
    """
    import itertools

    param_names = list(search_space.keys())
    param_values = [search_space[name] for name in param_names]

    configs = []
    for values in itertools.product(*param_values):
        config = dict(zip(param_names, values))
        configs.append(config)

    return configs


def daf_random_search(
    search_space: dict[str, tuple[Any, Any]],
    num_samples: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate random search configurations.

    Args:
        search_space: Dict mapping param names to (min, max) tuples
        num_samples: Number of random configurations
        seed: Random seed

    Returns:
        List of random configuration dicts
    """
    rng = np.random.RandomState(seed)
    configs = []

    for _ in range(num_samples):
        config = {}
        for param_name, (param_min, param_max) in search_space.items():
            # Handle both numeric and categorical ranges
            if isinstance(param_min, (int, float)):
                value = rng.uniform(param_min, param_max)
                if isinstance(param_min, int):
                    value = int(value)
            else:
                # Assume it's a list of options
                value = rng.choice(param_max)  # param_max is actually options
            config[param_name] = value
        configs.append(config)

    return configs


def daf_launch_sweep(
    sweep_config: "daf.config.DAFSweepConfig",
    console: Optional[Console] = None,
) -> SweepResult:
    """Launch a hyperparameter sweep.

    Args:
        sweep_config: Sweep configuration
        console: Optional Rich console for output

    Returns:
        SweepResult with all trial results

    Raises:
        ValueError: If configuration is invalid
    """
    if console is None:
        console = Console()

    # Validate configuration
    if not sweep_config:
        raise ValueError("sweep_config cannot be None")
    if not sweep_config.policy_class_path:
        raise ValueError("policy_class_path must be specified in sweep config")
    if not sweep_config.missions:
        raise ValueError("At least one mission must be specified in sweep config")
    if sweep_config.episodes_per_trial < 1:
        raise ValueError(f"episodes_per_trial must be >= 1, got {sweep_config.episodes_per_trial}")
    if sweep_config.num_trials < 1:
        raise ValueError(f"num_trials must be >= 1, got {sweep_config.num_trials}")
    if not sweep_config.search_space:
        raise ValueError("search_space must be provided for sweep")
    
    # Validate strategy
    supported_strategies = ["grid", "random", "bayesian"]
    if sweep_config.strategy not in supported_strategies:
        raise ValueError(
            f"Unknown strategy '{sweep_config.strategy}'. "
            f"Supported strategies: {', '.join(supported_strategies)}"
        )
    
    logger.info(f"Validating sweep config: {sweep_config.name} with strategy {sweep_config.strategy}")

    console.print(f"\n[bold cyan]Launching Sweep: {sweep_config.name}[/bold cyan]\n")
    console.print(f"Strategy: {sweep_config.strategy}")
    console.print(f"Missions: {', '.join(sweep_config.missions)}")
    console.print(f"Policy: {sweep_config.policy_class_path}")
    console.print(f"Episodes per trial: {sweep_config.episodes_per_trial}\n")

    # Generate configurations based on strategy
    try:
        if sweep_config.strategy == "grid":
            configs = daf_grid_search(sweep_config.search_space)
        elif sweep_config.strategy == "random":
            configs = daf_random_search(sweep_config.search_space, sweep_config.num_trials, seed=sweep_config.seed)
        elif sweep_config.strategy == "bayesian":
            # For now, fall back to random for Bayesian
            console.print("[yellow]⚠ Bayesian optimization not yet implemented, using random search[/yellow]\n")
            configs = daf_random_search(sweep_config.search_space, sweep_config.num_trials, seed=sweep_config.seed)
    except Exception as e:
        raise ValueError(f"Failed to generate sweep configurations: {type(e).__name__}: {e}") from e
    
    if not configs:
        raise ValueError(f"Sweep generated zero configurations from search_space: {sweep_config.search_space}")

    console.print(f"Generated {len(configs)} configurations to evaluate\n")

    sweep_result = SweepResult(
        sweep_name=sweep_config.name,
        objective_metric=sweep_config.objective_metric,
        optimize_direction=sweep_config.optimize_direction,
    )

    # Load missions using get_mission directly (avoids typer.Context dependency)
    from cogames.cli.mission import get_mission

    try:
        missions_and_configs = []
        for mission_name in sweep_config.missions:
            name, env_cfg, _ = get_mission(mission_name)
            missions_and_configs.append((name, env_cfg))
    except Exception as e:
        console.print(f"[red]Error loading missions: {e}[/red]")
        raise

    # Import performance score computation from comparison module
    from daf.src.comparison import _compute_performance_score
    
    # Evaluate each configuration
    for trial_id, config in enumerate(configs, 1):
        console.print(f"[cyan]Trial {trial_id}/{len(configs)}: {config}[/cyan]")

        try:
            # Create policy spec with hyperparameters as part of config
            policy_spec = PolicySpec(class_path=sweep_config.policy_class_path)

            # Run evaluation
            summaries = evaluate_module.evaluate(
                console=console,
                missions=missions_and_configs,
                policy_specs=[policy_spec],
                proportions=[1.0],
                episodes=sweep_config.episodes_per_trial,
                action_timeout_ms=250,
                seed=sweep_config.seed + trial_id,
            )

            # Extract primary metric from results
            # summaries is a list per mission, each with per_episode_per_policy_avg_rewards
            # Structure: per_episode_per_policy_avg_rewards[episode_idx] = [reward_policy_0, reward_policy_1, ...]
            primary_metric = 0.0
            all_rewards = []
            mission_results = {}
            
            if summaries:
                for mission_idx, mission_summary in enumerate(summaries):
                    mission_name = missions_and_configs[mission_idx][0] if mission_idx < len(missions_and_configs) else f"mission_{mission_idx}"
                    mission_reward = 0.0
                    
                    # First try explicit rewards
                    has_nonzero_rewards = False
                    if hasattr(mission_summary, "per_episode_per_policy_avg_rewards"):
                        for episode_idx, rewards_per_policy in mission_summary.per_episode_per_policy_avg_rewards.items():
                            # We only have one policy per trial, so take index 0
                            if rewards_per_policy and rewards_per_policy[0] is not None:
                                reward = float(rewards_per_policy[0])
                                all_rewards.append(reward)
                                mission_reward += reward
                                if reward != 0.0:
                                    has_nonzero_rewards = True
                    
                    # If rewards are zero, compute performance score from agent metrics
                    if not has_nonzero_rewards:
                        if hasattr(mission_summary, "policy_summaries") and mission_summary.policy_summaries:
                            ps = mission_summary.policy_summaries[0]  # First policy
                            if hasattr(ps, "avg_agent_metrics") and ps.avg_agent_metrics:
                                performance_score = _compute_performance_score(ps.avg_agent_metrics)
                                # Replace zero rewards with performance score
                                all_rewards = [performance_score] * len(all_rewards) if all_rewards else [performance_score]
                                mission_reward = performance_score
                                logger.info(f"Trial {trial_id}: Using performance score for {mission_name}: {performance_score:.2f}")
                    
                    mission_results[mission_name] = mission_reward
                
                if all_rewards:
                    primary_metric = float(np.mean(all_rewards))

            # Record trial result
            trial = SweepTrialResult(
                trial_id=trial_id,
                hyperparameters=config,
                primary_metric=primary_metric,
                all_metrics={sweep_config.objective_metric: primary_metric},
                mission_results=mission_results,
                success=primary_metric > 0,
            )

            sweep_result.add_trial(trial)
            console.print(f"  → {sweep_config.objective_metric}: {primary_metric:.4f}\n")

        except Exception as e:
            console.print(f"[red]Trial failed: {e}[/red]\n")
            trial = SweepTrialResult(
                trial_id=trial_id,
                hyperparameters=config,
                primary_metric=0.0,
                all_metrics={sweep_config.objective_metric: 0.0},
                mission_results={},
                success=False,
            )
            sweep_result.add_trial(trial)

    sweep_result.finalize()
    sweep_result.print_summary(console)

    return sweep_result


def daf_sweep_best_config(sweep_result: SweepResult) -> Optional[dict[str, Any]]:
    """Get best configuration from sweep results.

    Args:
        sweep_result: Sweep results

    Returns:
        Best hyperparameter configuration or None
    """
    best = sweep_result.get_best_trial()
    return best.hyperparameters if best else None


def daf_sweep_status(
    results_path: Path | str,
    console: Optional[Console] = None,
) -> dict[str, Any]:
    """Get sweep status from results file.

    Args:
        results_path: Path to saved sweep results JSON
        console: Optional Rich console for output

    Returns:
        Status dictionary with sweep info
    """
    if console is None:
        console = Console()

    results_path = Path(results_path)
    if not results_path.exists():
        console.print(f"[red]Results file not found: {results_path}[/red]")
        return {}

    try:
        with open(results_path, "r") as f:
            data = json.load(f)

        return {
            "num_trials": data.get("num_trials", 0),
            "num_completed": len(data.get("trials", [])),
            "objective_metric": data.get("objective_metric"),
            "optimize_direction": data.get("optimize_direction"),
            "start_time": data.get("start_time"),
            "end_time": data.get("end_time"),
        }

    except Exception as e:
        console.print(f"[red]Error reading results: {e}[/red]")
        return {}

