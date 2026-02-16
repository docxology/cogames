"""Policy comparison and analytics for DAF.

DAF sidecar utility: Uses `cogames.evaluate.evaluate()` for policy evaluation,
then adds statistical analysis and reporting.

Provides head-to-head policy evaluation with statistical significance testing,
ablation studies, and benchmark suite execution.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from rich.console import Console
from rich.table import Table

from cogames import evaluate as evaluate_module
from mettagrid.policy.policy import PolicySpec

logger = logging.getLogger("daf.comparison")


@dataclass
class PolicyComparisonResult:
    """Results from comparing two policies."""

    policy_a_name: str
    policy_b_name: str
    missions: list[str]
    episodes_per_mission: int

    # Per-mission results
    mission_rewards_a: dict[str, list[float]]  # mission -> list of rewards
    mission_rewards_b: dict[str, list[float]]

    # Aggregated results
    avg_reward_a: float
    avg_reward_b: float
    reward_std_a: float
    reward_std_b: float

    # Statistical significance
    p_value: float  # From t-test
    is_significant: bool  # p < 0.05
    effect_size: float  # Cohen's d

    # Timestamp
    timestamp: datetime = None

    def __post_init__(self) -> None:
        """Set default timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def summary_string(self) -> str:
        """Get human-readable summary of comparison.

        Returns:
            Summary string
        """
        winner = "A" if self.avg_reward_a > self.avg_reward_b else "B" if self.avg_reward_b > self.avg_reward_a else "Tie"
        sig = " (significant)" if self.is_significant else ""

        return (
            f"{self.policy_a_name} vs {self.policy_b_name}:\n"
            f"  A: {self.avg_reward_a:.4f} ± {self.reward_std_a:.4f}\n"
            f"  B: {self.avg_reward_b:.4f} ± {self.reward_std_b:.4f}\n"
            f"  Winner: {winner}{sig}"
        )


class ComparisonReport:
    """Complete report from multi-policy comparison."""

    def __init__(self, policies: list[str], missions: list[str], episodes_per_mission: int):
        """Initialize comparison report.

        Args:
            policies: List of policy names/paths
            missions: List of mission names
            episodes_per_mission: Episodes per mission per policy
        """
        self.policies = policies
        self.missions = missions
        self.episodes_per_mission = episodes_per_mission

        # Storage for results
        self.policy_mission_rewards: dict[str, dict[str, list[float]]] = {}
        self.policy_averages: dict[str, float] = {}
        self.policy_std_devs: dict[str, float] = {}
        self.pairwise_comparisons: dict[tuple[str, str], PolicyComparisonResult] = {}
        
        # Detailed agent metrics for richer visualization
        self.policy_detailed_metrics: dict[str, dict[str, dict[str, float]]] = {}

        self.timestamp = datetime.now()

    def add_policy_results(self, policy_name: str, mission_rewards: dict[str, list[float]]) -> None:
        """Add results for a policy across missions.

        Args:
            policy_name: Policy name/identifier
            mission_rewards: Dict mapping mission name to list of rewards
        """
        self.policy_mission_rewards[policy_name] = mission_rewards

        # Calculate aggregated statistics
        all_rewards = []
        for rewards in mission_rewards.values():
            all_rewards.extend(rewards)

        if all_rewards:
            self.policy_averages[policy_name] = float(np.mean(all_rewards))
            self.policy_std_devs[policy_name] = float(np.std(all_rewards))
        else:
            self.policy_averages[policy_name] = 0.0
            self.policy_std_devs[policy_name] = 0.0

    def compute_pairwise_comparisons(self, significance_level: float = 0.05) -> None:
        """Compute pairwise statistical comparisons between all policies.

        Args:
            significance_level: P-value threshold for significance
        """
        import warnings
        from scipy import stats

        policy_names = list(self.policies)

        for i, policy_a in enumerate(policy_names):
            for policy_b in policy_names[i + 1 :]:
                rewards_a = []
                rewards_b = []

                for mission in self.missions:
                    if policy_a in self.policy_mission_rewards and mission in self.policy_mission_rewards[policy_a]:
                        rewards_a.extend(self.policy_mission_rewards[policy_a][mission])
                    if policy_b in self.policy_mission_rewards and mission in self.policy_mission_rewards[policy_b]:
                        rewards_b.extend(self.policy_mission_rewards[policy_b][mission])

                if not rewards_a or not rewards_b:
                    continue

                # Calculate means and stds
                mean_a = float(np.mean(rewards_a))
                mean_b = float(np.mean(rewards_b))
                std_a = float(np.std(rewards_a))
                std_b = float(np.std(rewards_b))

                # T-test with handling for zero-variance data
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error", category=RuntimeWarning)
                        t_stat, p_value = stats.ttest_ind(rewards_a, rewards_b)
                        p_value = float(p_value)
                except (RuntimeWarning, FloatingPointError):
                    # Handle identical data or zero variance - compare means directly
                    if mean_a != mean_b:
                        # Different means with zero variance = definitely significant
                        p_value = 0.0
                    else:
                        # Identical data
                        p_value = 1.0

                # Cohen's d effect size
                pooled_std = np.sqrt((std_a ** 2 + std_b ** 2) / 2)
                effect_size = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0

                comparison = PolicyComparisonResult(
                    policy_a_name=policy_a,
                    policy_b_name=policy_b,
                    missions=self.missions,
                    episodes_per_mission=self.episodes_per_mission,
                    mission_rewards_a=self.policy_mission_rewards.get(policy_a, {}),
                    mission_rewards_b=self.policy_mission_rewards.get(policy_b, {}),
                    avg_reward_a=mean_a,
                    avg_reward_b=mean_b,
                    reward_std_a=std_a,
                    reward_std_b=std_b,
                    p_value=p_value,
                    is_significant=p_value < significance_level,
                    effect_size=float(effect_size),
                )

                self.pairwise_comparisons[(policy_a, policy_b)] = comparison

    @property
    def summary_statistics(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all policies.

        Returns:
            Dict mapping policy name to stats dict
        """
        stats = {}
        for policy in self.policies:
            stats[policy] = {
                "avg_reward": self.policy_averages.get(policy, 0.0),
                "std_dev": self.policy_std_devs.get(policy, 0.0),
            }
        return stats

    def print_summary(self, console: Optional[Console] = None) -> None:
        """Print comparison summary to console.

        Args:
            console: Optional Rich console for output
        """
        if console is None:
            console = Console()

        console.print(f"\n[bold cyan]Policy Comparison Report[/bold cyan]\n")

        # Summary table
        table = Table(title="Policy Performance Summary", show_header=True, header_style="bold magenta")
        table.add_column("Policy", style="cyan")
        table.add_column("Avg Reward", justify="right", style="green")
        table.add_column("Std Dev", justify="right", style="yellow")

        for policy in self.policies:
            avg = self.policy_averages.get(policy, 0.0)
            std = self.policy_std_devs.get(policy, 0.0)
            table.add_row(policy, f"{avg:.4f}", f"{std:.4f}")

        console.print(table)

        # Pairwise comparisons
        if self.pairwise_comparisons:
            console.print(f"\n[bold cyan]Pairwise Comparisons[/bold cyan]\n")

            for (policy_a, policy_b), comparison in self.pairwise_comparisons.items():
                winner = "A" if comparison.avg_reward_a > comparison.avg_reward_b else "B"
                sig_marker = " *" if comparison.is_significant else ""

                console.print(
                    f"{policy_a} vs {policy_b}:\n"
                    f"  A: {comparison.avg_reward_a:.4f} ± {comparison.reward_std_a:.4f}\n"
                    f"  B: {comparison.avg_reward_b:.4f} ± {comparison.reward_std_b:.4f}\n"
                    f"  Winner: {winner}{sig_marker} (p={comparison.p_value:.4f}, d={comparison.effect_size:.2f})\n"
                )

    def save_json(self, path: Path | str) -> None:
        """Save comparison report to JSON file.

        Args:
            path: Output path for JSON file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "timestamp": self.timestamp.isoformat(),
            "policies": self.policies,
            "missions": self.missions,
            "episodes_per_mission": self.episodes_per_mission,
            "summary_statistics": self.summary_statistics,
            "pairwise_comparisons": {
                f"{a}_vs_{b}": asdict(comparison)
                for (a, b), comparison in self.pairwise_comparisons.items()
            },
        }
        
        # Include detailed metrics if available
        if self.policy_detailed_metrics:
            data["detailed_metrics"] = self.policy_detailed_metrics

        # Convert timestamps in comparisons to strings
        for comparison_data in data["pairwise_comparisons"].values():
            comparison_data["timestamp"] = comparison_data["timestamp"].isoformat()

        # Custom serializer for numpy types
        def default_serializer(obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            if hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            return str(obj)
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=default_serializer)


def _compute_performance_score(metrics: dict[str, float]) -> float:
    """Compute a composite performance score from agent metrics.
    
    Uses a weighted combination of meaningful metrics to produce
    a non-zero performance score even when environment rewards are 0.
    
    Args:
        metrics: Agent metrics dictionary (e.g. from avg_agent_metrics)
        
    Returns:
        Composite performance score
    """
    if not metrics:
        return 0.0
    
    score = 0.0
    
    # Resource gathering (most important for game success)
    resource_metrics = [
        "carbon.gained", "silicon.gained", "oxygen.gained", 
        "germanium.gained", "energy.gained"
    ]
    for metric in resource_metrics:
        if metric in metrics:
            score += metrics[metric] * 1.0
    
    # Inventory diversity (indicates good resource management)
    if "inventory.diversity" in metrics:
        score += metrics["inventory.diversity"] * 10.0
    
    # Successful actions (indicates active, productive behavior)
    if "action.move.success" in metrics:
        score += metrics["action.move.success"] * 0.1
    
    # Penalize failures (indicates ineffective behavior)
    if "action.failed" in metrics:
        score -= metrics["action.failed"] * 0.05
    
    # Penalize being stuck (indicates poor navigation)
    if "status.max_steps_without_motion" in metrics:
        score -= metrics["status.max_steps_without_motion"] * 0.1
    
    return score


def daf_compare_policies(
    policies: list[PolicySpec],
    missions: list[tuple[str, "Any"]],  # (mission_name, env_config)
    episodes_per_mission: int = 5,
    action_timeout_ms: int = 250,
    seed: int = 42,
    console: Optional[Console] = None,
    use_performance_score: bool = True,
) -> ComparisonReport:
    """Compare multiple policies on given missions.

    Args:
        policies: List of PolicySpec objects to compare
        missions: List of (mission_name, env_config) tuples
        episodes_per_mission: Episodes to run per mission
        action_timeout_ms: Timeout for action generation
        seed: Random seed
        console: Optional Rich console for output
        use_performance_score: If True, compute composite performance score from
            agent metrics when raw rewards are 0. This provides meaningful
            comparisons even for environments without explicit reward functions.

    Returns:
        ComparisonReport with detailed results

    Raises:
        ValueError: If inputs are invalid
    """
    if console is None:
        console = Console()

    # Validate inputs
    if not policies:
        raise ValueError("No policies provided for comparison. Provide at least one policy.")
    if not missions:
        raise ValueError("No missions provided for comparison. Provide at least one mission.")
    if episodes_per_mission < 1:
        raise ValueError(f"episodes_per_mission must be >= 1, got {episodes_per_mission}")
    if action_timeout_ms < 1:
        raise ValueError(f"action_timeout_ms must be >= 1, got {action_timeout_ms}")
    
    # Log validation
    logger.info(f"Validating comparison inputs: {len(policies)} policies, {len(missions)} missions")
    
    for i, policy in enumerate(policies):
        if not isinstance(policy, PolicySpec):
            raise ValueError(f"Policy {i} is not a PolicySpec instance: {type(policy)}")
        if not policy.class_path:
            raise ValueError(f"Policy {i} has no class_path specified")
    
    for i, mission in enumerate(missions):
        if not isinstance(mission, tuple) or len(mission) != 2:
            raise ValueError(f"Mission {i} is not a (name, config) tuple: {mission}")
        mission_name, env_config = mission
        if not mission_name:
            raise ValueError(f"Mission {i} has no name")

    console.print(f"\n[bold cyan]DAF Policy Comparison[/bold cyan]\n")
    console.print(f"Comparing {len(policies)} policies on {len(missions)} missions")
    console.print(f"Episodes per mission: {episodes_per_mission}\n")

    mission_names = [m[0] for m in missions]
    policy_names = [p.class_path.split(".")[-1] for p in policies]

    report = ComparisonReport(policy_names, mission_names, episodes_per_mission)

    # Evaluate all policies
    summaries = evaluate_module.evaluate(
        console=console,
        missions=missions,
        policy_specs=policies,
        proportions=[1.0] * len(policies),
        episodes=episodes_per_mission,
        action_timeout_ms=action_timeout_ms,
        seed=seed,
    )

    # Extract results - summaries is a list per mission, not per policy
    # Each mission summary has per_episode_per_policy_avg_rewards and policy_summaries
    for policy_idx, policy_name in enumerate(policy_names):
        mission_rewards = {}

        for mission_idx, mission_summary in enumerate(summaries):
            mission_name = mission_names[mission_idx] if mission_idx < len(mission_names) else f"mission_{mission_idx}"
            mission_rewards[mission_name] = []
            
            # First try to get explicit rewards
            has_nonzero_rewards = False
            if hasattr(mission_summary, "per_episode_per_policy_avg_rewards"):
                for episode_idx, rewards_per_policy in mission_summary.per_episode_per_policy_avg_rewards.items():
                    if policy_idx < len(rewards_per_policy):
                        reward = rewards_per_policy[policy_idx]
                        if reward is not None:
                            mission_rewards[mission_name].append(float(reward))
                            if reward != 0.0:
                                has_nonzero_rewards = True
            
            # If rewards are all zero and use_performance_score is enabled,
            # compute a composite score from agent metrics
            if use_performance_score and not has_nonzero_rewards:
                if hasattr(mission_summary, "policy_summaries") and policy_idx < len(mission_summary.policy_summaries):
                    policy_summary = mission_summary.policy_summaries[policy_idx]
                    if hasattr(policy_summary, "avg_agent_metrics") and policy_summary.avg_agent_metrics:
                        # Compute performance score from metrics
                        performance_score = _compute_performance_score(policy_summary.avg_agent_metrics)
                        # Replace zero rewards with computed score
                        # We use the same score for each episode since we only have aggregated metrics
                        if mission_rewards[mission_name]:
                            mission_rewards[mission_name] = [performance_score] * len(mission_rewards[mission_name])
                        else:
                            mission_rewards[mission_name] = [performance_score] * episodes_per_mission
                        
                        logger.info(f"Using performance score for {policy_name} on {mission_name}: {performance_score:.2f}")

        report.add_policy_results(policy_name, mission_rewards)

    # Store detailed metrics for visualization
    report.policy_detailed_metrics = {}
    for mission_idx, mission_summary in enumerate(summaries):
        mission_name = mission_names[mission_idx] if mission_idx < len(mission_names) else f"mission_{mission_idx}"
        if hasattr(mission_summary, "policy_summaries"):
            for policy_idx, policy_name in enumerate(policy_names):
                if policy_idx < len(mission_summary.policy_summaries):
                    ps = mission_summary.policy_summaries[policy_idx]
                    if hasattr(ps, "avg_agent_metrics"):
                        if policy_name not in report.policy_detailed_metrics:
                            report.policy_detailed_metrics[policy_name] = {}
                        report.policy_detailed_metrics[policy_name][mission_name] = ps.avg_agent_metrics

    # Compute pairwise comparisons
    report.compute_pairwise_comparisons()

    # Print summary
    report.print_summary(console)

    return report


def daf_policy_ablation(
    base_policy: str,
    components_to_ablate: dict[str, bool],
    missions: list[tuple[str, "Any"]],
    episodes_per_mission: int = 5,
    console: Optional[Console] = None,
) -> ComparisonReport:
    """Run ablation study on policy components.

    Evaluates policy with and without each component to measure impact.

    Args:
        base_policy: Base policy class path
        components_to_ablate: Dict mapping component name to whether to ablate
        missions: List of missions to evaluate on
        episodes_per_mission: Episodes per mission
        console: Optional Rich console for output

    Returns:
        ComparisonReport comparing full vs ablated versions

    Raises:
        NotImplementedError: If policy doesn't support ablation
    """
    if console is None:
        console = Console()

    console.print("[yellow]⚠ Ablation studies require custom policy support[/yellow]")
    console.print("[yellow]Policy must implement get_ablated_version() method[/yellow]\n")

    raise NotImplementedError("Policy ablation requires custom implementation per policy")


def daf_benchmark_suite(
    policy_specs: list[PolicySpec],
    benchmark_name: str = "standard",
    console: Optional[Console] = None,
) -> ComparisonReport:
    """Run standardized benchmark suite on policies.

    Uses predefined mission sets from cogames.cogs_vs_clips.evals.

    Args:
        policy_specs: Policies to benchmark
        benchmark_name: Which benchmark suite to use
        console: Optional Rich console for output

    Returns:
        ComparisonReport with benchmark results
        
    Raises:
        ValueError: If benchmark name is invalid or no policies provided
    """
    if console is None:
        console = Console()

    # Validate inputs
    if not policy_specs:
        raise ValueError("No policies provided for benchmark. Provide at least one policy.")
    if not benchmark_name:
        raise ValueError("Benchmark name must be specified")

    console.print(f"\n[bold cyan]DAF Benchmark Suite: {benchmark_name}[/bold cyan]\n")

    # Load appropriate benchmark mission set
    benchmark_missions = None
    try:
        if benchmark_name == "standard":
            from cogames.cogs_vs_clips.evals.eval_missions import EVAL_MISSIONS
            benchmark_missions = EVAL_MISSIONS
        elif benchmark_name == "integrated":
            from cogames.cogs_vs_clips.evals.integrated_evals import EVAL_MISSIONS
            benchmark_missions = EVAL_MISSIONS
        elif benchmark_name == "spanning":
            from cogames.cogs_vs_clips.evals.spanning_evals import EVAL_MISSIONS
            benchmark_missions = EVAL_MISSIONS
        elif benchmark_name == "diagnostic":
            from cogames.cogs_vs_clips.evals.diagnostic_evals import DIAGNOSTIC_EVALS
            benchmark_missions = [m() for m in DIAGNOSTIC_EVALS]
        else:
            available = ["standard", "integrated", "spanning", "diagnostic"]
            raise ValueError(f"Unknown benchmark '{benchmark_name}'. Available: {', '.join(available)}")
    except ImportError as e:
        raise ValueError(f"Could not load benchmark '{benchmark_name}': {e}") from e
    except Exception as e:
        raise ValueError(f"Error loading benchmark '{benchmark_name}': {type(e).__name__}: {e}") from e
    
    if not benchmark_missions:
        raise ValueError(f"Benchmark '{benchmark_name}' has no missions")

    # Convert missions to (name, config) tuples
    missions = []
    for mission in benchmark_missions:
        if hasattr(mission, "make_env"):
            env_cfg = mission.make_env()
            missions.append((mission.name, env_cfg))

    console.print(f"Benchmark missions: {len(missions)}")

    # Compare policies
    return daf_compare_policies(
        policies=policy_specs,
        missions=missions,
        episodes_per_mission=3,
        console=console,
    )

