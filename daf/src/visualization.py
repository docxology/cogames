"""Visualization utilities for DAF analytics.

DAF sidecar utility: Extends visualization patterns from `scripts/run_evaluation.py`.

Provides plotting and visualization for training curves, policy comparisons,
and benchmark results. Uses matplotlib patterns established in CoGames evaluation scripts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from rich.console import Console

if TYPE_CHECKING:
    from daf.comparison import ComparisonReport
    from daf.sweeps import SweepResult

logger = logging.getLogger("daf.visualization")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def daf_plot_training_curves(
    checkpoint_dir: Path | str,
    output_dir: Path | str = "training_plots",
    smoothing: int = 10,
    console: Optional[Console] = None,
) -> None:
    """Visualize training progress from checkpoint directory.

    Args:
        checkpoint_dir: Directory containing training checkpoints
        output_dir: Output directory for plots
        smoothing: Window size for moving average smoothing
        console: Optional Rich console for output

    Raises:
        ImportError: If matplotlib not available
    """
    if console is None:
        console = Console()

    if not HAS_MATPLOTLIB:
        console.print("[yellow]⚠ matplotlib not available - skipping training curve plots[/yellow]")
        return

    console.print("[yellow]⚠ Training curve visualization requires checkpoint logs[/yellow]")
    console.print("[yellow]Feature pending implementation in next iteration[/yellow]\n")


def daf_plot_policy_comparison(
    comparison_report: "ComparisonReport",  # type: ignore
    output_dir: Path | str = "comparison_plots",
    console: Optional[Console] = None,
) -> None:
    """Generate comparison plots from ComparisonReport.

    Args:
        comparison_report: Report from daf_compare_policies
        output_dir: Output directory for plots
        console: Optional Rich console for output

    Raises:
        ImportError: If matplotlib not available
    """
    if console is None:
        console = Console()

    if not HAS_MATPLOTLIB:
        console.print("[yellow]⚠ matplotlib not available - skipping comparison plots[/yellow]")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Average rewards by policy
    fig, ax = plt.subplots(figsize=(10, 6))

    policies = comparison_report.policies
    avg_rewards = [comparison_report.policy_averages.get(p, 0.0) for p in policies]
    std_devs = [comparison_report.policy_std_devs.get(p, 0.0) for p in policies]

    x = np.arange(len(policies))
    colors = plt.get_cmap("Set2")(range(len(policies)))

    ax.bar(x, avg_rewards, yerr=std_devs, color=colors, alpha=0.8, capsize=5, edgecolor="black")

    ax.set_ylabel("Average Reward", fontsize=12, fontweight="bold")
    ax.set_xlabel("Policy", fontsize=12, fontweight="bold")
    ax.set_title("Policy Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "policy_rewards_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    console.print(f"[green]✓ Saved: policy_rewards_comparison.png[/green]")

    # Plot 2: Per-mission performance
    if comparison_report.policy_mission_rewards:
        fig, ax = plt.subplots(figsize=(12, 6))

        missions = comparison_report.missions
        num_policies = len(policies)
        x = np.arange(len(missions))
        width = 0.8 / num_policies

        for i, policy in enumerate(policies):
            rewards = []
            for mission in missions:
                mission_rewards = comparison_report.policy_mission_rewards.get(policy, {}).get(mission, [])
                avg = np.mean(mission_rewards) if mission_rewards else 0.0
                rewards.append(avg)

            offset = width * (i - num_policies / 2 + 0.5)
            ax.bar(x + offset, rewards, width, label=policy, alpha=0.8, edgecolor="black")

        ax.set_ylabel("Average Reward", fontsize=12, fontweight="bold")
        ax.set_xlabel("Mission", fontsize=12, fontweight="bold")
        ax.set_title("Performance by Mission", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(missions, rotation=45, ha="right")
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "performance_by_mission.png", dpi=150, bbox_inches="tight")
        plt.close()

        console.print(f"[green]✓ Saved: performance_by_mission.png[/green]")

    console.print(f"\n[green]Plots saved to: {output_dir}[/green]\n")


def daf_plot_sweep_results(
    sweep_result: "SweepResult",  # type: ignore
    output_dir: Path | str = "sweep_plots",
    console: Optional[Console] = None,
) -> None:
    """Generate plots from sweep results.

    Args:
        sweep_result: Result from daf_launch_sweep
        output_dir: Output directory for plots
        console: Optional Rich console for output

    Raises:
        ImportError: If matplotlib not available
    """
    if console is None:
        console = Console()

    if not HAS_MATPLOTLIB:
        console.print("[yellow]⚠ matplotlib not available - skipping sweep plots[/yellow]")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not sweep_result.trials:
        console.print("[yellow]No trials to plot[/yellow]")
        return

    # Plot 1: Trial performance over time
    fig, ax = plt.subplots(figsize=(12, 6))

    trial_ids = [t.trial_id for t in sweep_result.trials]
    metrics = [t.primary_metric for t in sweep_result.trials]
    colors = ["green" if m > 0 else "red" for m in metrics]

    ax.scatter(trial_ids, metrics, c=colors, s=100, alpha=0.6, edgecolor="black")

    # Add trend line
    if len(trial_ids) > 1:
        z = np.polyfit(trial_ids, metrics, 2)
        p = np.poly1d(z)
        ax.plot(trial_ids, p(trial_ids), "b--", alpha=0.5, linewidth=2)

    ax.set_ylabel(sweep_result.objective_metric, fontsize=12, fontweight="bold")
    ax.set_xlabel("Trial", fontsize=12, fontweight="bold")
    ax.set_title("Sweep Progress", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "sweep_progress.png", dpi=150, bbox_inches="tight")
    plt.close()

    console.print(f"[green]✓ Saved: sweep_progress.png[/green]")

    # Plot 2: Best trial metrics
    best_trial = sweep_result.get_best_trial()
    if best_trial and best_trial.hyperparameters:
        fig, ax = plt.subplots(figsize=(10, 6))

        hp_names = list(best_trial.hyperparameters.keys())
        hp_values = list(best_trial.hyperparameters.values())

        # Convert to numeric if possible, otherwise show as is
        numeric_values = []
        for v in hp_values:
            try:
                numeric_values.append(float(v))
            except (TypeError, ValueError):
                numeric_values.append(hash(str(v)) % 100)

        colors = plt.get_cmap("Set2")(range(len(hp_names)))
        ax.barh(hp_names, numeric_values, color=colors, alpha=0.8, edgecolor="black")

        ax.set_xlabel("Value", fontsize=12, fontweight="bold")
        ax.set_title(f"Best Configuration ({sweep_result.objective_metric}: {best_trial.primary_metric:.4f})",
                     fontsize=14, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "best_configuration.png", dpi=150, bbox_inches="tight")
        plt.close()

        console.print(f"[green]✓ Saved: best_configuration.png[/green]")

    console.print(f"\n[green]Plots saved to: {output_dir}[/green]\n")


def daf_export_comparison_html(
    comparison_report: "ComparisonReport",  # type: ignore
    output_path: Path | str = "comparison_report.html",
    console: Optional[Console] = None,
) -> None:
    """Export comparison report as interactive HTML.

    Args:
        comparison_report: Report from daf_compare_policies
        output_path: Output path for HTML file
        console: Optional Rich console for output
    """
    if console is None:
        console = Console()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate HTML
    html_parts = []
    html_parts.append("<!DOCTYPE html>")
    html_parts.append("<html>")
    html_parts.append("<head>")
    html_parts.append("  <meta charset='utf-8'>")
    html_parts.append("  <title>Policy Comparison Report</title>")
    html_parts.append("  <style>")
    html_parts.append("    body { font-family: Arial, sans-serif; margin: 20px; }")
    html_parts.append("    h1 { color: #333; }")
    html_parts.append("    table { border-collapse: collapse; width: 100%; margin: 20px 0; }")
    html_parts.append("    th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }")
    html_parts.append("    th { background-color: #4CAF50; color: white; }")
    html_parts.append("    tr:nth-child(even) { background-color: #f2f2f2; }")
    html_parts.append("    .significant { font-weight: bold; color: green; }")
    html_parts.append("    .not-significant { color: gray; }")
    html_parts.append("  </style>")
    html_parts.append("</head>")
    html_parts.append("<body>")

    # Header
    html_parts.append(f"<h1>Policy Comparison Report</h1>")
    html_parts.append(f"<p>Generated: {comparison_report.timestamp.isoformat()}</p>")
    html_parts.append(f"<p>Policies: {len(comparison_report.policies)}, Missions: {len(comparison_report.missions)}</p>")

    # Summary table
    html_parts.append("<h2>Summary Statistics</h2>")
    html_parts.append("<table>")
    html_parts.append("  <tr><th>Policy</th><th>Avg Reward</th><th>Std Dev</th></tr>")

    for policy in comparison_report.policies:
        avg = comparison_report.policy_averages.get(policy, 0.0)
        std = comparison_report.policy_std_devs.get(policy, 0.0)
        html_parts.append(f"  <tr><td>{policy}</td><td>{avg:.4f}</td><td>{std:.4f}</td></tr>")

    html_parts.append("</table>")

    # Pairwise comparisons
    if comparison_report.pairwise_comparisons:
        html_parts.append("<h2>Pairwise Comparisons</h2>")
        html_parts.append("<table>")
        html_parts.append("  <tr><th>Policy A</th><th>Policy B</th><th>A Reward</th><th>B Reward</th><th>P-value</th><th>Significant</th></tr>")

        for (policy_a, policy_b), comp in comparison_report.pairwise_comparisons.items():
            sig_class = "significant" if comp.is_significant else "not-significant"
            sig_text = "Yes" if comp.is_significant else "No"

            html_parts.append(
                f"  <tr><td>{policy_a}</td><td>{policy_b}</td>"
                f"<td>{comp.avg_reward_a:.4f}</td><td>{comp.avg_reward_b:.4f}</td>"
                f"<td>{comp.p_value:.4f}</td><td class='{sig_class}'>{sig_text}</td></tr>"
            )

        html_parts.append("</table>")

    html_parts.append("</body>")
    html_parts.append("</html>")

    with open(output_path, "w") as f:
        f.write("\n".join(html_parts))

    console.print(f"[green]✓ HTML report saved: {output_path}[/green]\n")


def daf_generate_leaderboard(
    comparison_report: "ComparisonReport",  # type: ignore
    output_path: Optional[Path | str] = None,
    console: Optional[Console] = None,
) -> str:
    """Generate policy leaderboard table.

    Args:
        comparison_report: Report from daf_compare_policies
        output_path: Optional path to save leaderboard as JSON
        console: Optional Rich console for output

    Returns:
        Markdown-formatted leaderboard string
    """
    if console is None:
        console = Console()

    # Sort policies by average reward
    sorted_policies = sorted(
        comparison_report.policies,
        key=lambda p: comparison_report.policy_averages.get(p, 0.0),
        reverse=True,
    )

    # Generate markdown table
    lines = []
    lines.append("| Rank | Policy | Avg Reward | Std Dev |")
    lines.append("|------|--------|-----------|---------|")

    for rank, policy in enumerate(sorted_policies, 1):
        avg = comparison_report.policy_averages.get(policy, 0.0)
        std = comparison_report.policy_std_devs.get(policy, 0.0)
        lines.append(f"| {rank} | {policy} | {avg:.4f} | {std:.4f} |")

    markdown_table = "\n".join(lines)

    # Optionally save as JSON
    if output_path:
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        leaderboard_data = {
            "leaderboard": [
                {
                    "rank": rank,
                    "policy": policy,
                    "avg_reward": comparison_report.policy_averages.get(policy, 0.0),
                    "std_dev": comparison_report.policy_std_devs.get(policy, 0.0),
                }
                for rank, policy in enumerate(sorted_policies, 1)
            ]
        }

        with open(output_path, "w") as f:
            json.dump(leaderboard_data, f, indent=2)

        console.print(f"[green]✓ Leaderboard saved: {output_path}[/green]")

    return markdown_table

