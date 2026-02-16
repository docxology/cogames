"""Visualization utilities for DAF analytics.

DAF sidecar utility: Extends visualization patterns from `scripts/run_evaluation.py`.

Provides comprehensive plotting and visualization for:
- Training curves and learning dynamics
- Policy comparisons with statistical analysis
- Hyperparameter sweep heatmaps and parallel coordinates
- Cross-simulation outcome analysis
- Agent behavior and resource collection patterns
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from daf.src.eval.comparison import ComparisonReport
    from daf.src.eval.sweeps import SweepResult, SweepTrialResult

logger = logging.getLogger("daf.visualization")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.patches as mpatches

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


# ============================================================================
# Color Palettes and Styling
# ============================================================================

# Professional color palettes
PALETTE_VIRIDIS = ["#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725"]
PALETTE_PLASMA = ["#0d0887", "#6a00a8", "#b12a90", "#e16462", "#fca636", "#f0f921"]
PALETTE_COOLWARM = ["#3b4cc0", "#6788ee", "#9abbff", "#c9d7f0", "#edd1c2", "#f7a789", "#e26952", "#b40426"]
PALETTE_COGAMES = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#f39c12", "#1abc9c", "#34495e", "#e67e22"]


def _get_colormap(name: str = "viridis"):
    """Get a matplotlib colormap, with fallback for missing dependencies."""
    if not HAS_MATPLOTLIB:
        return None
    try:
        return plt.get_cmap(name)
    except ValueError:
        return plt.get_cmap("viridis")


def _setup_plot_style():
    """Configure matplotlib style for consistent aesthetics."""
    if not HAS_MATPLOTLIB:
        return
    
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "ggplot")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f9fa",
        "axes.edgecolor": "#dee2e6",
        "axes.labelcolor": "#212529",
        "text.color": "#212529",
        "xtick.color": "#495057",
        "ytick.color": "#495057",
        "grid.color": "#dee2e6",
        "grid.alpha": 0.6,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


# ============================================================================
# Training Curve Visualization
# ============================================================================

def daf_plot_training_curves(
    checkpoint_dir: Path | str,
    output_dir: Path | str = "training_plots",
    smoothing: int = 10,
    metrics: Optional[list[str]] = None,
    console: Optional[Console] = None,
) -> list[Path]:
    """Visualize training progress from checkpoint directory.

    Args:
        checkpoint_dir: Directory containing training checkpoints or logs
        output_dir: Output directory for plots
        smoothing: Window size for moving average smoothing
        metrics: Specific metrics to plot (None = all available)
        console: Optional Rich console for output

    Returns:
        List of paths to generated plot files
    """
    if console is None:
        console = Console()

    if not HAS_MATPLOTLIB:
        console.print("[yellow]⚠ matplotlib not available - skipping training curve plots[/yellow]")
        return []

    _setup_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(checkpoint_dir)

    generated_files = []

    # Look for training logs
    log_files = list(checkpoint_dir.glob("*.json")) + list(checkpoint_dir.glob("**/training_log.json"))
    
    if not log_files:
        console.print("[yellow]⚠ No training logs found in checkpoint directory[/yellow]")
        console.print("[dim]Looking for: *.json or training_log.json[/dim]")
        return []

    for log_file in log_files:
        try:
            with open(log_file) as f:
                data = json.load(f)
            
            if isinstance(data, list):
                # List of step records
                steps = [d.get("step", i) for i, d in enumerate(data)]
                available_metrics = set()
                for d in data:
                    available_metrics.update(k for k in d.keys() if k != "step")
                
                plot_metrics = metrics or list(available_metrics)[:6]  # Limit to 6 metrics
                
                if plot_metrics:
                    fig, axes = plt.subplots(
                        len(plot_metrics), 1, 
                        figsize=(12, 3 * len(plot_metrics)),
                        squeeze=False
                    )
                    
                    for idx, metric in enumerate(plot_metrics):
                        ax = axes[idx, 0]
                        values = [d.get(metric, np.nan) for d in data]
                        
                        # Apply smoothing
                        if smoothing > 1 and len(values) > smoothing:
                            kernel = np.ones(smoothing) / smoothing
                            smoothed = np.convolve(values, kernel, mode="valid")
                            smooth_steps = steps[smoothing-1:]
                            ax.plot(steps, values, alpha=0.3, color=PALETTE_COGAMES[idx % len(PALETTE_COGAMES)])
                            ax.plot(smooth_steps, smoothed, linewidth=2, color=PALETTE_COGAMES[idx % len(PALETTE_COGAMES)])
                        else:
                            ax.plot(steps, values, linewidth=2, color=PALETTE_COGAMES[idx % len(PALETTE_COGAMES)])
                        
                        ax.set_ylabel(metric, fontweight="bold")
                        ax.set_xlabel("Step")
                        ax.grid(True, alpha=0.3)
                    
                    plt.suptitle(f"Training Progress: {log_file.stem}", fontsize=14, fontweight="bold")
                    plt.tight_layout()
                    
                    out_path = output_dir / f"training_curves_{log_file.stem}.png"
                    plt.savefig(out_path, dpi=150, bbox_inches="tight")
                    plt.close()
                    
                    generated_files.append(out_path)
                    console.print(f"[green]✓ Saved: {out_path.name}[/green]")
                    
        except Exception as e:
            logger.warning(f"Failed to process {log_file}: {e}")

    return generated_files


def daf_plot_learning_dynamics(
    training_data: dict[str, list[float]],
    output_path: Path | str = "learning_dynamics.png",
    title: str = "Learning Dynamics",
    console: Optional[Console] = None,
) -> Optional[Path]:
    """Plot learning dynamics showing reward, loss, and exploration metrics.

    Args:
        training_data: Dict with keys like 'reward', 'loss', 'entropy', 'epsilon'
        output_path: Output path for plot
        title: Plot title
        console: Optional Rich console

    Returns:
        Path to generated plot or None if failed
    """
    if console is None:
        console = Console()

    if not HAS_MATPLOTLIB:
        console.print("[yellow]⚠ matplotlib not available[/yellow]")
        return None

    _setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_metrics = len(training_data)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 3 * num_metrics), sharex=True)
    if num_metrics == 1:
        axes = [axes]

    colors = PALETTE_COGAMES[:num_metrics]
    
    for idx, (metric_name, values) in enumerate(training_data.items()):
        ax = axes[idx]
        steps = np.arange(len(values))
        
        ax.fill_between(steps, 0, values, alpha=0.3, color=colors[idx])
        ax.plot(steps, values, linewidth=2, color=colors[idx])
        
        ax.set_ylabel(metric_name, fontweight="bold")
        ax.grid(True, alpha=0.3)
        
        # Add min/max/final annotations
        if values:
            ax.axhline(y=np.mean(values), color=colors[idx], linestyle="--", alpha=0.5, label=f"Mean: {np.mean(values):.3f}")
            ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Training Step", fontweight="bold")
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    console.print(f"[green]✓ Saved: {output_path}[/green]")
    return output_path


# ============================================================================
# Policy Comparison Visualization
# ============================================================================

def daf_plot_policy_comparison(
    comparison_report: "ComparisonReport",
    output_dir: Path | str = "comparison_plots",
    console: Optional[Console] = None,
) -> list[Path]:
    """Generate comprehensive comparison plots from ComparisonReport.

    Args:
        comparison_report: Report from daf_compare_policies
        output_dir: Output directory for plots
        console: Optional Rich console for output

    Returns:
        List of paths to generated plot files
    """
    if console is None:
        console = Console()

    if not HAS_MATPLOTLIB:
        console.print("[yellow]⚠ matplotlib not available - skipping comparison plots[/yellow]")
        return []

    _setup_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    policies = comparison_report.policies
    avg_rewards = [comparison_report.policy_averages.get(p, 0.0) for p in policies]
    std_devs = [comparison_report.policy_std_devs.get(p, 0.0) for p in policies]

    # Plot 1: Bar chart with error bars
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(policies))
    colors = PALETTE_COGAMES[:len(policies)]

    bars = ax.bar(x, avg_rewards, yerr=std_devs, color=colors, alpha=0.85, 
                  capsize=5, edgecolor="black", linewidth=1.2)

    # Add value labels on bars
    for bar, val, std in zip(bars, avg_rewards, std_devs):
        ax.annotate(f"{val:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Average Reward", fontsize=12, fontweight="bold")
    ax.set_xlabel("Policy", fontsize=12, fontweight="bold")
    ax.set_title("Policy Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "policy_rewards_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    generated_files.append(out_path)
    console.print(f"[green]✓ Saved: policy_rewards_comparison.png[/green]")

    # Plot 2: Per-mission grouped bar chart
    if comparison_report.policy_mission_rewards:
        missions = comparison_report.missions
        num_policies = len(policies)
        
        fig, ax = plt.subplots(figsize=(max(12, len(missions) * 1.5), 6))
        x = np.arange(len(missions))
        width = 0.8 / num_policies

        for i, policy in enumerate(policies):
            rewards = []
            errors = []
            for mission in missions:
                mission_rewards = comparison_report.policy_mission_rewards.get(policy, {}).get(mission, [])
                avg = np.mean(mission_rewards) if mission_rewards else 0.0
                std = np.std(mission_rewards) if len(mission_rewards) > 1 else 0.0
                rewards.append(avg)
                errors.append(std)

            offset = width * (i - num_policies / 2 + 0.5)
            ax.bar(x + offset, rewards, width, yerr=errors, label=policy, 
                   alpha=0.85, edgecolor="black", linewidth=0.8, capsize=3)

        ax.set_ylabel("Average Reward", fontsize=12, fontweight="bold")
        ax.set_xlabel("Mission", fontsize=12, fontweight="bold")
        ax.set_title("Performance by Mission", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(missions, rotation=45, ha="right")
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        out_path = output_dir / "performance_by_mission.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        generated_files.append(out_path)
        console.print(f"[green]✓ Saved: performance_by_mission.png[/green]")

    # Plot 3: Reward distribution violin plots
    if comparison_report.policy_mission_rewards:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        all_rewards = []
        labels = []
        for policy in policies:
            rewards = []
            for mission in missions:
                mission_rewards = comparison_report.policy_mission_rewards.get(policy, {}).get(mission, [])
                rewards.extend(mission_rewards)
            all_rewards.append(rewards if rewards else [0])
            labels.append(policy)
        
        parts = ax.violinplot(all_rewards, positions=range(len(policies)), showmeans=True, showmedians=True)
        
        # Color the violins
        for idx, pc in enumerate(parts.get("bodies", [])):
            pc.set_facecolor(PALETTE_COGAMES[idx % len(PALETTE_COGAMES)])
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(policies)))
        ax.set_xticklabels(policies, rotation=45, ha="right")
        ax.set_ylabel("Reward Distribution", fontsize=12, fontweight="bold")
        ax.set_title("Reward Distribution by Policy", fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        out_path = output_dir / "reward_distributions.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        generated_files.append(out_path)
        console.print(f"[green]✓ Saved: reward_distributions.png[/green]")

    console.print(f"\n[green]Plots saved to: {output_dir}[/green]\n")
    return generated_files


def daf_plot_detailed_metrics_comparison(
    comparison_report: "ComparisonReport",
    output_dir: Path | str = "comparison_plots",
    console: Optional[Console] = None,
) -> list[Path]:
    """Generate detailed metric comparison plots from agent metrics.

    Args:
        comparison_report: Report with policy_detailed_metrics populated
        output_dir: Output directory for plots
        console: Optional Rich console for output

    Returns:
        List of paths to generated plot files
    """
    if console is None:
        console = Console()

    if not HAS_MATPLOTLIB:
        console.print("[yellow]⚠ matplotlib not available[/yellow]")
        return []

    detailed_metrics = getattr(comparison_report, 'policy_detailed_metrics', {})
    if not detailed_metrics:
        console.print("[dim]No detailed metrics available for visualization[/dim]")
        return []

    _setup_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    policies = list(detailed_metrics.keys())
    if not policies:
        return []

    # Get first mission's metrics (we'll combine all missions)
    first_policy = policies[0]
    first_mission = list(detailed_metrics[first_policy].keys())[0] if detailed_metrics[first_policy] else None
    if not first_mission:
        return []

    # Aggregate metrics across missions for each policy
    aggregated_metrics = {}
    for policy in policies:
        aggregated_metrics[policy] = {}
        for mission, metrics in detailed_metrics[policy].items():
            for metric_name, value in metrics.items():
                if metric_name not in aggregated_metrics[policy]:
                    aggregated_metrics[policy][metric_name] = []
                aggregated_metrics[policy][metric_name].append(value)
        # Average across missions
        for metric_name in aggregated_metrics[policy]:
            values = aggregated_metrics[policy][metric_name]
            aggregated_metrics[policy][metric_name] = np.mean(values) if values else 0.0

    # Group metrics by category
    metric_categories = {
        "Resources Gained": ["carbon.gained", "silicon.gained", "oxygen.gained", "germanium.gained"],
        "Resources Held": ["carbon.amount", "silicon.amount", "oxygen.amount", "germanium.amount"],
        "Energy": ["energy.amount", "energy.gained", "energy.lost"],
        "Actions": ["action.move.success", "action.move.failed", "action.failed", "action.noop.success", "action.change_vibe.success"],
        "Inventory": ["inventory.diversity", "inventory.diversity.ge.2", "inventory.diversity.ge.3", "inventory.diversity.ge.4", "inventory.diversity.ge.5"],
    }

    # Plot each category
    for category_name, metric_names in metric_categories.items():
        # Filter to metrics that exist
        available_metrics = [m for m in metric_names if any(m in aggregated_metrics[p] for p in policies)]
        if not available_metrics:
            continue

        fig, ax = plt.subplots(figsize=(max(10, len(available_metrics) * 1.5), 6))
        x = np.arange(len(available_metrics))
        width = 0.8 / len(policies)

        for i, policy in enumerate(policies):
            values = [aggregated_metrics[policy].get(m, 0.0) for m in available_metrics]
            offset = width * (i - len(policies) / 2 + 0.5)
            bars = ax.bar(x + offset, values, width, label=policy,
                         color=PALETTE_COGAMES[i % len(PALETTE_COGAMES)], alpha=0.85, 
                         edgecolor="black", linewidth=0.8)
            
            # Add value labels on top of bars
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.annotate(f"{val:.0f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                               xytext=(0, 3), textcoords="offset points",
                               ha="center", va="bottom", fontsize=8, rotation=45)

        ax.set_ylabel("Value", fontsize=12, fontweight="bold")
        ax.set_xlabel("Metric", fontsize=12, fontweight="bold")
        ax.set_title(f"{category_name} by Policy", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([m.split(".")[-1] for m in available_metrics], rotation=45, ha="right")
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        filename = f"metrics_{category_name.lower().replace(' ', '_')}.png"
        out_path = output_dir / filename
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        generated_files.append(out_path)
        console.print(f"[green]✓ Saved: {filename}[/green]")

    # Create summary radar chart with key metrics
    key_metrics = ["carbon.gained", "silicon.gained", "energy.gained", "inventory.diversity", "action.move.success"]
    available_key = [m for m in key_metrics if any(m in aggregated_metrics[p] for p in policies)]
    
    if len(available_key) >= 3:
        radar_path = _create_metrics_radar(policies, aggregated_metrics, available_key, output_dir / "metrics_radar.png")
        if radar_path:
            generated_files.append(radar_path)
            console.print(f"[green]✓ Saved: metrics_radar.png[/green]")

    # Create action distribution pie charts
    action_path = _create_action_distribution(policies, aggregated_metrics, output_dir / "action_distribution.png")
    if action_path:
        generated_files.append(action_path)
        console.print(f"[green]✓ Saved: action_distribution.png[/green]")

    return generated_files


def _create_metrics_radar(
    policies: list[str],
    aggregated_metrics: dict[str, dict[str, float]],
    metric_names: list[str],
    output_path: Path,
) -> Optional[Path]:
    """Create radar chart for key metrics comparison."""
    if not HAS_MATPLOTLIB:
        return None

    # Calculate angles
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for idx, policy in enumerate(policies):
        values = [aggregated_metrics[policy].get(m, 0.0) for m in metric_names]
        
        # Normalize values
        max_vals = [max(aggregated_metrics[p].get(m, 0.0) for p in policies) or 1 for m in metric_names]
        normalized = [v / m if m > 0 else 0 for v, m in zip(values, max_vals)]
        normalized += normalized[:1]
        
        ax.plot(angles, normalized, "o-", linewidth=2, label=policy, 
               color=PALETTE_COGAMES[idx % len(PALETTE_COGAMES)])
        ax.fill(angles, normalized, alpha=0.25, color=PALETTE_COGAMES[idx % len(PALETTE_COGAMES)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.split(".")[-1].replace("_", " ").title() for m in metric_names], fontsize=10)
    ax.set_title("Key Metrics Comparison", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def _create_action_distribution(
    policies: list[str],
    aggregated_metrics: dict[str, dict[str, float]],
    output_path: Path,
) -> Optional[Path]:
    """Create pie charts showing action distribution for each policy."""
    if not HAS_MATPLOTLIB:
        return None

    action_metrics = ["action.move.success", "action.move.failed", "action.noop.success", "action.change_vibe.success"]
    action_labels = ["Move Success", "Move Failed", "Noop", "Change Vibe"]

    fig, axes = plt.subplots(1, len(policies), figsize=(5 * len(policies), 5))
    if len(policies) == 1:
        axes = [axes]

    for ax, policy in zip(axes, policies):
        values = [aggregated_metrics[policy].get(m, 0.0) for m in action_metrics]
        
        # Only plot non-zero slices
        non_zero_values = []
        non_zero_labels = []
        for v, l in zip(values, action_labels):
            if v > 0:
                non_zero_values.append(v)
                non_zero_labels.append(l)
        
        if non_zero_values:
            colors = PALETTE_COGAMES[:len(non_zero_values)]
            wedges, texts, autotexts = ax.pie(non_zero_values, labels=non_zero_labels, colors=colors,
                                               autopct="%1.1f%%", startangle=90, 
                                               textprops={"fontsize": 9})
            ax.set_title(policy, fontsize=12, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
            ax.set_title(policy, fontsize=12, fontweight="bold")

    plt.suptitle("Action Distribution by Policy", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def daf_plot_policy_radar(
    policy_metrics: dict[str, dict[str, float]],
    output_path: Path | str = "policy_radar.png",
    title: str = "Multi-Metric Policy Comparison",
    console: Optional[Console] = None,
) -> Optional[Path]:
    """Generate radar/spider chart comparing policies across multiple metrics.

    Args:
        policy_metrics: Dict of policy_name -> {metric_name: value}
        output_path: Output path for plot
        title: Plot title
        console: Optional Rich console

    Returns:
        Path to generated plot or None if failed
    """
    if console is None:
        console = Console()

    if not HAS_MATPLOTLIB:
        console.print("[yellow]⚠ matplotlib not available[/yellow]")
        return None

    if not policy_metrics:
        console.print("[yellow]No policy metrics provided[/yellow]")
        return None

    _setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get all unique metrics
    all_metrics = set()
    for metrics in policy_metrics.values():
        all_metrics.update(metrics.keys())
    metrics_list = sorted(all_metrics)
    
    if len(metrics_list) < 3:
        console.print("[yellow]Need at least 3 metrics for radar chart[/yellow]")
        return None

    # Calculate angles for radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics_list), endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = PALETTE_COGAMES[:len(policy_metrics)]
    
    for idx, (policy_name, metrics) in enumerate(policy_metrics.items()):
        values = [metrics.get(m, 0) for m in metrics_list]
        
        # Normalize values to 0-1 range
        max_val = max(values) if values and max(values) > 0 else 1
        values = [v / max_val for v in values]
        values += values[:1]  # Complete the loop
        
        ax.plot(angles, values, "o-", linewidth=2, label=policy_name, color=colors[idx])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_list, fontsize=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    console.print(f"[green]✓ Saved: {output_path}[/green]")
    return output_path


def daf_plot_metric_distributions(
    comparison_report: "ComparisonReport",
    metric_name: str = "performance_score",
    output_dir: Path | str = "comparison_plots",
    console: Optional[Console] = None,
) -> list[Path]:
    """Plot distribution of a specific metric across replicates.
    
    Args:
        comparison_report: Report containing replicate data
        metric_name: Metric to plot (e.g. 'performance_score', 'energy.gained')
        output_dir: Output directory
        console: Optional console
        
    Returns:
        List of generated plot paths
    """
    if console is None:
        console = Console()
        
    if not HAS_MATPLOTLIB:
        return []
        
    if not comparison_report.policy_replicate_metrics:
        logger.warning("No replicate metrics available for distribution plot")
        return []

    _setup_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect data
    plot_data = [] # List of lists of values
    labels = []
    colors = []
    
    policies = comparison_report.policies
    for i, policy in enumerate(policies):
        values = []
        if policy in comparison_report.policy_replicate_metrics:
            for mission_replicates in comparison_report.policy_replicate_metrics[policy].values():
                for rep in mission_replicates:
                    if metric_name in rep:
                        values.append(rep[metric_name])
        
        if values:
            plot_data.append(values)
            labels.append(policy)
            colors.append(PALETTE_COGAMES[i % len(PALETTE_COGAMES)])
            
    if not plot_data:
        logger.warning(f"No data found for metric {metric_name}")
        return []
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create violin plot
    parts = ax.violinplot(plot_data, showmeans=True, showmedians=False)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
        
    # Overlay individual points (beeswarm style)
    for i, (values, color) in enumerate(zip(plot_data, colors)):
        x = np.random.normal(i + 1, 0.04, size=len(values))
        ax.scatter(x, values, alpha=0.9, color=color, edgecolor='black', linewidth=0.5, s=20)
        
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_title(f"Distribution of {metric_name}", fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filename = f"distribution_{metric_name.replace('.', '_')}.png"
    out_path = output_dir / filename
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]✓ Saved: {filename}[/green]")
    return [out_path]


def daf_plot_metric_correlations(
    comparison_report: "ComparisonReport",
    policy_name: str,
    output_dir: Path | str = "comparison_plots",
    top_k: int = 6,
    console: Optional[Console] = None,
) -> list[Path]:
    """Plot correlations between metrics for a specific policy.
    
    Args:
        comparison_report: Report containing replicate data
        policy_name: Policy to analyze
        output_dir: Output directory
        top_k: Number of top correlated metrics to plot against performance_score
        console: Optional console
        
    Returns:
        List of generated plot paths
    """
    if console is None:
        console = Console()
        
    if not HAS_MATPLOTLIB:
        return []

    from daf.src.eval.comparison import daf_correlate_metrics
    
    # Calculate correlations
    correlations = daf_correlate_metrics(comparison_report, policy_name, "performance_score")
    if not correlations:
        return []
        
    _setup_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter to top k metrics
    top_metrics = list(correlations.keys())[:top_k]
    
    # Collect data for plotting
    data = []
    if policy_name in comparison_report.policy_replicate_metrics:
        for mission_replicates in comparison_report.policy_replicate_metrics[policy_name].values():
            for rep in mission_replicates:
                data.append(rep)
    
    if not data:
        return []
        
    import pandas as pd
    df = pd.DataFrame(data)
    
    # Create scatter plots
    num_plots = len(top_metrics)
    cols = 3
    rows = (num_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    axes = np.array(axes).flatten()
    
    for i, metric in enumerate(top_metrics):
        if i >= len(axes): break
        ax = axes[i]
        
        if metric in df.columns and "performance_score" in df.columns:
            x = df[metric]
            y = df["performance_score"]
            
            # Scatter plot with regression line
            ax.scatter(x, y, alpha=0.6, color=PALETTE_COGAMES[i % len(PALETTE_COGAMES)])
            
            if len(x) > 1:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                ax.plot(x, p(x), "r--", alpha=0.8, linewidth=1)
                
            ax.set_xlabel(metric, fontsize=10, fontweight='bold')
            ax.set_ylabel("Performance Score", fontsize=10, fontweight='bold')
            ax.set_title(f"r = {correlations[metric]:.2f}", fontsize=11)
            ax.grid(True, alpha=0.3)
            
    # Hide empty subplots
    for i in range(len(top_metrics), len(axes)):
        axes[i].axis('off')
        
    plt.suptitle(f"Top Correlations: {policy_name}", fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    filename = f"correlations_{policy_name}.png"
    out_path = output_dir / filename
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]✓ Saved: {filename}[/green]")
    return [out_path]
# ============================================================================
# Sweep Visualization
# ============================================================================

def daf_plot_sweep_results(
    sweep_result: "SweepResult",
    output_dir: Path | str = "sweep_plots",
    console: Optional[Console] = None,
) -> list[Path]:
    """Generate comprehensive plots from sweep results.

    Args:
        sweep_result: Result from daf_launch_sweep
        output_dir: Output directory for plots
        console: Optional Rich console for output

    Returns:
        List of paths to generated plot files
    """
    if console is None:
        console = Console()

    if not HAS_MATPLOTLIB:
        console.print("[yellow]⚠ matplotlib not available - skipping sweep plots[/yellow]")
        return []

    _setup_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    if not sweep_result.trials:
        console.print("[yellow]No trials to plot[/yellow]")
        return []

    # Plot 1: Trial performance over time with trend
    fig, ax = plt.subplots(figsize=(12, 6))

    trial_ids = [t.trial_id for t in sweep_result.trials]
    metrics = [t.primary_metric for t in sweep_result.trials]
    
    # Color by performance
    norm_metrics = np.array(metrics)
    if norm_metrics.max() != norm_metrics.min():
        norm_metrics = (norm_metrics - norm_metrics.min()) / (norm_metrics.max() - norm_metrics.min())
    else:
        norm_metrics = np.ones_like(norm_metrics) * 0.5
    
    cmap = _get_colormap("RdYlGn")
    colors = [cmap(v) for v in norm_metrics]

    ax.scatter(trial_ids, metrics, c=colors, s=100, alpha=0.8, edgecolor="black", linewidth=1)

    # Add trend line
    if len(trial_ids) > 2:
        z = np.polyfit(trial_ids, metrics, min(2, len(trial_ids) - 1))
        p = np.poly1d(z)
        x_smooth = np.linspace(min(trial_ids), max(trial_ids), 100)
        ax.plot(x_smooth, p(x_smooth), "b--", alpha=0.6, linewidth=2, label="Trend")

    # Mark best trial
    best_trial = sweep_result.get_best_trial()
    if best_trial:
        ax.axhline(y=best_trial.primary_metric, color="green", linestyle=":", 
                   alpha=0.7, linewidth=2, label=f"Best: {best_trial.primary_metric:.4f}")

    ax.set_ylabel(sweep_result.objective_metric, fontsize=12, fontweight="bold")
    ax.set_xlabel("Trial", fontsize=12, fontweight="bold")
    ax.set_title("Sweep Progress", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "sweep_progress.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    generated_files.append(out_path)
    console.print(f"[green]✓ Saved: sweep_progress.png[/green]")

    # Plot 2: Hyperparameter importance (if we have enough trials)
    if len(sweep_result.trials) >= 5:
        hp_importance = _compute_hyperparameter_importance(sweep_result)
        if hp_importance:
            fig, ax = plt.subplots(figsize=(10, max(4, len(hp_importance) * 0.5)))
            
            sorted_hp = sorted(hp_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            names = [h[0] for h in sorted_hp]
            values = [h[1] for h in sorted_hp]
            colors = ["green" if v > 0 else "red" for v in values]
            
            ax.barh(names, values, color=colors, alpha=0.7, edgecolor="black")
            ax.axvline(x=0, color="black", linewidth=0.8)
            ax.set_xlabel("Correlation with Performance", fontsize=12, fontweight="bold")
            ax.set_title("Hyperparameter Importance", fontsize=14, fontweight="bold")
            ax.grid(axis="x", alpha=0.3)
            
            plt.tight_layout()
            out_path = output_dir / "hyperparameter_importance.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            generated_files.append(out_path)
            console.print(f"[green]✓ Saved: hyperparameter_importance.png[/green]")

    # Plot 3: Best configuration bar chart
    if best_trial and best_trial.hyperparameters:
        fig, ax = plt.subplots(figsize=(10, max(4, len(best_trial.hyperparameters) * 0.6)))

        hp_names = list(best_trial.hyperparameters.keys())
        hp_values = list(best_trial.hyperparameters.values())

        # Convert to display values
        display_values = []
        display_labels = []
        for name, val in zip(hp_names, hp_values):
            try:
                display_values.append(float(val))
                display_labels.append(f"{val}")
            except (TypeError, ValueError):
                display_values.append(hash(str(val)) % 100)
                display_labels.append(str(val))

        colors = PALETTE_COGAMES[:len(hp_names)]
        bars = ax.barh(hp_names, display_values, color=colors, alpha=0.8, edgecolor="black")
        
        # Add value labels
        for bar, label in zip(bars, display_labels):
            ax.annotate(label, xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=10)

        ax.set_xlabel("Value", fontsize=12, fontweight="bold")
        ax.set_title(f"Best Configuration ({sweep_result.objective_metric}: {best_trial.primary_metric:.4f})",
                     fontsize=14, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        out_path = output_dir / "best_configuration.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        generated_files.append(out_path)
        console.print(f"[green]✓ Saved: best_configuration.png[/green]")

    console.print(f"\n[green]Plots saved to: {output_dir}[/green]\n")
    return generated_files


def daf_plot_sweep_heatmap(
    sweep_result: "SweepResult",
    hp_x: str,
    hp_y: str,
    output_path: Path | str = "sweep_heatmap.png",
    aggregation: str = "mean",
    console: Optional[Console] = None,
) -> Optional[Path]:
    """Generate heatmap of performance across two hyperparameters.

    Args:
        sweep_result: Result from daf_launch_sweep
        hp_x: Hyperparameter name for x-axis
        hp_y: Hyperparameter name for y-axis
        output_path: Output path for plot
        aggregation: How to aggregate multiple trials with same hp values ("mean", "max", "min")
        console: Optional Rich console

    Returns:
        Path to generated plot or None if failed
    """
    if console is None:
        console = Console()

    if not HAS_MATPLOTLIB:
        console.print("[yellow]⚠ matplotlib not available[/yellow]")
        return None

    if not sweep_result.trials:
        console.print("[yellow]No trials to plot[/yellow]")
        return None

    _setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract unique values for each hyperparameter
    x_values = sorted(set(t.hyperparameters.get(hp_x) for t in sweep_result.trials if hp_x in t.hyperparameters))
    y_values = sorted(set(t.hyperparameters.get(hp_y) for t in sweep_result.trials if hp_y in t.hyperparameters))

    if len(x_values) < 2 or len(y_values) < 2:
        console.print(f"[yellow]Need at least 2 unique values for both {hp_x} and {hp_y}[/yellow]")
        return None

    # Build performance matrix
    perf_matrix = np.full((len(y_values), len(x_values)), np.nan)
    counts = np.zeros((len(y_values), len(x_values)))
    
    for trial in sweep_result.trials:
        if hp_x in trial.hyperparameters and hp_y in trial.hyperparameters:
            x_val = trial.hyperparameters[hp_x]
            y_val = trial.hyperparameters[hp_y]
            
            try:
                x_idx = x_values.index(x_val)
                y_idx = y_values.index(y_val)
            except ValueError:
                continue
            
            if np.isnan(perf_matrix[y_idx, x_idx]):
                perf_matrix[y_idx, x_idx] = trial.primary_metric
            else:
                if aggregation == "mean":
                    perf_matrix[y_idx, x_idx] = (perf_matrix[y_idx, x_idx] * counts[y_idx, x_idx] + trial.primary_metric) / (counts[y_idx, x_idx] + 1)
                elif aggregation == "max":
                    perf_matrix[y_idx, x_idx] = max(perf_matrix[y_idx, x_idx], trial.primary_metric)
                elif aggregation == "min":
                    perf_matrix[y_idx, x_idx] = min(perf_matrix[y_idx, x_idx], trial.primary_metric)
            
            counts[y_idx, x_idx] += 1

    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(8, len(x_values)), max(6, len(y_values) * 0.8)))
    
    cmap = _get_colormap("RdYlGn" if sweep_result.optimize_direction == "maximize" else "RdYlGn_r")
    im = ax.imshow(perf_matrix, cmap=cmap, aspect="auto")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(sweep_result.objective_metric, fontsize=12, fontweight="bold")
    
    # Set ticks
    ax.set_xticks(range(len(x_values)))
    ax.set_yticks(range(len(y_values)))
    ax.set_xticklabels([str(v) for v in x_values], rotation=45, ha="right")
    ax.set_yticklabels([str(v) for v in y_values])
    
    # Add value annotations
    for i in range(len(y_values)):
        for j in range(len(x_values)):
            if not np.isnan(perf_matrix[i, j]):
                text_color = "white" if perf_matrix[i, j] < np.nanmean(perf_matrix) else "black"
                ax.annotate(f"{perf_matrix[i, j]:.3f}", xy=(j, i),
                           ha="center", va="center", fontsize=9, color=text_color)
    
    ax.set_xlabel(hp_x, fontsize=12, fontweight="bold")
    ax.set_ylabel(hp_y, fontsize=12, fontweight="bold")
    ax.set_title(f"Performance Heatmap: {hp_x} vs {hp_y}", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    console.print(f"[green]✓ Saved: {output_path}[/green]")
    return output_path


def daf_plot_sweep_parallel_coordinates(
    sweep_result: "SweepResult",
    output_path: Path | str = "sweep_parallel_coords.png",
    top_n: Optional[int] = None,
    console: Optional[Console] = None,
) -> Optional[Path]:
    """Generate parallel coordinates plot for hyperparameter exploration.

    Args:
        sweep_result: Result from daf_launch_sweep
        output_path: Output path for plot
        top_n: If specified, only plot top N trials
        console: Optional Rich console

    Returns:
        Path to generated plot or None if failed
    """
    if console is None:
        console = Console()

    if not HAS_MATPLOTLIB:
        console.print("[yellow]⚠ matplotlib not available[/yellow]")
        return None

    if not sweep_result.trials:
        console.print("[yellow]No trials to plot[/yellow]")
        return None

    _setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get trials to plot
    trials = sweep_result.top_trials(top_n) if top_n else sweep_result.trials
    
    # Get all hyperparameter names
    all_hp_names = set()
    for trial in trials:
        all_hp_names.update(trial.hyperparameters.keys())
    hp_names = sorted(all_hp_names)
    
    if len(hp_names) < 2:
        console.print("[yellow]Need at least 2 hyperparameters for parallel coordinates[/yellow]")
        return None

    # Add metric as last dimension
    dimensions = hp_names + [sweep_result.objective_metric]
    
    # Normalize values for each dimension
    dim_values = {dim: [] for dim in dimensions}
    for trial in trials:
        for hp in hp_names:
            val = trial.hyperparameters.get(hp, np.nan)
            try:
                dim_values[hp].append(float(val))
            except (TypeError, ValueError):
                dim_values[hp].append(hash(str(val)) % 100)
        dim_values[sweep_result.objective_metric].append(trial.primary_metric)
    
    # Normalize each dimension to [0, 1]
    normalized = {}
    for dim in dimensions:
        values = np.array(dim_values[dim])
        if np.nanmax(values) != np.nanmin(values):
            normalized[dim] = (values - np.nanmin(values)) / (np.nanmax(values) - np.nanmin(values))
        else:
            normalized[dim] = np.ones_like(values) * 0.5
    
    # Plot
    fig, ax = plt.subplots(figsize=(max(10, len(dimensions) * 1.5), 6))
    
    cmap = _get_colormap("RdYlGn" if sweep_result.optimize_direction == "maximize" else "RdYlGn_r")
    
    for trial_idx in range(len(trials)):
        y_values = [normalized[dim][trial_idx] for dim in dimensions]
        x_values = range(len(dimensions))
        
        metric_val = normalized[sweep_result.objective_metric][trial_idx]
        color = cmap(metric_val)
        
        ax.plot(x_values, y_values, "o-", color=color, alpha=0.6, linewidth=1.5, markersize=6)
    
    ax.set_xticks(range(len(dimensions)))
    ax.set_xticklabels(dimensions, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Normalized Value", fontsize=12, fontweight="bold")
    ax.set_title("Parallel Coordinates: Hyperparameter Space Exploration", fontsize=14, fontweight="bold")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(f"{sweep_result.objective_metric} (normalized)", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    console.print(f"[green]✓ Saved: {output_path}[/green]")
    return output_path


def _compute_hyperparameter_importance(sweep_result: "SweepResult") -> dict[str, float]:
    """Compute correlation between hyperparameters and performance."""
    from scipy.stats import spearmanr
    
    if len(sweep_result.trials) < 5:
        return {}
    
    # Get all hyperparameter names
    all_hp_names = set()
    for trial in sweep_result.trials:
        all_hp_names.update(trial.hyperparameters.keys())
    
    metrics = [t.primary_metric for t in sweep_result.trials]
    importance = {}
    
    for hp_name in all_hp_names:
        values = []
        valid_metrics = []
        
        for trial, metric in zip(sweep_result.trials, metrics):
            if hp_name in trial.hyperparameters:
                val = trial.hyperparameters[hp_name]
                try:
                    values.append(float(val))
                    valid_metrics.append(metric)
                except (TypeError, ValueError):
                    continue
        
        if len(values) >= 5:
            try:
                corr, _ = spearmanr(values, valid_metrics)
                if not np.isnan(corr):
                    importance[hp_name] = corr
            except Exception:
                pass
    
    return importance


# ============================================================================
# Cross-Simulation Analytics
# ============================================================================

def daf_plot_metrics_correlation_matrix(
    metrics_data: dict[str, list[float]],
    output_path: Path | str = "metrics_correlation.png",
    title: str = "Metrics Correlation Matrix",
    console: Optional[Console] = None,
) -> Optional[Path]:
    """Generate correlation matrix heatmap for simulation metrics.

    Args:
        metrics_data: Dict of metric_name -> list of values across simulations
        output_path: Output path for plot
        title: Plot title
        console: Optional Rich console

    Returns:
        Path to generated plot or None if failed
    """
    if console is None:
        console = Console()

    if not HAS_MATPLOTLIB:
        console.print("[yellow]⚠ matplotlib not available[/yellow]")
        return None

    if len(metrics_data) < 2:
        console.print("[yellow]Need at least 2 metrics for correlation matrix[/yellow]")
        return None

    _setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build correlation matrix
    metric_names = list(metrics_data.keys())
    n = len(metric_names)
    corr_matrix = np.zeros((n, n))
    
    from scipy.stats import pearsonr
    
    for i, m1 in enumerate(metric_names):
        for j, m2 in enumerate(metric_names):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                v1, v2 = metrics_data[m1], metrics_data[m2]
                min_len = min(len(v1), len(v2))
                if min_len >= 2:
                    try:
                        corr, _ = pearsonr(v1[:min_len], v2[:min_len])
                        corr_matrix[i, j] = corr if not np.isnan(corr) else 0
                    except Exception:
                        corr_matrix[i, j] = 0

    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(6, n * 0.8)))
    
    cmap = _get_colormap("RdBu_r")
    im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation", fontsize=12, fontweight="bold")
    
    # Set ticks
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(metric_names, rotation=45, ha="right")
    ax.set_yticklabels(metric_names)
    
    # Add correlation values
    for i in range(n):
        for j in range(n):
            text_color = "white" if abs(corr_matrix[i, j]) > 0.5 else "black"
            ax.annotate(f"{corr_matrix[i, j]:.2f}", xy=(j, i),
                       ha="center", va="center", fontsize=9, color=text_color)
    
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    console.print(f"[green]✓ Saved: {output_path}[/green]")
    return output_path


def daf_plot_episode_reward_distribution(
    episode_rewards: list[list[float]],
    policy_names: Optional[list[str]] = None,
    output_path: Path | str = "episode_reward_distribution.png",
    console: Optional[Console] = None,
) -> Optional[Path]:
    """Plot distribution of rewards across episodes for each policy.

    Args:
        episode_rewards: List of reward lists, one per policy
        policy_names: Optional names for each policy
        output_path: Output path for plot
        console: Optional Rich console

    Returns:
        Path to generated plot or None if failed
    """
    if console is None:
        console = Console()

    if not HAS_MATPLOTLIB:
        console.print("[yellow]⚠ matplotlib not available[/yellow]")
        return None

    _setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if policy_names is None:
        policy_names = [f"Policy {i+1}" for i in range(len(episode_rewards))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Histogram
    ax1 = axes[0]
    for idx, (rewards, name) in enumerate(zip(episode_rewards, policy_names)):
        if rewards:
            ax1.hist(rewards, bins=20, alpha=0.6, label=name, 
                    color=PALETTE_COGAMES[idx % len(PALETTE_COGAMES)], edgecolor="black")
    
    ax1.set_xlabel("Reward", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax1.set_title("Reward Distribution", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Right: Box plot
    ax2 = axes[1]
    non_empty_rewards = [r for r in episode_rewards if r]
    non_empty_names = [n for r, n in zip(episode_rewards, policy_names) if r]
    
    if non_empty_rewards:
        bp = ax2.boxplot(non_empty_rewards, tick_labels=non_empty_names, patch_artist=True)
        
        for idx, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(PALETTE_COGAMES[idx % len(PALETTE_COGAMES)])
            patch.set_alpha(0.7)
    
    ax2.set_ylabel("Reward", fontsize=12, fontweight="bold")
    ax2.set_title("Reward Box Plot", fontsize=14, fontweight="bold")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    console.print(f"[green]✓ Saved: {output_path}[/green]")
    return output_path


def daf_plot_cumulative_performance(
    performance_over_time: dict[str, list[float]],
    output_path: Path | str = "cumulative_performance.png",
    title: str = "Cumulative Performance Over Episodes",
    console: Optional[Console] = None,
) -> Optional[Path]:
    """Plot cumulative performance curves for multiple policies/runs.

    Args:
        performance_over_time: Dict of run_name -> list of per-episode rewards
        output_path: Output path for plot
        title: Plot title
        console: Optional Rich console

    Returns:
        Path to generated plot or None if failed
    """
    if console is None:
        console = Console()

    if not HAS_MATPLOTLIB:
        console.print("[yellow]⚠ matplotlib not available[/yellow]")
        return None

    _setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = PALETTE_COGAMES[:len(performance_over_time)]
    
    for idx, (name, rewards) in enumerate(performance_over_time.items()):
        if rewards:
            cumsum = np.cumsum(rewards)
            episodes = np.arange(1, len(cumsum) + 1)
            
            ax.plot(episodes, cumsum, linewidth=2, label=name, color=colors[idx])
            ax.fill_between(episodes, 0, cumsum, alpha=0.2, color=colors[idx])
    
    ax.set_xlabel("Episode", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cumulative Reward", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    console.print(f"[green]✓ Saved: {output_path}[/green]")
    return output_path


def daf_plot_action_frequency(
    action_counts: dict[str, dict[str, int]],
    output_path: Path | str = "action_frequency.png",
    title: str = "Action Frequency by Policy",
    console: Optional[Console] = None,
) -> Optional[Path]:
    """Plot action frequency distribution for each policy.

    Args:
        action_counts: Dict of policy_name -> {action_name: count}
        output_path: Output path for plot
        title: Plot title
        console: Optional Rich console

    Returns:
        Path to generated plot or None if failed
    """
    if console is None:
        console = Console()

    if not HAS_MATPLOTLIB:
        console.print("[yellow]⚠ matplotlib not available[/yellow]")
        return None

    if not action_counts:
        console.print("[yellow]No action data provided[/yellow]")
        return None

    _setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get all unique actions
    all_actions = set()
    for counts in action_counts.values():
        all_actions.update(counts.keys())
    actions = sorted(all_actions)
    policies = list(action_counts.keys())

    fig, ax = plt.subplots(figsize=(max(10, len(actions) * 0.8), 6))

    x = np.arange(len(actions))
    width = 0.8 / len(policies)

    for i, policy in enumerate(policies):
        counts = [action_counts[policy].get(a, 0) for a in actions]
        # Normalize to percentages
        total = sum(counts) if sum(counts) > 0 else 1
        percentages = [c / total * 100 for c in counts]
        
        offset = width * (i - len(policies) / 2 + 0.5)
        ax.bar(x + offset, percentages, width, label=policy,
               color=PALETTE_COGAMES[i % len(PALETTE_COGAMES)], alpha=0.8, edgecolor="black")

    ax.set_xlabel("Action", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency (%)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(actions, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    console.print(f"[green]✓ Saved: {output_path}[/green]")
    return output_path


def daf_plot_resource_collection_timeline(
    resource_data: dict[str, list[tuple[int, str, int]]],
    output_path: Path | str = "resource_timeline.png",
    title: str = "Resource Collection Timeline",
    console: Optional[Console] = None,
) -> Optional[Path]:
    """Plot resource collection over time for multiple policies.

    Args:
        resource_data: Dict of policy_name -> list of (step, resource_type, amount)
        output_path: Output path for plot
        title: Plot title
        console: Optional Rich console

    Returns:
        Path to generated plot or None if failed
    """
    if console is None:
        console = Console()

    if not HAS_MATPLOTLIB:
        console.print("[yellow]⚠ matplotlib not available[/yellow]")
        return None

    if not resource_data:
        console.print("[yellow]No resource data provided[/yellow]")
        return None

    _setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get all resource types
    all_resources = set()
    for events in resource_data.values():
        for _, resource_type, _ in events:
            all_resources.add(resource_type)
    resources = sorted(all_resources)
    
    # Create subplots for each resource type
    fig, axes = plt.subplots(len(resources), 1, figsize=(12, 3 * len(resources)), sharex=True)
    if len(resources) == 1:
        axes = [axes]

    resource_colors = {r: PALETTE_COGAMES[i % len(PALETTE_COGAMES)] for i, r in enumerate(resources)}
    
    for ax, resource in zip(axes, resources):
        for policy_idx, (policy_name, events) in enumerate(resource_data.items()):
            # Filter events for this resource
            resource_events = [(step, amt) for step, rtype, amt in events if rtype == resource]
            
            if resource_events:
                steps, amounts = zip(*resource_events)
                cumsum = np.cumsum(amounts)
                
                ax.plot(steps, cumsum, linewidth=2, label=policy_name, 
                       linestyle="-" if policy_idx == 0 else "--")
        
        ax.set_ylabel(resource, fontsize=11, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)
    
    axes[-1].set_xlabel("Step", fontsize=12, fontweight="bold")
    plt.suptitle(title, fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    console.print(f"[green]✓ Saved: {output_path}[/green]")
    return output_path


# ============================================================================
# Statistical Summary Dashboard
# ============================================================================

def daf_generate_summary_dashboard(
    sweep_result: Optional["SweepResult"] = None,
    comparison_report: Optional["ComparisonReport"] = None,
    additional_metrics: Optional[dict[str, Any]] = None,
    output_dir: Path | str = "dashboard",
    console: Optional[Console] = None,
) -> list[Path]:
    """Generate comprehensive summary dashboard with multiple visualizations.

    Args:
        sweep_result: Optional sweep results to visualize
        comparison_report: Optional comparison report to visualize
        additional_metrics: Optional dict of additional metrics to include
        output_dir: Output directory for all dashboard files
        console: Optional Rich console

    Returns:
        List of paths to generated files
    """
    if console is None:
        console = Console()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    console.print("\n[bold cyan]Generating Summary Dashboard[/bold cyan]\n")

    # Generate sweep visualizations
    if sweep_result and sweep_result.trials:
        console.print("[dim]Processing sweep results...[/dim]")
        
        sweep_files = daf_plot_sweep_results(sweep_result, output_dir / "sweeps", console=Console(quiet=True))
        generated_files.extend(sweep_files)
        
        # Try to generate heatmaps for first two hyperparameters
        hp_names = set()
        for trial in sweep_result.trials:
            hp_names.update(trial.hyperparameters.keys())
        hp_list = sorted(hp_names)
        
        if len(hp_list) >= 2:
            heatmap_path = daf_plot_sweep_heatmap(
                sweep_result, hp_list[0], hp_list[1],
                output_dir / "sweeps" / "param_heatmap.png",
                console=Console(quiet=True)
            )
            if heatmap_path:
                generated_files.append(heatmap_path)
        
        parallel_path = daf_plot_sweep_parallel_coordinates(
            sweep_result, output_dir / "sweeps" / "parallel_coords.png",
            console=Console(quiet=True)
        )
        if parallel_path:
            generated_files.append(parallel_path)

    # Generate comparison visualizations
    if comparison_report:
        console.print("[dim]Processing comparison results...[/dim]")
        
        comp_files = daf_plot_policy_comparison(comparison_report, output_dir / "comparisons", console=Console(quiet=True))
        generated_files.extend(comp_files)

    # Generate HTML summary
    html_path = _generate_dashboard_html(output_dir, sweep_result, comparison_report, additional_metrics)
    if html_path:
        generated_files.append(html_path)

    console.print(f"\n[bold green]✓ Dashboard generated with {len(generated_files)} files[/bold green]")
    console.print(f"[dim]Location: {output_dir}[/dim]\n")
    
    return generated_files


def _generate_dashboard_html(
    output_dir: Path,
    sweep_result: Optional["SweepResult"],
    comparison_report: Optional["ComparisonReport"],
    additional_metrics: Optional[dict[str, Any]],
) -> Optional[Path]:
    """Generate HTML dashboard summarizing all results."""
    html_path = output_dir / "dashboard.html"
    
    # Collect all images
    images = list(output_dir.rglob("*.png"))
    
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "  <meta charset='utf-8'>",
        "  <title>DAF Analysis Dashboard</title>",
        "  <style>",
        "    * { box-sizing: border-box; }",
        "    body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f5f6fa; }",
        "    .container { max-width: 1400px; margin: 0 auto; }",
        "    h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }",
        "    h2 { color: #34495e; margin-top: 30px; }",
        "    .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }",
        "    .stat-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; }",
        "    .stat-value { font-size: 2em; font-weight: bold; color: #3498db; }",
        "    .stat-label { color: #7f8c8d; margin-top: 5px; }",
        "    .image-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }",
        "    .image-card { background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }",
        "    .image-card img { width: 100%; height: auto; border-radius: 5px; }",
        "    .image-card h3 { margin: 10px 0 0 0; font-size: 0.9em; color: #34495e; }",
        "    .timestamp { color: #95a5a6; font-size: 0.9em; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <div class='container'>",
        f"    <h1>DAF Analysis Dashboard</h1>",
        f"    <p class='timestamp'>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
    ]
    
    # Stats cards
    html_parts.append("    <div class='stats-grid'>")
    
    if sweep_result and sweep_result.trials:
        best = sweep_result.get_best_trial()
        best_metric = f"{best.primary_metric:.4f}" if best else "N/A"
        html_parts.append(f"""
        <div class='stat-card'>
            <div class='stat-value'>{len(sweep_result.trials)}</div>
            <div class='stat-label'>Total Trials</div>
        </div>
        <div class='stat-card'>
            <div class='stat-value'>{best_metric}</div>
            <div class='stat-label'>Best {sweep_result.objective_metric}</div>
        </div>
        """)
    
    if comparison_report:
        html_parts.append(f"""
        <div class='stat-card'>
            <div class='stat-value'>{len(comparison_report.policies)}</div>
            <div class='stat-label'>Policies Compared</div>
        </div>
        <div class='stat-card'>
            <div class='stat-value'>{len(comparison_report.missions)}</div>
            <div class='stat-label'>Missions</div>
        </div>
        """)
    
    if additional_metrics:
        for key, value in list(additional_metrics.items())[:4]:
            val_str = f"{value:.4f}" if isinstance(value, float) else str(value)
            html_parts.append(f"""
            <div class='stat-card'>
                <div class='stat-value'>{val_str}</div>
                <div class='stat-label'>{key}</div>
            </div>
            """)
    
    html_parts.append("    </div>")
    
    # Images
    if images:
        html_parts.append("    <h2>Visualizations</h2>")
        html_parts.append("    <div class='image-grid'>")
        
        for img in sorted(images):
            rel_path = img.relative_to(output_dir)
            name = img.stem.replace("_", " ").title()
            html_parts.append(f"""
            <div class='image-card'>
                <img src='{rel_path}' alt='{name}'>
                <h3>{name}</h3>
            </div>
            """)
        
        html_parts.append("    </div>")
    
    html_parts.extend([
        "  </div>",
        "</body>",
        "</html>",
    ])
    
    with open(html_path, "w") as f:
        f.write("\n".join(html_parts))
    
    return html_path


# ============================================================================
# HTML Report Generation
# ============================================================================

def daf_export_comparison_html(
    comparison_report: "ComparisonReport",
    output_path: Path | str = "comparison_report.html",
    console: Optional[Console] = None,
) -> Path:
    """Export comparison report as interactive HTML.

    Args:
        comparison_report: Report from daf_compare_policies
        output_path: Output path for HTML file
        console: Optional Rich console for output

    Returns:
        Path to generated HTML file
    """
    if console is None:
        console = Console()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html_parts = []
    html_parts.append("<!DOCTYPE html>")
    html_parts.append("<html>")
    html_parts.append("<head>")
    html_parts.append("  <meta charset='utf-8'>")
    html_parts.append("  <title>Policy Comparison Report</title>")
    html_parts.append("  <style>")
    html_parts.append("    body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f8f9fa; }")
    html_parts.append("    .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 20px rgba(0,0,0,0.1); }")
    html_parts.append("    h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 15px; }")
    html_parts.append("    h2 { color: #34495e; margin-top: 30px; }")
    html_parts.append("    table { border-collapse: collapse; width: 100%; margin: 20px 0; }")
    html_parts.append("    th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }")
    html_parts.append("    th { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }")
    html_parts.append("    tr:nth-child(even) { background-color: #f8f9fa; }")
    html_parts.append("    tr:hover { background-color: #e8f4f8; }")
    html_parts.append("    .significant { font-weight: bold; color: #27ae60; }")
    html_parts.append("    .not-significant { color: #95a5a6; }")
    html_parts.append("    .winner { background: #d5f5e3; }")
    html_parts.append("    .stat-box { display: inline-block; padding: 10px 20px; margin: 5px; background: #ecf0f1; border-radius: 5px; }")
    html_parts.append("  </style>")
    html_parts.append("</head>")
    html_parts.append("<body>")
    html_parts.append("  <div class='container'>")

    html_parts.append(f"    <h1>Policy Comparison Report</h1>")
    html_parts.append(f"    <p>Generated: {comparison_report.timestamp.isoformat()}</p>")
    html_parts.append(f"    <div class='stat-box'><strong>Policies:</strong> {len(comparison_report.policies)}</div>")
    html_parts.append(f"    <div class='stat-box'><strong>Missions:</strong> {len(comparison_report.missions)}</div>")
    html_parts.append(f"    <div class='stat-box'><strong>Episodes/Mission:</strong> {comparison_report.episodes_per_mission}</div>")

    # Summary table
    html_parts.append("    <h2>Summary Statistics</h2>")
    html_parts.append("    <table>")
    html_parts.append("      <tr><th>Rank</th><th>Policy</th><th>Avg Reward</th><th>Std Dev</th></tr>")

    sorted_policies = sorted(
        comparison_report.policies,
        key=lambda p: comparison_report.policy_averages.get(p, 0.0),
        reverse=True,
    )

    for rank, policy in enumerate(sorted_policies, 1):
        avg = comparison_report.policy_averages.get(policy, 0.0)
        std = comparison_report.policy_std_devs.get(policy, 0.0)
        row_class = " class='winner'" if rank == 1 else ""
        html_parts.append(f"      <tr{row_class}><td>{rank}</td><td>{policy}</td><td>{avg:.4f}</td><td>{std:.4f}</td></tr>")

    html_parts.append("    </table>")

    # Pairwise comparisons
    if comparison_report.pairwise_comparisons:
        html_parts.append("    <h2>Pairwise Statistical Comparisons</h2>")
        html_parts.append("    <table>")
        html_parts.append("      <tr><th>Policy A</th><th>Policy B</th><th>A Reward</th><th>B Reward</th><th>P-value</th><th>Significant</th><th>Effect Size</th></tr>")

        for (policy_a, policy_b), comp in comparison_report.pairwise_comparisons.items():
            sig_class = "significant" if comp.is_significant else "not-significant"
            sig_text = "✓ Yes" if comp.is_significant else "✗ No"
            effect = getattr(comp, "effect_size", 0)

            html_parts.append(
                f"      <tr><td>{policy_a}</td><td>{policy_b}</td>"
                f"<td>{comp.avg_reward_a:.4f}</td><td>{comp.avg_reward_b:.4f}</td>"
                f"<td>{comp.p_value:.4f}</td><td class='{sig_class}'>{sig_text}</td>"
                f"<td>{effect:.3f}</td></tr>"
            )

        html_parts.append("    </table>")

    # Detailed metrics section
    detailed_metrics = getattr(comparison_report, 'policy_detailed_metrics', {})
    if detailed_metrics:
        html_parts.append("    <h2>Detailed Agent Metrics</h2>")
        
        # Get all metric names
        all_metrics = set()
        for policy_data in detailed_metrics.values():
            for mission_data in policy_data.values():
                all_metrics.update(mission_data.keys())
        
        # Group metrics by category
        metric_categories = {
            "Resources": [m for m in all_metrics if any(r in m for r in ["carbon", "silicon", "oxygen", "germanium"])],
            "Energy": [m for m in all_metrics if "energy" in m],
            "Actions": [m for m in all_metrics if "action" in m],
            "Inventory": [m for m in all_metrics if "inventory" in m],
            "Status": [m for m in all_metrics if "status" in m],
        }
        
        for category, metrics in metric_categories.items():
            if not metrics:
                continue
                
            html_parts.append(f"    <h3>{category}</h3>")
            html_parts.append("    <table>")
            header = "      <tr><th>Metric</th>"
            for policy in comparison_report.policies:
                header += f"<th>{policy}</th>"
            header += "</tr>"
            html_parts.append(header)
            
            for metric in sorted(metrics):
                row = f"      <tr><td>{metric}</td>"
                values = []
                for policy in comparison_report.policies:
                    policy_metrics = detailed_metrics.get(policy, {})
                    # Get first mission's metrics (or average across missions)
                    val = 0.0
                    count = 0
                    for mission_metrics in policy_metrics.values():
                        if metric in mission_metrics:
                            val += mission_metrics[metric]
                            count += 1
                    if count > 0:
                        val /= count
                    values.append(val)
                    row += f"<td>{val:.2f}</td>"
                
                # Highlight best value
                if values:
                    max_val = max(values)
                    min_val = min(values)
                    # For "failed" metrics, lower is better
                    is_lower_better = "failed" in metric or "lost" in metric
                    best_val = min_val if is_lower_better else max_val
                    
                    row = f"      <tr><td>{metric}</td>"
                    for policy, val in zip(comparison_report.policies, values):
                        style = " class='winner'" if val == best_val and best_val != 0 else ""
                        row += f"<td{style}>{val:.2f}</td>"
                
                row += "</tr>"
                html_parts.append(row)
            
            html_parts.append("    </table>")

    html_parts.append("  </div>")
    html_parts.append("</body>")
    html_parts.append("</html>")

    with open(output_path, "w") as f:
        f.write("\n".join(html_parts))

    console.print(f"[green]✓ HTML report saved: {output_path}[/green]\n")
    return output_path


def daf_generate_leaderboard(
    comparison_report: "ComparisonReport",
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

    sorted_policies = sorted(
        comparison_report.policies,
        key=lambda p: comparison_report.policy_averages.get(p, 0.0),
        reverse=True,
    )

    lines = []
    lines.append("| Rank | Policy | Avg Reward | Std Dev | Missions |")
    lines.append("|------|--------|-----------|---------|----------|")

    for rank, policy in enumerate(sorted_policies, 1):
        avg = comparison_report.policy_averages.get(policy, 0.0)
        std = comparison_report.policy_std_devs.get(policy, 0.0)
        missions = len(comparison_report.policy_mission_rewards.get(policy, {}))
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else ""
        lines.append(f"| {medal}{rank} | {policy} | {avg:.4f} | {std:.4f} | {missions} |")

    markdown_table = "\n".join(lines)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        leaderboard_data = {
            "generated": datetime.now().isoformat(),
            "leaderboard": [
                {
                    "rank": rank,
                    "policy": policy,
                    "avg_reward": comparison_report.policy_averages.get(policy, 0.0),
                    "std_dev": comparison_report.policy_std_devs.get(policy, 0.0),
                    "missions_evaluated": len(comparison_report.policy_mission_rewards.get(policy, {})),
                }
                for rank, policy in enumerate(sorted_policies, 1)
            ]
        }

        with open(output_path, "w") as f:
            json.dump(leaderboard_data, f, indent=2)

        console.print(f"[green]✓ Leaderboard saved: {output_path}[/green]")

    return markdown_table
