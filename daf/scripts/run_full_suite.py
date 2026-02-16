#!/usr/bin/env python3
"""Run full CoGames evaluation suite with comprehensive visualizations.

End-to-end pipeline: auth → environment → comparison → sweep → training
→ GIF visualization → scoring → dashboard.

This script executes real cogames evaluations (not unit tests) and generates
a complete visualization dashboard.

Usage:
    python daf/scripts/run_full_suite.py [OPTIONS]

    # Quick evaluation with 3 episodes
    python daf/scripts/run_full_suite.py --quick

    # Full sweep with custom policies
    python daf/scripts/run_full_suite.py --policies lstm cogames.policy.starter_agent.StarterPolicy

    # Custom missions
    python daf/scripts/run_full_suite.py --missions cogsguard_machina_1 cogsguard_arena

    # Include training stage
    python daf/scripts/run_full_suite.py --with-training --train-steps 1000
"""

from __future__ import annotations

# Pre-load torch to prevent SIGABRT from torch's C extension conflicting
# with signal handlers on macOS ARM + Python 3.13
try:
    import torch  # noqa: F401
except ImportError:
    pass

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table


console = Console()


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_POLICIES = ["cogames.policy.starter_agent.StarterPolicy"]
DEFAULT_MISSIONS = ["cogsguard_machina_1.basic"]  # Format: site.mission
SWEEP_SEARCH_SPACE = {
    "learning_rate": [0.0001, 0.001, 0.01],
    "hidden_size": [64, 128],
}


def get_output_dir() -> Path:
    """Get timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "daf_output" / "full_suite" / f"suite_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ============================================================================
# Phase 0: Authentication Check
# ============================================================================

def check_authentication(server_url: str = "https://cogames.ai") -> dict:
    """Check authentication status before running suite.

    Thin orchestration of cogames.auth.BaseCLIAuthenticator via DAF wrapper.
    """
    console.print("\n[bold cyan]═══ Phase 0: Authentication Check ═══[/bold cyan]\n")

    try:
        from daf.src.train.auth_integration import daf_check_auth_status

        status = daf_check_auth_status(server_url=server_url)

        if status.get("authenticated"):
            console.print(f"[green]✓ Authenticated as: {status.get('username', 'unknown')}[/green]")
        else:
            console.print("[yellow]⚠ Not authenticated — tournament submission will be unavailable[/yellow]")
            console.print("[dim]  Run: daf_login() to authenticate[/dim]")

        return {"status": "success", "authenticated": status.get("authenticated", False)}

    except Exception as e:
        console.print(f"[yellow]⚠ Auth check skipped: {e}[/yellow]")
        return {"status": "skipped", "authenticated": False, "reason": str(e)}


# ============================================================================
# Phase 1: Environment Validation
# ============================================================================

def validate_environment(missions: list[str]) -> bool:
    """Validate environment before running suite.

    Thin orchestration of cogames environment checks via DAF wrapper.
    Warnings about optional dependencies (CUDA, ray) do not block execution.
    Only critical errors (missing core packages, invalid missions) fail.
    """
    console.print("\n[bold cyan]═══ Phase 1: Environment Validation ═══[/bold cyan]\n")

    from daf.src.environment_checks import daf_check_environment

    result = daf_check_environment(missions=missions)

    # Check for critical errors vs. warnings
    critical_errors = [
        e for e in getattr(result, 'errors', [])
        if 'Not installed' in str(e) and 'Optional' not in str(e)
    ]

    if result.is_healthy():
        console.print("[green]✓ Environment validation passed[/green]")
        return True
    elif not critical_errors and hasattr(result, 'warnings'):
        # Only warnings (CUDA, ray, etc.) - proceed anyway
        console.print("[yellow]⚠ Environment has warnings (non-critical):[/yellow]")
        for w in result.warnings[:3]:  # Show first 3
            console.print(f"  [dim]{w}[/dim]")
        console.print("[green]✓ Proceeding with evaluation...[/green]")
        return True
    else:
        console.print(f"[red]✗ Critical environment issues: {result.errors}[/red]")
        return False


# ============================================================================
# Phase 2: Policy Comparison
# ============================================================================

def run_policy_comparison(
    policies: list[str],
    missions: list[str],
    episodes_per_mission: int,
    output_dir: Path,
) -> dict:
    """Run multi-policy comparison on missions.

    Thin orchestration of cogames.evaluate.evaluate() via daf_compare_policies.
    """
    console.print("\n[bold cyan]═══ Phase 2: Policy Comparison ═══[/bold cyan]\n")

    from daf.src.eval.comparison import daf_compare_policies, ComparisonReport
    from daf.src.viz import visualization
    from mettagrid.policy.policy import PolicySpec
    from cogames.cli.mission import get_mission

    # Create policy specs
    policy_specs = [PolicySpec(class_path=p) for p in policies]

    console.print(f"Policies: {', '.join(policies)}")
    console.print(f"Missions: {', '.join(missions)}")
    console.print(f"Episodes per mission: {episodes_per_mission}\n")

    # Load missions using cogames.cli.mission.get_mission directly
    missions_and_configs = []
    try:
        for mission_name in missions:
            name, env_cfg, _ = get_mission(mission_name)
            missions_and_configs.append((name, env_cfg))
    except Exception as e:
        console.print(f"[red]Error loading missions: {e}[/red]")
        return {"status": "failed", "error": str(e)}

    # Run comparison (thin orchestration of cogames.evaluate.evaluate)
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Running policy comparison...", total=None)

            report = daf_compare_policies(
                policies=policy_specs,
                missions=missions_and_configs,
                episodes_per_mission=episodes_per_mission,
                console=Console(quiet=True),  # Suppress inner logging
            )

            progress.update(task, completed=True)

        # Save results
        comparison_dir = output_dir / "comparisons"
        comparison_dir.mkdir(parents=True, exist_ok=True)

        report.save_json(comparison_dir / "comparison_results.json")

        # Generate visualizations
        console.print("\n[dim]Generating comparison visualizations...[/dim]")

        visualization.daf_plot_policy_comparison(
            report,
            output_dir=comparison_dir,
            console=Console(quiet=True),
        )

        # Generate detailed metrics visualizations (resources, actions, etc.)
        visualization.daf_plot_detailed_metrics_comparison(
            report,
            output_dir=comparison_dir,
            console=Console(quiet=True),
        )

        visualization.daf_export_comparison_html(
            report,
            output_path=comparison_dir / "report.html",
            console=Console(quiet=True),
        )

        visualization.daf_generate_leaderboard(
            report,
            output_path=comparison_dir / "leaderboard.json",
            console=Console(quiet=True),
        )

        # Generate new statistical plots
        console.print("[dim]Generating statistical analysis plots...[/dim]")
        
        # Distribution plots
        visualization.daf_plot_metric_distributions(
            report, 
            metric_name="performance_score", 
            output_dir=comparison_dir,
            console=Console(quiet=True)
        )
        visualization.daf_plot_metric_distributions(
            report, 
            metric_name="energy.gained", 
            output_dir=comparison_dir,
            console=Console(quiet=True)
        )

        # Correlation plots for each policy
        for policy_name in report.policies:
            visualization.daf_plot_metric_correlations(
                report,
                policy_name,
                output_dir=comparison_dir,
                console=Console(quiet=True)
            )

        console.print(f"[green]✓ Comparison complete. Results in: {comparison_dir}[/green]")

        return {
            "status": "success",
            "output_dir": str(comparison_dir),
            "policies": policies,
            "missions": missions,
            "summary": report.summary_statistics,
        }

    except Exception as e:
        console.print(f"[red]✗ Comparison failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


# ============================================================================
# Phase 3: Hyperparameter Sweep
# ============================================================================

def run_hyperparameter_sweep(
    policy: str,
    missions: list[str],
    search_space: dict,
    episodes_per_trial: int,
    output_dir: Path,
) -> dict:
    """Run hyperparameter sweep on a policy.

    Thin orchestration of cogames.evaluate.evaluate() via daf_launch_sweep.
    """
    console.print("\n[bold cyan]═══ Phase 3: Hyperparameter Sweep ═══[/bold cyan]\n")

    from daf.src.eval.sweeps import daf_launch_sweep, SweepResult
    from daf.src.core.config import DAFSweepConfig
    from daf.src.viz import visualization

    console.print(f"Policy: {policy}")
    console.print(f"Missions: {', '.join(missions)}")
    console.print(f"Search space: {search_space}")
    console.print(f"Episodes per trial: {episodes_per_trial}\n")

    # Create sweep config
    sweep_config = DAFSweepConfig(
        name=f"{policy}_sweep",
        policy_class_path=policy,
        missions=missions,
        strategy="grid",
        search_space=search_space,
        episodes_per_trial=episodes_per_trial,
        objective_metric="total_reward",
        optimize_direction="maximize",
        num_trials=9,  # Will be overridden by grid search
        seed=42,
    )

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Running hyperparameter sweep...", total=None)

            sweep_result = daf_launch_sweep(
                sweep_config,
                console=Console(quiet=True),
            )

            progress.update(task, completed=True)

        # Save results
        sweep_dir = output_dir / "sweeps" / policy
        sweep_dir.mkdir(parents=True, exist_ok=True)

        sweep_result.save_json(sweep_dir / "sweep_results.json")

        # Generate visualizations
        console.print("\n[dim]Generating sweep visualizations...[/dim]")

        visualization.daf_plot_sweep_results(
            sweep_result,
            output_dir=sweep_dir,
            console=Console(quiet=True),
        )

        # Generate heatmap if we have 2+ hyperparameters
        hp_names = list(search_space.keys())
        if len(hp_names) >= 2:
            visualization.daf_plot_sweep_heatmap(
                sweep_result,
                hp_x=hp_names[0],
                hp_y=hp_names[1],
                output_path=sweep_dir / "heatmap.png",
                console=Console(quiet=True),
            )

        # Parallel coordinates
        visualization.daf_plot_sweep_parallel_coordinates(
            sweep_result,
            output_path=sweep_dir / "parallel.png",
            console=Console(quiet=True),
        )

        best = sweep_result.get_best_trial()

        console.print(f"[green]✓ Sweep complete. Results in: {sweep_dir}[/green]")
        if best:
            console.print(f"[green]  Best config: {best.hyperparameters}[/green]")
            console.print(f"[green]  Best metric: {best.primary_metric:.4f}[/green]")

        return {
            "status": "success",
            "output_dir": str(sweep_dir),
            "num_trials": len(sweep_result.trials),
            "best_config": best.hyperparameters if best else None,
            "best_metric": best.primary_metric if best else None,
        }

    except Exception as e:
        console.print(f"[red]✗ Sweep failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


# ============================================================================
# Phase 4: Training (Optional)
# ============================================================================

def run_training(
    policy: str,
    missions: list[str],
    train_steps: int,
    output_dir: Path,
) -> dict:
    """Run distributed training on a policy.

    Thin orchestration of cogames.train.train() via daf_launch_distributed_training.
    """
    console.print("\n[bold cyan]═══ Phase 4: Distributed Training ═══[/bold cyan]\n")

    try:
        from daf.src.train.distributed_training import daf_launch_distributed_training
        from cogames.cli.mission import get_mission

        console.print(f"Policy: {policy}")
        console.print(f"Missions: {', '.join(missions)}")
        console.print(f"Training steps: {train_steps}\n")

        # Load first mission for training
        name, env_cfg, _ = get_mission(missions[0])

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Running distributed training...", total=None)

            # Detect best available device
            import torch as _torch
            if _torch.backends.mps.is_available():
                _device = _torch.device("mps")
            elif _torch.cuda.is_available():
                _device = _torch.device("cuda")
            else:
                _device = _torch.device("cpu")

            result = daf_launch_distributed_training(
                env_cfg=env_cfg,
                policy_class_path=policy,
                device=_device,
                num_steps=train_steps,
                console=Console(quiet=True),
            )

            progress.update(task, completed=True)

        # Save training results
        train_dir = output_dir / "training"
        train_dir.mkdir(parents=True, exist_ok=True)

        train_summary = {
            "policy": policy,
            "mission": missions[0],
            "steps": train_steps,
            "status": result.status if hasattr(result, 'status') else "completed",
        }

        with open(train_dir / "training_results.json", "w") as f:
            json.dump(train_summary, f, indent=2, default=str)

        console.print(f"[green]✓ Training complete. Results in: {train_dir}[/green]")

        return {
            "status": "success",
            "output_dir": str(train_dir),
            "steps_completed": train_steps,
        }

    except Exception as e:
        console.print(f"[red]✗ Training failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        # Dump error to file for debugging
        try:
            with open("daf_output/training_error.log", "w") as f:
                f.write(f"Error: {e}\n")
                traceback.print_exc(file=f)
        except Exception:
            pass
        return {"status": "failed", "error": str(e)}


# ============================================================================
# Phase 5: GIF Visualization
# ============================================================================

def generate_gif_visualizations(
    output_dir: Path,
    missions: list[str],
    policy: str,
    fps: int = 10,
    step_interval: int = 5,
) -> dict:
    """Generate animated GIF visualizations from simulation replays.

    Thin orchestration of cogames.evaluate.evaluate() for replay capture
    + daf gif_generator for rendering.
    """
    console.print("\n[bold cyan]═══ Phase 5: GIF Visualization ═══[/bold cyan]\n")

    try:
        from daf.src.viz.gif_generator import generate_gif_from_replay
        from cogames import evaluate as evaluate_module
        from cogames.cli.mission import get_mission
        from mettagrid.policy.policy import PolicySpec

        gif_dir = output_dir / "gifs"
        replay_dir = output_dir / "replays"
        gif_dir.mkdir(parents=True, exist_ok=True)
        replay_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"[dim]Generating replays for: {policy}[/dim]")
        console.print(f"[dim]Missions: {', '.join(missions)}[/dim]\n")

        # Run simulation to generate replays (real cogames.evaluate)
        for mission_name in missions[:1]:  # Only first mission for GIF to save time
            try:
                name, env_cfg, _ = get_mission(mission_name)
                policy_spec = PolicySpec(class_path=policy)

                # Run evaluation with replay saving (real cogames method)
                evaluate_module.evaluate(
                    console=Console(quiet=True),
                    missions=[(name, env_cfg)],
                    policy_specs=[policy_spec],
                    proportions=[1.0],
                    episodes=1,
                    action_timeout_ms=250,
                    seed=42,
                    save_replay=str(replay_dir),
                )
            except Exception as e:
                console.print(f"[yellow]Warning: Could not generate replay for {mission_name}: {e}[/yellow]")

        # Generate GIFs from replays
        replay_files = list(replay_dir.glob("*.json*"))
        gif_files = []

        for replay_file in replay_files:
            try:
                gif_name = replay_file.stem.replace(".json", "") + ".gif"
                gif_path = gif_dir / gif_name

                console.print(f"[dim]Rendering GIF: {gif_name}...[/dim]")
                generate_gif_from_replay(
                    replay_file,
                    gif_path,
                    fps=fps,
                    step_interval=step_interval,
                )
                gif_files.append(str(gif_path))
                console.print(f"  [green]✓ {gif_name}[/green]")
            except Exception as e:
                console.print(f"  [red]✗ Failed: {e}[/red]")

        console.print(f"\n[green]✓ Generated {len(gif_files)} GIF(s)[/green]")

        return {
            "status": "success",
            "output_dir": str(gif_dir),
            "gifs_generated": len(gif_files),
            "gif_files": gif_files,
        }

    except Exception as e:
        console.print(f"[red]✗ GIF generation failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


# ============================================================================
# Phase 6: Scoring Summary
# ============================================================================

def generate_scoring_summary(
    comparison_result: dict,
    output_dir: Path,
) -> dict:
    """Generate scoring summary using metta_alo scoring via DAF wrapper.

    Thin orchestration of metta_alo.scoring via daf_scoring_summary.
    """
    console.print("\n[bold cyan]═══ Phase 6: Scoring Summary ═══[/bold cyan]\n")

    try:
        from daf.src.eval.scoring_analysis import daf_scoring_summary

        if comparison_result.get("status") != "success":
            console.print("[yellow]⚠ Skipping scoring — comparison did not succeed[/yellow]")
            return {"status": "skipped", "reason": "comparison_failed"}

        summary = comparison_result.get("summary", {})
        if not summary:
            console.print("[yellow]⚠ No comparison summary available for scoring[/yellow]")
            return {"status": "skipped", "reason": "no_summary_data"}

        # Build reward data from comparison summary
        reward_data = {}
        for policy_name, stats in summary.items():
            if isinstance(stats, dict) and "avg_reward" in stats:
                reward_data[policy_name] = [stats["avg_reward"]]

        if reward_data:
            # Determine pool scores for VOR (use first policy or specific baseline)
            pool_scores = list(reward_data.values())[0]
            for p_name in ["baseline", "StarterPolicy", "cogames.policy.starter_agent.StarterPolicy"]:
                if p_name in reward_data:
                    pool_scores = reward_data[p_name]
                    break

            scoring = daf_scoring_summary(reward_data, pool_scores)
            scoring_dir = output_dir / "scoring"
            scoring_dir.mkdir(parents=True, exist_ok=True)

            with open(scoring_dir / "scoring_summary.json", "w") as f:
                json.dump(scoring, f, indent=2, default=str)

            console.print(f"[green]✓ Scoring summary generated: {scoring_dir}[/green]")
            return {"status": "success", "output_dir": str(scoring_dir), "summary": scoring}

        console.print("[yellow]⚠ No reward data available for scoring[/yellow]")
        return {"status": "skipped", "reason": "no_reward_data"}

    except Exception as e:
        console.print(f"[yellow]⚠ Scoring summary skipped: {e}[/yellow]")
        return {"status": "skipped", "reason": str(e)}


# ============================================================================
# Phase 7: Dashboard Generation
# ============================================================================

def generate_dashboard(
    output_dir: Path,
    comparison_result: dict,
    sweep_results: list[dict],
) -> dict:
    """Generate comprehensive HTML dashboard.

    Thin orchestration of daf visualization suite.
    """
    console.print("\n[bold cyan]═══ Phase 7: Dashboard Generation ═══[/bold cyan]\n")

    from daf.src.viz import visualization

    dashboard_dir = output_dir / "dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Collect all visualization data
        sweep_result = None
        comparison_report = None

        # Try to load sweep result if available
        if sweep_results and sweep_results[0].get("status") == "success":
            sweep_path = Path(sweep_results[0]["output_dir"]) / "sweep_results.json"
            if sweep_path.exists():
                from daf.src.eval.sweeps import SweepResult, SweepTrialResult
                with open(sweep_path) as f:
                    data = json.load(f)

                sweep_result = SweepResult(
                    sweep_name=data["sweep_name"],
                    objective_metric=data["objective_metric"],
                    optimize_direction=data["optimize_direction"],
                )
                for trial_data in data.get("trials", []):
                    trial = SweepTrialResult(
                        trial_id=trial_data["trial_id"],
                        hyperparameters=trial_data["hyperparameters"],
                        primary_metric=trial_data["primary_metric"],
                        all_metrics=trial_data["all_metrics"],
                        mission_results=trial_data["mission_results"],
                        success=trial_data["success"],
                    )
                    sweep_result.add_trial(trial)

        # Try to load comparison report if available
        if comparison_result.get("status") == "success":
            comparison_path = Path(comparison_result["output_dir"]) / "comparison_results.json"
            if comparison_path.exists():
                from daf.src.eval.comparison import ComparisonReport
                with open(comparison_path) as f:
                    data = json.load(f)

                comparison_report = ComparisonReport(
                    policies=data["policies"],
                    missions=data["missions"],
                    episodes_per_mission=data["episodes_per_mission"],
                )
                for policy, stats in data["summary_statistics"].items():
                    comparison_report.policy_averages[policy] = stats["avg_reward"]
                    comparison_report.policy_std_devs[policy] = stats["std_dev"]

        # Generate comprehensive dashboard
        files = visualization.daf_generate_summary_dashboard(
            sweep_result=sweep_result,
            comparison_report=comparison_report,
            output_dir=dashboard_dir,
            console=Console(quiet=True),
        )

        console.print(f"[green]✓ Dashboard generated: {dashboard_dir / 'dashboard.html'}[/green]")

        return {
            "status": "success",
            "output_dir": str(dashboard_dir),
            "files_generated": len(files),
        }

    except Exception as e:
        console.print(f"[red]✗ Dashboard generation failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


# ============================================================================
# Summary Report
# ============================================================================

def generate_summary_report(
    output_dir: Path,
    start_time: float,
    auth_result: dict,
    comparison_result: dict,
    sweep_results: list[dict],
    training_result: dict,
    gif_result: dict,
    scoring_result: dict,
    dashboard_result: dict,
) -> None:
    """Generate final summary report."""
    console.print("\n[bold cyan]═══ Suite Summary ═══[/bold cyan]\n")

    elapsed = time.time() - start_time

    # Summary table
    table = Table(title="Full Suite Results", show_header=True, header_style="bold magenta")
    table.add_column("Phase", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details")

    # Auth row
    auth_status = "✓ Authenticated" if auth_result.get("authenticated") else "⚠ Skipped"
    table.add_row("Authentication", auth_status, "")

    # Comparison row
    comp_status = "✓ Passed" if comparison_result.get("status") == "success" else "✗ Failed"
    comp_details = f"{len(comparison_result.get('policies', []))} policies"
    table.add_row("Policy Comparison", comp_status, comp_details)

    # Sweep rows
    for sweep in sweep_results:
        sweep_status = "✓ Passed" if sweep.get("status") == "success" else "✗ Failed"
        sweep_details = f"{sweep.get('num_trials', 0)} trials"
        if sweep.get("best_metric"):
            sweep_details += f", best: {sweep['best_metric']:.4f}"
        table.add_row("Hyperparameter Sweep", sweep_status, sweep_details)

    # Training row
    if training_result.get("status") != "skipped":
        train_status = "✓ Passed" if training_result.get("status") == "success" else "✗ Failed"
        train_details = f"{training_result.get('steps_completed', 0)} steps"
        table.add_row("Training", train_status, train_details)

    # GIF row
    if gif_result.get("status") != "skipped":
        gif_status = "✓ Passed" if gif_result.get("status") == "success" else "✗ Failed"
        gif_details = f"{gif_result.get('gifs_generated', 0)} GIFs"
        table.add_row("GIF Visualization", gif_status, gif_details)

    # Scoring row
    if scoring_result.get("status") != "skipped":
        score_status = "✓ Passed" if scoring_result.get("status") == "success" else "✗ Failed"
        table.add_row("Scoring Summary", score_status, "")

    # Dashboard row
    dash_status = "✓ Passed" if dashboard_result.get("status") == "success" else "✗ Failed"
    dash_details = f"{dashboard_result.get('files_generated', 0)} files"
    table.add_row("Dashboard", dash_status, dash_details)

    console.print(table)

    # Timing
    console.print(f"\n[dim]Total execution time: {elapsed:.1f}s[/dim]")
    console.print(f"[dim]Output directory: {output_dir}[/dim]\n")

    # Save summary JSON
    summary = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "output_dir": str(output_dir),
        "auth": auth_result,
        "comparison": comparison_result,
        "sweeps": sweep_results,
        "training": training_result,
        "gif": gif_result,
        "scoring": scoring_result,
        "dashboard": dashboard_result,
    }

    with open(output_dir / "SUITE_SUMMARY.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ASCII summary report
    with open(output_dir / "SUITE_SUMMARY.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("CoGames Full Suite Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Duration: {elapsed:.1f}s\n\n")

        f.write("RESULTS:\n")
        f.write(f"  Auth:         {auth_result.get('status', 'N/A')}\n")
        f.write(f"  Comparison:   {comparison_result.get('status', 'N/A')}\n")
        for i, sweep in enumerate(sweep_results):
            f.write(f"  Sweep {i+1}:      {sweep.get('status', 'N/A')}\n")
        f.write(f"  Training:     {training_result.get('status', 'N/A')}\n")
        f.write(f"  GIF:          {gif_result.get('status', 'N/A')}\n")
        f.write(f"  Scoring:      {scoring_result.get('status', 'N/A')}\n")
        f.write(f"  Dashboard:    {dashboard_result.get('status', 'N/A')}\n\n")

        f.write("OUTPUT FILES:\n")
        f.write(f"  {output_dir}/comparisons/report.html\n")
        f.write(f"  {output_dir}/sweeps/*/sweep_progress.png\n")
        if training_result.get("status") == "success":
            f.write(f"  {output_dir}/training/training_results.json\n")
        if gif_result.get("status") == "success":
            f.write(f"  {output_dir}/gifs/*.gif\n")
        if scoring_result.get("status") == "success":
            f.write(f"  {output_dir}/scoring/scoring_summary.json\n")
        f.write(f"  {output_dir}/dashboard/dashboard.html\n")

    # Final message
    console.print(Panel(
        f"[bold green]Suite Complete![/bold green]\n\n"
        f"Open the dashboard in your browser:\n"
        f"[cyan]file://{output_dir}/dashboard/dashboard.html[/cyan]\n\n"
        f"Or view comparison report:\n"
        f"[cyan]file://{output_dir}/comparisons/report.html[/cyan]",
        title="✨ CoGames Full Suite",
        border_style="green",
    ))


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run full CoGames evaluation suite with visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick run (3 episodes)
  python daf/scripts/run_full_suite.py --quick

  # Standard run with default policies
  python daf/scripts/run_full_suite.py

  # Custom policies and CogsGuard missions
  python daf/scripts/run_full_suite.py --policies lstm baseline --missions cogsguard_machina_1

  # Skip sweep phase
  python daf/scripts/run_full_suite.py --no-sweep

  # Full end-to-end with training
  python daf/scripts/run_full_suite.py --with-training --train-steps 5000
        """
    )

    parser.add_argument(
        "--policies",
        nargs="+",
        default=DEFAULT_POLICIES,
        help=f"Policies to evaluate (default: {DEFAULT_POLICIES})",
    )
    parser.add_argument(
        "--missions",
        nargs="+",
        default=DEFAULT_MISSIONS,
        help=f"Missions to run on (default: {DEFAULT_MISSIONS})",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Episodes per mission for comparison (default: 5)",
    )
    parser.add_argument(
        "--sweep-episodes",
        type=int,
        default=3,
        help="Episodes per trial for sweep (default: 3)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 3 episodes, minimal sweep",
    )
    parser.add_argument(
        "--no-sweep",
        action="store_true",
        help="Skip hyperparameter sweep phase",
    )
    parser.add_argument(
        "--sweep-policy",
        default="baseline",
        help="Policy to use for hyperparameter sweep (default: baseline)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Custom output directory (default: timestamped)",
    )
    parser.add_argument(
        "--with-gif",
        action="store_true",
        help="Generate animated GIF visualizations from replays",
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=10,
        help="GIF frames per second (default: 10)",
    )
    parser.add_argument(
        "--gif-step-interval",
        type=int,
        default=5,
        help="Render every Nth step for GIF (default: 5, reduces file size)",
    )
    parser.add_argument(
        "--with-training",
        action="store_true",
        help="Include distributed training stage",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=5000,
        help="Number of training steps (default: 5000)",
    )
    parser.add_argument(
        "--train-policy",
        default="baseline",
        help="Policy to train (default: baseline)",
    )
    parser.add_argument(
        "--server-url",
        default="https://cogames.ai",
        help="CoGames server URL for auth check (default: https://cogames.ai)",
    )

    args = parser.parse_args()

    # Quick mode adjustments
    if args.quick:
        args.episodes = 3
        args.sweep_episodes = 1

    # Setup output directory
    output_dir = args.output_dir or get_output_dir()

    # Banner
    console.print(Panel(
        "[bold cyan]CoGames Full Evaluation Suite[/bold cyan]\n\n"
        f"Policies: {', '.join(args.policies)}\n"
        f"Missions: {', '.join(args.missions)}\n"
        f"Episodes: {args.episodes}\n"
        f"Training: {'Yes' if args.with_training else 'No'}\n"
        f"Output: {output_dir}",
        title="🎮 DAF Full Suite v2.2",
        border_style="cyan",
    ))

    start_time = time.time()

    # Phase 0: Authentication check
    auth_result = check_authentication(server_url=args.server_url)

    # Phase 1: Environment validation
    if not validate_environment(args.missions):
        console.print("[red]Environment validation failed. Exiting.[/red]")
        return 1

    # Phase 2: Policy comparison
    comparison_result = run_policy_comparison(
        policies=args.policies,
        missions=args.missions,
        episodes_per_mission=args.episodes,
        output_dir=output_dir,
    )

    # Phase 3: Hyperparameter sweep (optional)
    sweep_results = []
    if not args.no_sweep:
        sweep_result = run_hyperparameter_sweep(
            policy=args.sweep_policy,
            missions=args.missions,
            search_space=SWEEP_SEARCH_SPACE,
            episodes_per_trial=args.sweep_episodes,
            output_dir=output_dir,
        )
        sweep_results.append(sweep_result)

    # Phase 4: Distributed training (optional)
    training_result = {"status": "skipped"}
    if args.with_training:
        training_result = run_training(
            policy=args.train_policy,
            missions=args.missions,
            train_steps=args.train_steps,
            output_dir=output_dir,
        )

    # Phase 5: GIF visualization (optional)
    gif_result = {"status": "skipped"}
    if args.with_gif:
        gif_result = generate_gif_visualizations(
            output_dir=output_dir,
            missions=args.missions,
            policy=args.policies[0] if args.policies else "baseline",
            fps=args.gif_fps,
            step_interval=args.gif_step_interval,
        )

    # Phase 6: Scoring summary
    scoring_result = generate_scoring_summary(
        comparison_result=comparison_result,
        output_dir=output_dir,
    )

    # Phase 7: Dashboard generation
    dashboard_result = generate_dashboard(
        output_dir=output_dir,
        comparison_result=comparison_result,
        sweep_results=sweep_results,
    )

    # Generate summary
    generate_summary_report(
        output_dir=output_dir,
        start_time=start_time,
        auth_result=auth_result,
        comparison_result=comparison_result,
        sweep_results=sweep_results,
        training_result=training_result,
        gif_result=gif_result,
        scoring_result=scoring_result,
        dashboard_result=dashboard_result,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
