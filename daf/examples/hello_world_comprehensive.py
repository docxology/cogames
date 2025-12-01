#!/usr/bin/env python3
"""Comprehensive Hello World Policy Development & Testing Suite.

This script demonstrates the complete workflow for:
1. Creating custom policies (explorer, greedy collector)
2. Running evaluations with real cogames simulations
3. Hyperparameter sweeps with grid search
4. Policy comparison with statistical analysis
5. Full visualization and reporting
6. MettaScope replay generation

Usage:
    # Full comprehensive suite
    python daf/examples/hello_world_comprehensive.py

    # Quick mode (fewer episodes)
    python daf/examples/hello_world_comprehensive.py --quick

    # Generate replays for MettaScope
    python daf/examples/hello_world_comprehensive.py --with-replays
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table


console = Console()


# ============================================================================
# Custom Policy Implementations
# ============================================================================

@dataclass
class ExplorerHyperparameters:
    """Hyperparameters for the Explorer policy."""
    exploration_rate: float = 0.3      # Probability of random exploration
    memory_decay: float = 0.95         # How quickly visited locations fade
    prefer_unexplored: bool = True     # Bias toward unexplored areas
    energy_threshold: float = 0.2      # When to prioritize recharging


@dataclass 
class GreedyCollectorHyperparameters:
    """Hyperparameters for the Greedy Collector policy."""
    greed_factor: float = 0.8          # How strongly to prefer high-value targets
    lookahead_steps: int = 5           # How far ahead to plan
    risk_tolerance: float = 0.5        # Willingness to take risky paths
    batch_size: int = 3                # Resources to collect before returning


def create_policy_from_hyperparams(policy_type: str, hyperparams: dict) -> str:
    """Create policy configuration string from hyperparameters.
    
    For demonstration, we use the baseline policy as the backend,
    but in practice you would implement custom logic.
    """
    # In a real implementation, hyperparams would affect policy behavior
    # For now, we return the baseline which is deterministic
    return "baseline"


# ============================================================================
# Mission Configuration
# ============================================================================

HELLO_WORLD_MISSIONS = [
    "hello_world.hello_world_unclip",
    "hello_world.easy_hearts",
    "hello_world.open_world",
]

# Different difficulty levels for testing
MISSION_CONFIGS = {
    "easy": ["hello_world.easy_hearts"],
    "medium": ["hello_world.hello_world_unclip", "hello_world.easy_hearts"],
    "hard": ["hello_world.open_world", "hello_world.oxygen_bottleneck"],
    "comprehensive": HELLO_WORLD_MISSIONS,
}

# Hyperparameter search spaces
EXPLORER_SEARCH_SPACE = {
    "exploration_rate": [0.1, 0.3, 0.5],
    "energy_threshold": [0.1, 0.2, 0.3],
}

GREEDY_SEARCH_SPACE = {
    "greed_factor": [0.5, 0.7, 0.9],
    "lookahead_steps": [3, 5, 7],
}


# ============================================================================
# Workflow Functions
# ============================================================================

def get_output_dir(name: str = "hello_world_comprehensive") -> Path:
    """Get timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "daf_output" / "hello_world" / f"{name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_policy_evaluation(
    policies: list[str],
    missions: list[str],
    episodes: int,
    output_dir: Path,
    with_replays: bool = False,
) -> dict:
    """Run policy evaluation on missions."""
    console.print("\n[bold cyan]═══ Policy Evaluation ═══[/bold cyan]\n")
    
    from daf.src.comparison import daf_compare_policies, ComparisonReport
    from daf.src import visualization
    from mettagrid.policy.policy import PolicySpec
    from cogames.cli.mission import get_mission
    
    # Create policy specs
    policy_specs = [PolicySpec(class_path=p) for p in policies]
    
    console.print(f"[dim]Policies: {', '.join(policies)}[/dim]")
    console.print(f"[dim]Missions: {', '.join(missions)}[/dim]")
    console.print(f"[dim]Episodes: {episodes}[/dim]\n")
    
    # Load missions
    missions_and_configs = []
    for mission_name in missions:
        try:
            name, env_cfg, _ = get_mission(mission_name)
            missions_and_configs.append((name, env_cfg))
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load {mission_name}: {e}[/yellow]")
    
    if not missions_and_configs:
        return {"status": "failed", "error": "No missions loaded"}
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Running evaluations...", total=None)
            
            report = daf_compare_policies(
                policies=policy_specs,
                missions=missions_and_configs,
                episodes_per_mission=episodes,
                console=Console(quiet=True),
            )
            
            progress.update(task, completed=True)
        
        # Save results
        eval_dir = output_dir / "evaluations"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        report.save_json(eval_dir / "evaluation_results.json")
        
        # Generate visualizations
        console.print("[dim]Generating visualizations...[/dim]")
        
        visualization.daf_plot_policy_comparison(
            report,
            output_dir=eval_dir,
            console=Console(quiet=True),
        )
        
        visualization.daf_export_comparison_html(
            report,
            output_path=eval_dir / "report.html",
            console=Console(quiet=True),
        )
        
        visualization.daf_generate_leaderboard(
            report,
            output_path=eval_dir / "leaderboard.json",
            console=Console(quiet=True),
        )
        
        # Print summary table
        print_evaluation_summary(report)
        
        return {
            "status": "success",
            "output_dir": str(eval_dir),
            "summary": report.summary_statistics,
        }
        
    except Exception as e:
        console.print(f"[red]Evaluation failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


def run_hyperparameter_sweep(
    policy: str,
    missions: list[str],
    search_space: dict,
    episodes_per_trial: int,
    output_dir: Path,
) -> dict:
    """Run hyperparameter sweep with visualization."""
    console.print("\n[bold cyan]═══ Hyperparameter Sweep ═══[/bold cyan]\n")
    
    from daf.src.sweeps import daf_launch_sweep, SweepResult
    from daf.src.config import DAFSweepConfig
    from daf.src import visualization
    
    console.print(f"[dim]Policy: {policy}[/dim]")
    console.print(f"[dim]Missions: {', '.join(missions)}[/dim]")
    console.print(f"[dim]Search space: {search_space}[/dim]")
    console.print(f"[dim]Episodes per trial: {episodes_per_trial}[/dim]\n")
    
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
        num_trials=9,
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
            task = progress.add_task("Running sweep...", total=None)
            
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
        console.print("[dim]Generating sweep visualizations...[/dim]")
        
        visualization.daf_plot_sweep_results(
            sweep_result,
            output_dir=sweep_dir,
            console=Console(quiet=True),
        )
        
        # Generate heatmap if 2+ hyperparameters
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
            output_path=sweep_dir / "parallel_coords.png",
            console=Console(quiet=True),
        )
        
        best = sweep_result.get_best_trial()
        
        # Print sweep summary
        print_sweep_summary(sweep_result)
        
        return {
            "status": "success",
            "output_dir": str(sweep_dir),
            "num_trials": len(sweep_result.trials),
            "best_config": best.hyperparameters if best else None,
            "best_metric": best.primary_metric if best else None,
        }
        
    except Exception as e:
        console.print(f"[red]Sweep failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


def run_with_replays(
    policy: str,
    mission: str,
    episodes: int,
    output_dir: Path,
) -> dict:
    """Run evaluation and save replays for MettaScope visualization."""
    console.print("\n[bold cyan]═══ MettaScope Replay Generation ═══[/bold cyan]\n")
    
    from cogames import evaluate as evaluate_module
    from mettagrid.policy.policy import PolicySpec
    from cogames.cli.mission import get_mission
    
    console.print(f"[dim]Policy: {policy}[/dim]")
    console.print(f"[dim]Mission: {mission}[/dim]")
    console.print(f"[dim]Episodes: {episodes}[/dim]\n")
    
    # Create replay directory
    replay_dir = output_dir / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load mission
        name, env_cfg, _ = get_mission(mission)
        
        policy_spec = PolicySpec(class_path=policy)
        
        console.print("[dim]Running simulation with replay capture...[/dim]")
        
        # save_replay is passed directly to evaluate, not as a config field
        summaries = evaluate_module.evaluate(
            console=Console(quiet=True),
            missions=[(name, env_cfg)],
            policy_specs=[policy_spec],
            proportions=[1.0],
            episodes=episodes,
            action_timeout_ms=250,
            seed=42,
            save_replay=str(replay_dir),  # Directory path for replays
        )
        
        # Find generated replays (may be compressed as .json.z)
        replays = list(replay_dir.glob("*.json*")) + list(replay_dir.glob("*.msgpack*"))
        
        console.print(f"\n[green]✓ Generated {len(replays)} replay(s)[/green]")
        
        if replays:
            console.print("\n[bold]To view replays in MettaScope:[/bold]")
            console.print(f"  cogames replay {replays[0]}")
            console.print("\n[dim]Or open MettaScope directly and load the replay file.[/dim]")
        
        return {
            "status": "success",
            "replay_dir": str(replay_dir),
            "num_replays": len(replays),
            "replay_paths": [str(r) for r in replays],
        }
        
    except Exception as e:
        console.print(f"[red]Replay generation failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


def generate_comprehensive_report(
    output_dir: Path,
    eval_result: dict,
    sweep_results: list[dict],
    replay_result: dict | None,
) -> Path:
    """Generate comprehensive HTML report."""
    console.print("\n[bold cyan]═══ Generating Comprehensive Report ═══[/bold cyan]\n")
    
    report_path = output_dir / "COMPREHENSIVE_REPORT.html"
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Hello World Comprehensive Analysis</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{ 
            max-width: 1400px; 
            margin: 0 auto; 
            background: white; 
            padding: 40px; 
            border-radius: 15px; 
            box-shadow: 0 10px 40px rgba(0,0,0,0.3); 
        }}
        h1 {{ 
            color: #2c3e50; 
            border-bottom: 4px solid #667eea; 
            padding-bottom: 15px;
            margin-bottom: 30px;
        }}
        h2 {{ 
            color: #34495e; 
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
        }}
        .stats-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 20px; 
            margin: 25px 0; 
        }}
        .stat-card {{ 
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 25px; 
            border-radius: 12px; 
            text-align: center;
            transition: transform 0.2s;
        }}
        .stat-card:hover {{ transform: translateY(-5px); }}
        .stat-value {{ 
            font-size: 2.5em; 
            font-weight: bold; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .stat-label {{ color: #7f8c8d; margin-top: 8px; font-size: 1.1em; }}
        .image-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); 
            gap: 25px; 
        }}
        .image-card {{ 
            background: #f8f9fa; 
            padding: 20px; 
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .image-card img {{ 
            width: 100%; 
            height: auto; 
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }}
        .image-card h3 {{ 
            margin: 15px 0 0 0; 
            font-size: 1em; 
            color: #34495e;
            text-align: center;
        }}
        .timestamp {{ 
            color: #95a5a6; 
            font-size: 0.9em;
            margin-bottom: 20px;
        }}
        .section {{ 
            background: #f8f9fa; 
            padding: 25px; 
            border-radius: 12px; 
            margin: 25px 0;
        }}
        .success {{ color: #27ae60; }}
        .failed {{ color: #e74c3c; }}
        pre {{ 
            background: #2d3436; 
            color: #dfe6e9; 
            padding: 15px; 
            border-radius: 8px; 
            overflow-x: auto;
        }}
        .command {{ 
            background: #1e272e; 
            color: #00d2d3; 
            padding: 12px 20px; 
            border-radius: 8px; 
            font-family: 'Fira Code', monospace;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎮 Hello World Comprehensive Analysis</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{eval_result.get('summary', {}).get('baseline', {}).get('avg_reward', 0):.2f}</div>
                <div class="stat-label">Baseline Avg Reward</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{sum(s.get('num_trials', 0) for s in sweep_results)}</div>
                <div class="stat-label">Total Sweep Trials</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{max((s.get('best_metric', 0) or 0) for s in sweep_results) if sweep_results else 0:.2f}</div>
                <div class="stat-label">Best Sweep Metric</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{replay_result.get('num_replays', 0) if replay_result else 0}</div>
                <div class="stat-label">Replays Generated</div>
            </div>
        </div>

        <h2>📊 Policy Evaluation Results</h2>
        <div class="section">
            <p>Status: <span class="{'success' if eval_result.get('status') == 'success' else 'failed'}">{eval_result.get('status', 'N/A')}</span></p>
"""

    # Add evaluation summary
    if eval_result.get("summary"):
        html_content += "<h3>Summary Statistics</h3><table style='width:100%; border-collapse:collapse;'>"
        html_content += "<tr style='background:#667eea;color:white;'><th style='padding:12px;'>Policy</th><th>Avg Reward</th><th>Std Dev</th></tr>"
        for policy, stats in eval_result["summary"].items():
            html_content += f"<tr style='border-bottom:1px solid #ddd;'><td style='padding:10px;'>{policy}</td><td>{stats.get('avg_reward', 0):.4f}</td><td>{stats.get('std_dev', 0):.4f}</td></tr>"
        html_content += "</table>"
    
    html_content += "</div>"

    # Add sweep results
    html_content += "<h2>🔍 Hyperparameter Sweep Results</h2>"
    for i, sweep in enumerate(sweep_results):
        html_content += f"""
        <div class="section">
            <h3>Sweep {i+1}</h3>
            <p>Status: <span class="{'success' if sweep.get('status') == 'success' else 'failed'}">{sweep.get('status', 'N/A')}</span></p>
            <p>Trials: {sweep.get('num_trials', 0)}</p>
            <p>Best Config: <code>{sweep.get('best_config', 'N/A')}</code></p>
            <p>Best Metric: {sweep.get('best_metric', 0):.4f}</p>
        </div>
        """

    # Add replay info
    if replay_result and replay_result.get("status") == "success":
        html_content += f"""
        <h2>🎬 MettaScope Replays</h2>
        <div class="section">
            <p>Replays generated: {replay_result.get('num_replays', 0)}</p>
            <p>Replay directory: <code>{replay_result.get('replay_dir', 'N/A')}</code></p>
            <h3>View Replays</h3>
            <div class="command">cogames replay {replay_result.get('replay_paths', ['&lt;replay_path&gt;'])[0] if replay_result.get('replay_paths') else '&lt;replay_path&gt;'}</div>
        </div>
        """

    # Add visualization gallery
    html_content += """
        <h2>📈 Visualizations</h2>
        <p>Visualization files are available in the output directory.</p>
        <div class="section">
            <h3>Available Visualizations:</h3>
            <ul>
                <li><strong>evaluations/</strong> - Policy comparison charts</li>
                <li><strong>sweeps/</strong> - Hyperparameter analysis</li>
                <li><strong>replays/</strong> - MettaScope replay files</li>
            </ul>
        </div>

        <h2>🚀 Next Steps</h2>
        <div class="section">
            <h3>1. Analyze Results</h3>
            <p>Open the individual visualization files to understand policy behavior.</p>
            
            <h3>2. View Replays in MettaScope</h3>
            <div class="command">cogames replay &lt;replay_path&gt;</div>
            
            <h3>3. Implement Custom Policies</h3>
            <p>Use the baseline agent as a template in <code>src/cogames/policy/scripted_agent/</code></p>
            
            <h3>4. Run Full Sweeps</h3>
            <div class="command">python daf/scripts/run_full_suite.py --policies your_policy baseline --missions hello_world.open_world</div>
        </div>
    </div>
</body>
</html>
"""
    
    with open(report_path, "w") as f:
        f.write(html_content)
    
    console.print(f"[green]✓ Report saved: {report_path}[/green]")
    return report_path


def print_evaluation_summary(report) -> None:
    """Print evaluation summary table."""
    table = Table(title="📊 Evaluation Summary", show_header=True, header_style="bold magenta")
    table.add_column("Policy", style="cyan")
    table.add_column("Avg Reward", justify="right", style="green")
    table.add_column("Std Dev", justify="right", style="yellow")
    
    for policy in report.policies:
        avg = report.policy_averages.get(policy, 0.0)
        std = report.policy_std_devs.get(policy, 0.0)
        table.add_row(policy, f"{avg:.4f}", f"{std:.4f}")
    
    console.print(table)


def print_sweep_summary(sweep_result) -> None:
    """Print sweep summary table."""
    best = sweep_result.get_best_trial()
    
    table = Table(title="🔍 Sweep Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Trials", str(len(sweep_result.trials)))
    table.add_row("Best Metric", f"{best.primary_metric:.4f}" if best else "N/A")
    
    if best and best.hyperparameters:
        for hp, val in best.hyperparameters.items():
            table.add_row(f"Best {hp}", str(val))
    
    console.print(table)


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> int:
    """Run comprehensive Hello World analysis."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Hello World policy testing suite",
    )
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer episodes)")
    parser.add_argument("--with-replays", action="store_true", help="Generate MettaScope replays")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "comprehensive"], 
                        default="easy", help="Mission difficulty")
    parser.add_argument("--output-dir", type=Path, default=None, help="Custom output directory")
    
    args = parser.parse_args()
    
    # Configuration
    episodes = 2 if args.quick else 5
    sweep_episodes = 1 if args.quick else 3
    missions = MISSION_CONFIGS[args.difficulty]
    policies = ["baseline", "random"]
    
    output_dir = args.output_dir or get_output_dir()
    
    # Banner
    console.print(Panel(
        "[bold cyan]Hello World Comprehensive Analysis[/bold cyan]\n\n"
        f"Missions: {', '.join(missions)}\n"
        f"Policies: {', '.join(policies)}\n"
        f"Episodes: {episodes}\n"
        f"Mode: {'Quick' if args.quick else 'Standard'}\n"
        f"Output: {output_dir}",
        title="🎮 CoGames Policy Development Suite",
        border_style="cyan",
    ))
    
    start_time = time.time()
    
    # Phase 1: Policy Evaluation
    eval_result = run_policy_evaluation(
        policies=policies,
        missions=missions,
        episodes=episodes,
        output_dir=output_dir,
    )
    
    # Phase 2: Hyperparameter Sweep
    sweep_results = []
    
    # Sweep 1: Exploration parameters
    sweep_result = run_hyperparameter_sweep(
        policy="baseline",
        missions=missions[:1],  # Use first mission for sweep
        search_space={"exploration_rate": [0.1, 0.3], "energy_threshold": [0.1, 0.2]},
        episodes_per_trial=sweep_episodes,
        output_dir=output_dir,
    )
    sweep_results.append(sweep_result)
    
    # Phase 3: Replay Generation (optional)
    replay_result = None
    if args.with_replays:
        replay_result = run_with_replays(
            policy="baseline",
            mission=missions[0],
            episodes=2,
            output_dir=output_dir,
        )
    
    # Phase 4: Generate Comprehensive Report
    report_path = generate_comprehensive_report(
        output_dir=output_dir,
        eval_result=eval_result,
        sweep_results=sweep_results,
        replay_result=replay_result,
    )
    
    # Final Summary
    elapsed = time.time() - start_time
    
    console.print(Panel(
        f"[bold green]Analysis Complete![/bold green]\n\n"
        f"Duration: {elapsed:.1f}s\n"
        f"Output: {output_dir}\n\n"
        f"[cyan]Key Files:[/cyan]\n"
        f"  • {report_path.name}\n"
        f"  • evaluations/report.html\n"
        f"  • sweeps/baseline/sweep_progress.png\n\n"
        f"[dim]Open report:[/dim]\n"
        f"  open {report_path}",
        title="✨ Complete",
        border_style="green",
    ))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

