#!/usr/bin/env python3
"""
GIF Visualization Example for CoGames DAF.

This example demonstrates how to:
1. Run a simulation and save replay files
2. Generate animated GIFs from replays
3. Open replays in MettaScope web viewer
4. Generate individual frames for video creation

Usage:
    python daf/examples/gif_visualization_example.py
    python daf/examples/gif_visualization_example.py --mission hello_world.easy_hearts
    python daf/examples/gif_visualization_example.py --frames-only  # Generate PNG frames
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

console = Console()


def run_simulation_with_replay(
    mission: str,
    policy: str,
    output_dir: Path,
    episodes: int = 1,
    seed: int = 42,
) -> list[Path]:
    """Run simulation and save replay files.
    
    Returns list of replay file paths.
    """
    from cogames import evaluate as evaluate_module
    from cogames.cli.mission import get_mission
    from mettagrid.policy.policy import PolicySpec
    
    console.print(f"\n[bold cyan]Running simulation...[/bold cyan]")
    console.print(f"  Mission: {mission}")
    console.print(f"  Policy: {policy}")
    console.print(f"  Episodes: {episodes}")
    
    replay_dir = output_dir / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    
    # Load mission
    name, env_cfg, _ = get_mission(mission)
    policy_spec = PolicySpec(class_path=policy)
    
    # Run evaluation with replay saving
    evaluate_module.evaluate(
        console=Console(quiet=True),
        missions=[(name, env_cfg)],
        policy_specs=[policy_spec],
        proportions=[1.0],
        episodes=episodes,
        action_timeout_ms=250,
        seed=seed,
        save_replay=str(replay_dir),
    )
    
    # Find generated replays
    replays = list(replay_dir.glob("*.json*"))
    console.print(f"[green]✓ Generated {len(replays)} replay(s)[/green]")
    
    return replays


def generate_gifs(
    replay_paths: list[Path],
    output_dir: Path,
    fps: int = 10,
    cell_size: int = 16,
) -> list[Path]:
    """Generate GIFs from replay files."""
    from daf.src.gif_generator import generate_gif_from_replay
    
    console.print(f"\n[bold cyan]Generating GIFs...[/bold cyan]")
    
    gif_dir = output_dir / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)
    
    gif_paths = []
    for replay_path in replay_paths:
        gif_name = replay_path.stem.replace(".json", "") + ".gif"
        gif_path = gif_dir / gif_name
        
        try:
            generate_gif_from_replay(
                replay_path,
                gif_path,
                fps=fps,
                cell_size=cell_size,
            )
            gif_paths.append(gif_path)
            console.print(f"  [green]✓[/green] {gif_path.name}")
        except Exception as e:
            console.print(f"  [red]✗[/red] {replay_path.name}: {e}")
    
    return gif_paths


def generate_frames(
    replay_paths: list[Path],
    output_dir: Path,
    cell_size: int = 16,
) -> dict[Path, list[Path]]:
    """Generate individual frames from replays."""
    from daf.src.gif_generator import generate_frames_only
    
    console.print(f"\n[bold cyan]Generating frames...[/bold cyan]")
    
    all_frames = {}
    for replay_path in replay_paths:
        frame_dir = output_dir / "frames" / replay_path.stem.replace(".json", "")
        
        try:
            frames = generate_frames_only(
                replay_path,
                frame_dir,
                cell_size=cell_size,
            )
            all_frames[replay_path] = frames
            console.print(f"  [green]✓[/green] {len(frames)} frames -> {frame_dir}")
        except Exception as e:
            console.print(f"  [red]✗[/red] {replay_path.name}: {e}")
    
    return all_frames


def show_mettascope_instructions(replay_paths: list[Path]) -> None:
    """Print instructions for viewing replays in MettaScope."""
    console.print("\n[bold cyan]MettaScope Viewing Options[/bold cyan]\n")
    
    # CLI option
    console.print("[bold]Option 1: Local MettaScope (requires Nim)[/bold]")
    for path in replay_paths[:3]:
        console.print(f"  cogames replay {path}")
    
    console.print("\n[bold]Option 2: Web MettaScope[/bold]")
    console.print("  Upload replay to a public URL and visit:")
    console.print("  https://metta-ai.github.io/metta/mettascope/mettascope.html?replay=<YOUR_URL>")
    
    console.print("\n[bold]Option 3: FFmpeg video (from frames)[/bold]")
    console.print("  ffmpeg -framerate 10 -pattern_type glob -i 'frames/*.png' \\")
    console.print("         -c:v libx264 -pix_fmt yuv420p output.mp4")


def main():
    parser = argparse.ArgumentParser(
        description="Generate GIF visualizations from CoGames simulations"
    )
    parser.add_argument(
        "-m", "--mission",
        default="hello_world.easy_hearts",
        help="Mission name (default: hello_world.easy_hearts)"
    )
    parser.add_argument(
        "-p", "--policy",
        default="baseline",
        help="Policy to use (default: baseline)"
    )
    parser.add_argument(
        "-e", "--episodes",
        type=int,
        default=1,
        help="Number of episodes (default: 1)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="GIF frames per second (default: 10)"
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=16,
        help="Cell size in pixels (default: 16)"
    )
    parser.add_argument(
        "--frames-only",
        action="store_true",
        help="Generate only PNG frames (no GIF)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        help="Output directory (default: daf_output/visualizations/<timestamp>)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--replay",
        type=Path,
        help="Use existing replay file instead of running simulation"
    )
    
    args = parser.parse_args()
    
    # Set up output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("daf_output/visualizations") / f"viz_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(Panel.fit(
        "[bold]CoGames GIF Visualization Generator[/bold]\n"
        f"Output: {output_dir}",
        border_style="cyan"
    ))
    
    # Get replay files
    if args.replay:
        replay_paths = [args.replay]
        console.print(f"\n[dim]Using existing replay: {args.replay}[/dim]")
    else:
        replay_paths = run_simulation_with_replay(
            mission=args.mission,
            policy=args.policy,
            output_dir=output_dir,
            episodes=args.episodes,
            seed=args.seed,
        )
    
    if not replay_paths:
        console.print("[red]No replays available for visualization[/red]")
        return
    
    # Generate visualizations
    if args.frames_only:
        frames = generate_frames(
            replay_paths,
            output_dir,
            cell_size=args.cell_size,
        )
        console.print(f"\n[green]✓ Generated frames for {len(frames)} replay(s)[/green]")
    else:
        gif_paths = generate_gifs(
            replay_paths,
            output_dir,
            fps=args.fps,
            cell_size=args.cell_size,
        )
        console.print(f"\n[green]✓ Generated {len(gif_paths)} GIF(s)[/green]")
    
    # Show MettaScope instructions
    show_mettascope_instructions(replay_paths)
    
    # Summary table
    console.print("\n")
    table = Table(title="Generated Files", show_header=True)
    table.add_column("Type", style="cyan")
    table.add_column("Path", style="green")
    
    for replay in replay_paths:
        table.add_row("Replay", str(replay))
    
    if not args.frames_only:
        gif_dir = output_dir / "gifs"
        for gif in gif_dir.glob("*.gif"):
            table.add_row("GIF", str(gif))
    else:
        frames_dir = output_dir / "frames"
        for frame_dir in frames_dir.iterdir():
            if frame_dir.is_dir():
                num_frames = len(list(frame_dir.glob("*.png")))
                table.add_row("Frames", f"{frame_dir} ({num_frames} files)")
    
    console.print(table)
    
    console.print(f"\n[bold green]✓ Visualization complete![/bold green]")
    console.print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

