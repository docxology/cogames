"""
GIF Animation Generator for CoGames Simulations.

This module provides utilities for generating animated GIF visualizations
from CoGames replay files or live simulation runs.

Approaches:
1. Direct rendering from replay files using matplotlib/PIL
2. Integration with MettaScope web viewer for browser-based GIF capture
3. Headless simulation rendering to frames then GIF

Usage:
    from daf.src.gif_generator import generate_gif_from_replay, generate_gif_from_simulation
    
    # From replay file
    generate_gif_from_replay("replay.json.z", "output.gif")
    
    # From live simulation
    generate_gif_from_simulation(env_cfg, policy_spec, "output.gif", episodes=1)
"""

import json
import logging
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from mettagrid import MettaGridConfig
    from mettagrid.policy.policy import PolicySpec

logger = logging.getLogger("daf.gif_generator")


# Color palette for different object types
OBJECT_COLORS = {
    "agent": (66, 135, 245),       # Blue
    "wall": (128, 128, 128),       # Gray
    "resource": (34, 197, 94),     # Green
    "chest": (234, 179, 8),        # Yellow
    "generator": (168, 85, 247),   # Purple
    "altar": (236, 72, 153),       # Pink
    "converter": (249, 115, 22),   # Orange
    "default": (200, 200, 200),    # Light gray
}

AGENT_COLORS = [
    (66, 135, 245),   # Blue
    (239, 68, 68),    # Red
    (34, 197, 94),    # Green
    (234, 179, 8),    # Yellow
    (168, 85, 247),   # Purple
    (236, 72, 153),   # Pink
    (249, 115, 22),   # Orange
    (6, 182, 212),    # Cyan
]


def load_replay(replay_path: Path | str) -> dict:
    """Load and decompress a replay file.
    
    Args:
        replay_path: Path to replay file (.json or .json.z)
        
    Returns:
        Parsed replay data dictionary
    """
    replay_path = Path(replay_path)
    
    if not replay_path.exists():
        raise FileNotFoundError(f"Replay file not found: {replay_path}")
    
    with open(replay_path, "rb") as f:
        data = f.read()
    
    # Decompress if needed
    if replay_path.suffix == ".z" or replay_path.name.endswith(".json.z"):
        data = zlib.decompress(data)
    
    return json.loads(data.decode("utf-8"))


def _interpolate_value(changes: list | Any, step: int) -> Any:
    """Get the value at a specific step from sequence-keyed data.
    
    Args:
        changes: Either a static value or list of [step, value] pairs
        step: The simulation step to query
        
    Returns:
        The value at the given step
    """
    if not isinstance(changes, list):
        return changes
    
    if len(changes) == 0:
        return None
    
    # Check if it's a list of [step, value] pairs (e.g., [[0, [x,y]], [5, [x2,y2]]])
    if isinstance(changes[0], (list, tuple)) and len(changes[0]) == 2:
        first_elem = changes[0][0]
        # If first element of pair is an int, it's a sequence-keyed format
        if isinstance(first_elem, int):
            value = changes[0][1]
            for change in changes:
                if change[0] <= step:
                    value = change[1]
                else:
                    break
            return value
    
    # Otherwise it's a static value (like [x, y] for location)
    return changes


def render_frame_matplotlib(
    replay: dict,
    step: int,
    width: int = 800,
    height: int = 600,
    cell_size: int = 16,
) -> np.ndarray:
    """Render a single frame from replay data using matplotlib.
    
    Args:
        replay: Parsed replay data
        step: Simulation step to render
        width: Output image width
        height: Output image height
        cell_size: Size of each grid cell in pixels
        
    Returns:
        RGB numpy array of shape (height, width, 3)
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Circle
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError:
        raise ImportError("matplotlib required for frame rendering. Install with: pip install matplotlib")
    
    map_width, map_height = replay.get("map_size", [32, 32])
    
    # Calculate figure size to fit map
    fig_width = (map_width * cell_size) / 100
    fig_height = (map_height * cell_size) / 100
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    ax.set_xlim(0, map_width)
    ax.set_ylim(0, map_height)
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a2e")
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Remove margins
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Render objects
    objects = replay.get("objects", [])
    
    for obj in objects:
        # Get type name directly from object
        type_name = obj.get("type_name", "unknown")
        
        # Get position at this step
        location = _interpolate_value(obj.get("location"), step)
        if location is None:
            continue
        
        # Location is [x, y] or [col, row]
        if isinstance(location, (list, tuple)) and len(location) == 2:
            x, y = location
        else:
            continue
        
        # Determine color based on type_name
        if "agent" in type_name.lower():
            agent_id = obj.get("id", 0) % len(AGENT_COLORS)
            color = AGENT_COLORS[agent_id]
            color_normalized = tuple(c / 255 for c in color)
            circle = Circle((x + 0.5, map_height - y - 0.5), 0.4, 
                           facecolor=color_normalized, edgecolor="white", linewidth=1)
            ax.add_patch(circle)
        elif "wall" in type_name.lower():
            color = OBJECT_COLORS["wall"]
            color_normalized = tuple(c / 255 for c in color)
            rect = Rectangle((x, map_height - y - 1), 1, 1, 
                            facecolor=color_normalized, edgecolor="none")
            ax.add_patch(rect)
        elif "chest" in type_name.lower():
            color = OBJECT_COLORS["chest"]
            color_normalized = tuple(c / 255 for c in color)
            rect = Rectangle((x + 0.1, map_height - y - 0.9), 0.8, 0.8, 
                            facecolor=color_normalized, edgecolor="white", linewidth=0.5)
            ax.add_patch(rect)
        elif "extractor" in type_name.lower() or "generator" in type_name.lower():
            color = OBJECT_COLORS["generator"]
            color_normalized = tuple(c / 255 for c in color)
            rect = Rectangle((x + 0.1, map_height - y - 0.9), 0.8, 0.8, 
                            facecolor=color_normalized, edgecolor="white", linewidth=0.5)
            ax.add_patch(rect)
        elif "charger" in type_name.lower():
            color = (6, 182, 212)  # Cyan
            color_normalized = tuple(c / 255 for c in color)
            rect = Rectangle((x + 0.1, map_height - y - 0.9), 0.8, 0.8, 
                            facecolor=color_normalized, edgecolor="white", linewidth=0.5)
            ax.add_patch(rect)
        elif "assembler" in type_name.lower() or "converter" in type_name.lower():
            color = OBJECT_COLORS["converter"]
            color_normalized = tuple(c / 255 for c in color)
            rect = Rectangle((x + 0.1, map_height - y - 0.9), 0.8, 0.8, 
                            facecolor=color_normalized, edgecolor="white", linewidth=0.5)
            ax.add_patch(rect)
        elif "altar" in type_name.lower():
            color = OBJECT_COLORS["altar"]
            color_normalized = tuple(c / 255 for c in color)
            rect = Rectangle((x + 0.1, map_height - y - 0.9), 0.8, 0.8, 
                            facecolor=color_normalized, edgecolor="white", linewidth=0.5)
            ax.add_patch(rect)
        else:
            # Skip unknown types to reduce clutter
            pass
    
    # Add step counter
    ax.text(0.5, map_height - 0.5, f"Step: {step}", fontsize=10, color="white",
            ha="left", va="top", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
    
    # Convert to numpy array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    
    # Get the RGBA buffer (compatible with newer matplotlib versions)
    width, height = fig.get_size_inches() * fig.get_dpi()
    width, height = int(width), int(height)
    
    # Use buffer_rgba() for newer matplotlib versions
    buf = np.asarray(canvas.buffer_rgba())
    buf = buf.reshape(height, width, 4)
    # Convert RGBA to RGB
    buf = buf[:, :, :3]
    
    plt.close(fig)
    
    return buf


def generate_gif_from_replay(
    replay_path: Path | str,
    output_path: Path | str,
    fps: int = 10,
    cell_size: int = 16,
    start_step: int = 0,
    end_step: Optional[int] = None,
    step_interval: int = 1,
) -> Path:
    """Generate an animated GIF from a replay file.
    
    Args:
        replay_path: Path to replay file (.json or .json.z)
        output_path: Path for output GIF
        fps: Frames per second
        cell_size: Size of each grid cell in pixels
        start_step: First step to render
        end_step: Last step to render (None = all steps)
        step_interval: Render every Nth step
        
    Returns:
        Path to generated GIF
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL required for GIF generation. Install with: pip install pillow")
    
    replay_path = Path(replay_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading replay from {replay_path}")
    replay = load_replay(replay_path)
    
    max_steps = replay.get("max_steps", 100)
    if end_step is None:
        end_step = max_steps
    
    steps = range(start_step, min(end_step, max_steps), step_interval)
    
    logger.info(f"Rendering {len(list(steps))} frames...")
    
    frames = []
    for step in steps:
        frame = render_frame_matplotlib(replay, step, cell_size=cell_size)
        img = Image.fromarray(frame)
        frames.append(img)
    
    if not frames:
        raise ValueError("No frames to render")
    
    # Save as GIF
    duration_ms = int(1000 / fps)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,  # Infinite loop
        optimize=True,
    )
    
    logger.info(f"GIF saved to {output_path}")
    return output_path


def generate_gif_from_simulation(
    env_cfg: "MettaGridConfig",
    policy_spec: "PolicySpec",
    output_path: Path | str,
    seed: int = 42,
    fps: int = 10,
    cell_size: int = 16,
    max_steps: Optional[int] = None,
    step_interval: int = 1,
) -> Path:
    """Generate a GIF from a live simulation run.
    
    This runs the simulation, captures frames, and creates a GIF.
    
    Args:
        env_cfg: MettaGridConfig for the simulation
        policy_spec: Policy specification for agents
        output_path: Path for output GIF
        seed: Random seed
        fps: Frames per second
        cell_size: Grid cell size in pixels
        max_steps: Maximum steps to run (None = use config)
        step_interval: Capture every Nth step
        
    Returns:
        Path to generated GIF
    """
    from tempfile import TemporaryDirectory
    
    output_path = Path(output_path)
    
    # Run simulation with replay capture
    with TemporaryDirectory() as tmpdir:
        from cogames import evaluate as evaluate_module
        from rich.console import Console
        
        console = Console(quiet=True)
        
        # Run evaluation with replay saving
        evaluate_module.evaluate(
            console=console,
            missions=[("sim", env_cfg)],
            policy_specs=[policy_spec],
            proportions=[1.0],
            episodes=1,
            action_timeout_ms=250,
            seed=seed,
            save_replay=tmpdir,
        )
        
        # Find the generated replay
        replays = list(Path(tmpdir).glob("*.json*"))
        if not replays:
            raise RuntimeError("No replay generated from simulation")
        
        replay_path = replays[0]
        
        # Generate GIF from replay
        return generate_gif_from_replay(
            replay_path,
            output_path,
            fps=fps,
            cell_size=cell_size,
            step_interval=step_interval,
            end_step=max_steps,
        )


def get_mettascope_url(replay_url: str) -> str:
    """Get the MettaScope web viewer URL for a replay.
    
    Args:
        replay_url: HTTP URL to the replay file
        
    Returns:
        Full MettaScope viewer URL
    """
    return f"https://metta-ai.github.io/metta/mettascope/mettascope.html?replay={replay_url}"


def generate_frames_only(
    replay_path: Path | str,
    output_dir: Path | str,
    cell_size: int = 16,
    start_step: int = 0,
    end_step: Optional[int] = None,
    step_interval: int = 1,
    format: str = "png",
) -> list[Path]:
    """Generate individual frame images from a replay.
    
    Useful for creating videos with external tools like ffmpeg.
    
    Args:
        replay_path: Path to replay file
        output_dir: Directory to save frames
        cell_size: Grid cell size in pixels
        start_step: First step to render
        end_step: Last step to render
        step_interval: Render every Nth step
        format: Image format (png, jpg)
        
    Returns:
        List of paths to generated frame images
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL required. Install with: pip install pillow")
    
    replay_path = Path(replay_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    replay = load_replay(replay_path)
    max_steps = replay.get("max_steps", 100)
    
    if end_step is None:
        end_step = max_steps
    
    steps = range(start_step, min(end_step, max_steps), step_interval)
    
    frame_paths = []
    for i, step in enumerate(steps):
        frame = render_frame_matplotlib(replay, step, cell_size=cell_size)
        img = Image.fromarray(frame)
        frame_path = output_dir / f"frame_{i:05d}.{format}"
        img.save(frame_path)
        frame_paths.append(frame_path)
    
    return frame_paths


# CLI interface for standalone usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate GIF from CoGames replay")
    parser.add_argument("replay", help="Path to replay file (.json or .json.z)")
    parser.add_argument("-o", "--output", default="output.gif", help="Output GIF path")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    parser.add_argument("--cell-size", type=int, default=16, help="Cell size in pixels")
    parser.add_argument("--start", type=int, default=0, help="Start step")
    parser.add_argument("--end", type=int, help="End step")
    parser.add_argument("--interval", type=int, default=1, help="Step interval")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    generate_gif_from_replay(
        args.replay,
        args.output,
        fps=args.fps,
        cell_size=args.cell_size,
        start_step=args.start,
        end_step=args.end,
        step_interval=args.interval,
    )
    
    print(f"GIF generated: {args.output}")

