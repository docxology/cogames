"""Mission meta-analysis utilities for DAF.

DAF sidecar utility: Uses `cogames.cli.mission.get_mission_name_and_config()` for mission loading.

Provides mission discovery, analysis, and validation based on README.md patterns.
All mission loading invokes CoGames mission registry methods.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from rich.console import Console

logger = logging.getLogger("daf.mission_analysis")


class MissionAnalysis:
    """Analysis results for a mission or mission set."""

    def __init__(self, mission_name: str):
        """Initialize mission analysis.

        Args:
            mission_name: Name of the mission
        """
        self.mission_name = mission_name
        self.is_loadable: bool = False
        self.env_config: Optional[Any] = None
        self.mission_config: Optional[Any] = None
        self.num_agents: Optional[int] = None
        self.max_steps: Optional[int] = None
        self.has_map_builder: bool = False
        self.map_size: Optional[tuple[int, int]] = None
        self.errors: list[str] = []

    def summary(self) -> str:
        """Get human-readable summary.

        Returns:
            Summary string
        """
        if not self.is_loadable:
            return f"{self.mission_name}: NOT LOADABLE ({', '.join(self.errors)})"

        parts = [f"{self.mission_name}:"]
        if self.num_agents:
            parts.append(f"{self.num_agents} agents")
        if self.max_steps:
            parts.append(f"{self.max_steps} max steps")
        if self.map_size:
            parts.append(f"map {self.map_size[0]}x{self.map_size[1]}")

        return " ".join(parts)


def daf_analyze_mission(mission_name: str, console: Optional[Console] = None) -> MissionAnalysis:
    """Analyze a single mission configuration.

    Loads mission and extracts metadata about agents, steps, map size, etc.

    Args:
        mission_name: Mission name or path
        console: Optional Rich console for output

    Returns:
        MissionAnalysis with mission metadata
    """
    if console is None:
        console = Console()

    analysis = MissionAnalysis(mission_name)

    try:
        from cogames.cli.mission import get_mission

        # Use get_mission directly (avoids typer.Context dependency)
        resolved_name, env_cfg, mission_cfg = get_mission(mission_name)

        analysis.is_loadable = True
        analysis.env_config = env_cfg
        analysis.mission_config = mission_cfg

        # Extract metadata
        if env_cfg:
            if hasattr(env_cfg, "game"):
                analysis.num_agents = getattr(env_cfg.game, "num_agents", None)
                analysis.max_steps = getattr(env_cfg.game, "max_steps", None)

                # Check map builder
                map_builder = getattr(env_cfg.game, "map_builder", None)
                if map_builder:
                    analysis.has_map_builder = True
                    if hasattr(map_builder, "width") and hasattr(map_builder, "height"):
                        analysis.map_size = (map_builder.width, map_builder.height)

    except Exception as e:
        analysis.errors.append(str(e))
        logger.debug(f"Failed to analyze mission {mission_name}: {e}", exc_info=True)

    return analysis


def daf_discover_missions_from_readme(readme_path: Optional[Path] = None) -> list[str]:
    """Discover missions mentioned in README.md.

    Parses README.md to find mission names mentioned in examples and documentation.

    Args:
        readme_path: Path to README.md (defaults to repo root)

    Returns:
        List of mission names found in README
    """
    if readme_path is None:
        # Try to find README.md relative to this file
        readme_path = Path(__file__).parent.parent.parent / "README.md"

    if not readme_path.exists():
        logger.warning(f"README.md not found at {readme_path}")
        return []

    missions = set()

    try:
        content = readme_path.read_text()

        # Look for mission patterns in README
        # Examples: "training_facility_1", "assembler_2", "machina_1", etc.
        import re

        # Pattern for mission names (alphanumeric + underscore + dash)
        mission_pattern = r"\b([a-z][a-z0-9_-]+(?:_\d+)?)\b"

        # Find all potential mission names
        matches = re.findall(mission_pattern, content.lower())

        # Filter to known mission patterns
        known_prefixes = [
            "training_facility",
            "assembler",
            "machina",
            "hello_world",
            "harvest",
            "repair",
        ]

        for match in matches:
            for prefix in known_prefixes:
                if match.startswith(prefix):
                    missions.add(match)
                    break

    except Exception as e:
        logger.warning(f"Failed to parse README.md: {e}")

    return sorted(list(missions))


def daf_analyze_mission_set(mission_names: list[str], console: Optional[Console] = None) -> dict[str, MissionAnalysis]:
    """Analyze multiple missions.

    Args:
        mission_names: List of mission names
        console: Optional Rich console for output

    Returns:
        Dict mapping mission name to MissionAnalysis
    """
    if console is None:
        console = Console()

    analyses = {}

    for mission_name in mission_names:
        analysis = daf_analyze_mission(mission_name, console)
        analyses[mission_name] = analysis

    return analyses


def daf_validate_mission_set(mission_names: list[str], console: Optional[Console] = None) -> tuple[list[str], list[str]]:
    """Validate a set of missions.

    Args:
        mission_names: List of mission names to validate
        console: Optional Rich console for output

    Returns:
        Tuple of (valid_mission_names, invalid_mission_names)
    """
    if console is None:
        console = Console()

    valid = []
    invalid = []

    for mission_name in mission_names:
        analysis = daf_analyze_mission(mission_name, console)
        if analysis.is_loadable:
            valid.append(mission_name)
        else:
            invalid.append(mission_name)

    return valid, invalid


def daf_get_mission_metadata(mission_name: str) -> dict[str, Any]:
    """Get metadata dictionary for a mission.

    Args:
        mission_name: Mission name or path

    Returns:
        Dict with mission metadata
    """
    analysis = daf_analyze_mission(mission_name)

    metadata = {
        "mission_name": analysis.mission_name,
        "is_loadable": analysis.is_loadable,
        "num_agents": analysis.num_agents,
        "max_steps": analysis.max_steps,
        "has_map_builder": analysis.has_map_builder,
        "map_size": analysis.map_size,
        "errors": analysis.errors,
    }

    return metadata

