"""Mission meta-analysis utilities for DAF.

DAF sidecar utility: Uses `cogames.cli.mission.get_mission()` for mission loading
and `cogames.cogs_vs_clips.sites` for site/variant discovery.

Provides mission discovery, analysis, variant enumeration, and validation.
All mission loading invokes CoGames mission registry methods.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from rich.console import Console

logger = logging.getLogger("daf.mission_analysis")

# Current CoGames site name prefixes (from cogames.cogs_vs_clips.sites)
KNOWN_SITE_PREFIXES = [
    "cogsguard_machina_1",
    "cogsguard_arena",
    "evals",
    "training_facility",
    "hello_world",
    "machina_1",
]


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
        self.variants_applied: list[str] = []
        self.site_name: Optional[str] = None
        self.errors: list[str] = []

    def summary(self) -> str:
        """Get human-readable summary.

        Returns:
            Summary string
        """
        if not self.is_loadable:
            return f"{self.mission_name}: NOT LOADABLE ({', '.join(self.errors)})"

        parts = [f"{self.mission_name}:"]
        if self.site_name:
            parts.append(f"site={self.site_name}")
        if self.num_agents:
            parts.append(f"{self.num_agents} agents")
        if self.max_steps:
            parts.append(f"{self.max_steps} max steps")
        if self.map_size:
            parts.append(f"map {self.map_size[0]}x{self.map_size[1]}")
        if self.variants_applied:
            parts.append(f"variants={self.variants_applied}")

        return " ".join(parts)


def daf_analyze_mission(mission_name: str, console: Optional[Console] = None) -> MissionAnalysis:
    """Analyze a single mission configuration.

    Loads mission and extracts metadata about agents, steps, map size, etc.

    Args:
        mission_name: Mission name or path (e.g. 'cogsguard_machina_1.basic')
        console: Optional Rich console for output

    Returns:
        MissionAnalysis with mission metadata
    """
    if console is None:
        console = Console()

    analysis = MissionAnalysis(mission_name)

    try:
        from cogames.cli.mission import get_mission

        resolved_name, env_cfg, mission_obj = get_mission(mission_name)

        analysis.is_loadable = True
        analysis.env_config = env_cfg
        analysis.mission_config = mission_obj

        # Extract site name from mission object
        if mission_obj is not None and hasattr(mission_obj, "site"):
            analysis.site_name = getattr(mission_obj.site, "name", None)

        # Extract metadata from env config
        if env_cfg:
            if hasattr(env_cfg, "game"):
                analysis.num_agents = getattr(env_cfg.game, "num_agents", None)
                analysis.max_steps = getattr(env_cfg.game, "max_steps", None)

                map_builder = getattr(env_cfg.game, "map_builder", None)
                if map_builder:
                    analysis.has_map_builder = True
                    if hasattr(map_builder, "width") and hasattr(map_builder, "height"):
                        analysis.map_size = (map_builder.width, map_builder.height)

    except Exception as e:
        analysis.errors.append(str(e))
        logger.debug(f"Failed to analyze mission {mission_name}: {e}", exc_info=True)

    return analysis


def daf_list_all_sites() -> list[dict[str, Any]]:
    """List all registered CoGames sites.

    Returns:
        List of site metadata dicts with name, description, min/max cogs
    """
    try:
        from cogames.cogs_vs_clips.sites import SITES

        return [
            {
                "name": site.name,
                "description": site.description,
                "min_cogs": site.min_cogs,
                "max_cogs": site.max_cogs,
            }
            for site in SITES
        ]
    except ImportError:
        logger.warning("Could not import cogames.cogs_vs_clips.sites")
        return []


def daf_list_all_variants() -> list[dict[str, str]]:
    """List all registered mission variant classes.

    Returns:
        List of dicts with variant class name and description
    """
    try:
        from cogames.cogs_vs_clips import variants as variants_module
        from cogames.core import CoGameMissionVariant

        result = []
        for name in dir(variants_module):
            obj = getattr(variants_module, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, CoGameMissionVariant)
                and obj is not CoGameMissionVariant
            ):
                result.append({
                    "class_name": name,
                    "description": obj.__doc__.strip() if obj.__doc__ else "",
                })
        return result

    except ImportError:
        logger.warning("Could not import cogames.cogs_vs_clips.variants")
        return []


def daf_analyze_mission_variants(
    mission_name: str,
    variant_names: list[str],
    console: Optional[Console] = None,
) -> dict[str, MissionAnalysis]:
    """Analyze a mission under different variant configurations.

    Args:
        mission_name: Base mission name (e.g. 'cogsguard_machina_1')
        variant_names: List of variant names to apply individually
        console: Optional Rich console for output

    Returns:
        Dict mapping variant name to MissionAnalysis
    """
    if console is None:
        console = Console()

    results: dict[str, MissionAnalysis] = {}

    for variant in variant_names:
        analysis = MissionAnalysis(f"{mission_name}+{variant}")
        try:
            from cogames.cli.mission import get_mission

            resolved_name, env_cfg, mission_obj = get_mission(
                mission_name, variants_arg=[variant]
            )

            analysis.is_loadable = True
            analysis.env_config = env_cfg
            analysis.mission_config = mission_obj
            analysis.variants_applied = [variant]

            if mission_obj is not None and hasattr(mission_obj, "site"):
                analysis.site_name = getattr(mission_obj.site, "name", None)

            if env_cfg and hasattr(env_cfg, "game"):
                analysis.num_agents = getattr(env_cfg.game, "num_agents", None)
                analysis.max_steps = getattr(env_cfg.game, "max_steps", None)

        except Exception as e:
            analysis.errors.append(str(e))
            logger.debug(f"Failed to analyze {mission_name}+{variant}: {e}", exc_info=True)

        results[variant] = analysis

    return results


def daf_discover_missions_from_readme(readme_path: Optional[Path] = None) -> list[str]:
    """Discover missions mentioned in README.md.

    Parses README.md to find mission names mentioned in examples and documentation.

    Args:
        readme_path: Path to README.md (defaults to repo root)

    Returns:
        List of mission names found in README
    """
    if readme_path is None:
        readme_path = Path(__file__).parent.parent.parent / "README.md"

    if not readme_path.exists():
        logger.warning(f"README.md not found at {readme_path}")
        return []

    missions: set[str] = set()

    try:
        content = readme_path.read_text()
        import re

        mission_pattern = r"\b([a-z][a-z0-9_-]+(?:_\d+)?)\b"
        matches = re.findall(mission_pattern, content.lower())

        for match in matches:
            for prefix in KNOWN_SITE_PREFIXES:
                if match.startswith(prefix):
                    missions.add(match)
                    break

    except Exception as e:
        logger.warning(f"Failed to parse README.md: {e}")

    return sorted(list(missions))


def daf_analyze_mission_set(
    mission_names: list[str], console: Optional[Console] = None
) -> dict[str, MissionAnalysis]:
    """Analyze multiple missions.

    Args:
        mission_names: List of mission names
        console: Optional Rich console for output

    Returns:
        Dict mapping mission name to MissionAnalysis
    """
    if console is None:
        console = Console()

    return {name: daf_analyze_mission(name, console) for name in mission_names}


def daf_validate_mission_set(
    mission_names: list[str], console: Optional[Console] = None
) -> tuple[list[str], list[str]]:
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

    return {
        "mission_name": analysis.mission_name,
        "is_loadable": analysis.is_loadable,
        "site_name": analysis.site_name,
        "num_agents": analysis.num_agents,
        "max_steps": analysis.max_steps,
        "has_map_builder": analysis.has_map_builder,
        "map_size": analysis.map_size,
        "variants_applied": analysis.variants_applied,
        "errors": analysis.errors,
    }

