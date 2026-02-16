"""Policy analysis utilities for DAF.

DAF sidecar utility: Wraps cogames policy framework for analysis and comparison.

Provides policy discovery, architecture analysis, and multi-policy
comparison using real policy classes from cogames.policy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.table import Table

logger = logging.getLogger("daf.policy_analysis")


def daf_list_available_policies() -> list[dict[str, str]]:
    """List all available policy implementations.

    Discovers policies from cogames.policy namespace.

    Returns:
        List of dicts with 'name', 'class_path', 'description' keys
    """
    policies = []

    # Known policy classes in the cogames ecosystem
    known_policies = [
        {
            "name": "starter",
            "class_path": "cogames.policy.starter_agent.StarterPolicy",
            "description": "Basic trainable starter policy with LSTM backbone",
        },
        {
            "name": "pufferlib_cogs",
            "class_path": "cogames.policy.pufferlib_policy.PufferlibCogsPolicy",
            "description": "PufferLib integration policy for CoGames environments",
        },
        {
            "name": "baseline",
            "class_path": "cogames.policy.scripted_agent.baseline_agent.BaselinePolicy",
            "description": "Scripted baseline agent with heuristic behavior",
        },
        {
            "name": "tutorial",
            "class_path": "cogames.policy.tutorial_policy.TutorialPolicy",
            "description": "Tutorial policy for learning game mechanics",
        },
        {
            "name": "trainable_template",
            "class_path": "cogames.policy.trainable_policy_template.TrainablePolicyTemplate",
            "description": "Template for creating custom trainable policies",
        },
    ]

    for policy_info in known_policies:
        try:
            # Verify the class is importable
            module_path, class_name = policy_info["class_path"].rsplit(".", 1)
            import importlib
            mod = importlib.import_module(module_path)
            if hasattr(mod, class_name):
                policies.append(policy_info)
                logger.debug(f"Found policy: {policy_info['name']}")
        except (ImportError, ModuleNotFoundError):
            logger.debug(f"Policy not available: {policy_info['class_path']}")
            continue

    return policies


def daf_analyze_policy(
    policy_class_path: str,
    weights_path: Optional[Path] = None,
    console: Optional[Console] = None,
) -> dict[str, Any]:
    """Analyze a policy's architecture and capabilities.

    Args:
        policy_class_path: Full class path (e.g. 'cogames.policy.starter_agent.StarterPolicy')
        weights_path: Optional path to trained weights
        console: Optional Rich console for output

    Returns:
        Dict with architecture info: class_name, module, is_recurrent,
        has_weights, parameter_count, etc.
    """
    if console is None:
        console = Console()

    result: dict[str, Any] = {
        "class_path": policy_class_path,
        "importable": False,
        "class_name": None,
        "module": None,
        "has_weights": weights_path is not None and Path(weights_path).exists(),
        "is_recurrent": None,
        "parameter_count": None,
        "docstring": None,
        "errors": [],
    }

    try:
        import importlib

        module_path, class_name = policy_class_path.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)

        result["importable"] = True
        result["class_name"] = class_name
        result["module"] = module_path
        result["docstring"] = cls.__doc__.strip() if cls.__doc__ else None

        # Check for recurrent architecture markers
        result["is_recurrent"] = any(
            hasattr(cls, attr) for attr in ["lstm", "gru", "recurrent", "hidden_size"]
        )

        # Check for known architecture methods
        result["has_forward"] = hasattr(cls, "forward")
        result["has_get_action"] = hasattr(cls, "get_action") or hasattr(cls, "act")

        logger.info(f"Analyzed policy: {class_name} (recurrent={result['is_recurrent']})")

    except (ImportError, ModuleNotFoundError) as e:
        result["errors"].append(f"Import error: {e}")
        logger.warning(f"Could not import policy {policy_class_path}: {e}")
    except AttributeError as e:
        result["errors"].append(f"Class not found: {e}")
        logger.warning(f"Class not found in module: {e}")
    except Exception as e:
        result["errors"].append(f"Analysis error: {e}")
        logger.warning(f"Error analyzing policy: {e}")

    if console and result["importable"]:
        console.print(f"[bold]{result['class_name']}[/bold] ({result['module']})")
        if result["docstring"]:
            console.print(f"  {result['docstring'][:100]}")
        console.print(f"  Recurrent: {result['is_recurrent']}")
        console.print(f"  Has weights: {result['has_weights']}")

    return result


def daf_compare_policy_architectures(
    policy_specs: list[str],
    console: Optional[Console] = None,
) -> list[dict[str, Any]]:
    """Compare architectures of multiple policies.

    Args:
        policy_specs: List of policy class paths
        console: Optional Rich console for output

    Returns:
        List of analysis dicts for each policy
    """
    if console is None:
        console = Console()

    analyses = []
    for spec in policy_specs:
        analysis = daf_analyze_policy(spec, console=None)
        analyses.append(analysis)

    if console:
        table = Table(title="Policy Architecture Comparison")
        table.add_column("Policy")
        table.add_column("Importable")
        table.add_column("Recurrent")
        table.add_column("Has Forward")

        for a in analyses:
            table.add_row(
                a.get("class_name", a["class_path"]),
                "✓" if a["importable"] else "✗",
                str(a.get("is_recurrent", "?")),
                "✓" if a.get("has_forward") else "✗",
            )
        console.print(table)

    return analyses


def daf_get_policy_spec(
    short_name: str,
    weights_path: Optional[str] = None,
) -> Optional[Any]:
    """Get a PolicySpec from a short name.

    Args:
        short_name: Short policy name (e.g. 'starter', 'baseline')
        weights_path: Optional path to weights file

    Returns:
        PolicySpec instance or None if not found
    """
    policies = daf_list_available_policies()
    for p in policies:
        if p["name"] == short_name:
            try:
                from mettagrid.policy.policy import PolicySpec

                return PolicySpec(
                    class_path=p["class_path"],
                    data_path=weights_path,
                )
            except ImportError:
                logger.warning("mettagrid.policy.policy not available")
                return None

    logger.warning(f"Unknown policy short name: {short_name}")
    return None
