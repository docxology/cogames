"""Scoring analysis utilities for DAF.

DAF sidecar utility: Wraps `metta_alo.scoring` for VOR computation,
weighted scoring, and agent allocation.

Provides analytics-level access to the metta_alo scoring functions
with additional reporting and validation.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from rich.console import Console
from rich.table import Table

logger = logging.getLogger("daf.scoring_analysis")


def daf_compute_vor(
    candidate_scores: list[float],
    pool_scores: list[float],
    console: Optional[Console] = None,
) -> dict[str, float]:
    """Compute Value Over Replacement for a candidate vs pool.

    Uses metta_alo.scoring.value_over_replacement() on the means of
    candidate and pool score lists.

    Args:
        candidate_scores: Scores for the candidate policy
        pool_scores: Scores for the replacement pool
        console: Optional Rich console for output

    Returns:
        Dict with 'vor', 'candidate_mean', 'pool_mean' keys
    """
    from metta_alo.scoring import value_over_replacement

    import numpy as np

    candidate_mean = float(np.mean(candidate_scores))
    pool_mean = float(np.mean(pool_scores))
    vor = value_over_replacement(candidate_mean, pool_mean)

    result = {
        "vor": vor,
        "candidate_mean": candidate_mean,
        "pool_mean": pool_mean,
    }

    if console:
        console.print(f"[bold]VOR:[/bold] {vor:+.4f} "
                       f"(candidate: {candidate_mean:.4f}, pool: {pool_mean:.4f})")

    logger.info(f"VOR computed: {vor:+.4f}")
    return result


def daf_compute_weighted_score(
    metrics: dict[str, float],
    weights: dict[str, float],
    console: Optional[Console] = None,
) -> dict[str, Any]:
    """Compute weighted composite score from named metrics.

    This is a DAF-level utility that computes a weighted average of
    named metrics. It does NOT wrap compute_weighted_scores() (which
    operates on UUID-keyed match objects).

    Args:
        metrics: Dict mapping metric name to value
        weights: Dict mapping metric name to weight
        console: Optional Rich console for output

    Returns:
        Dict with 'score', 'components' keys
    """
    # Compute weighted average directly — the metta_alo function
    # operates on UUID-keyed match objects, not simple dicts.
    total_weight = sum(weights.get(k, 0) for k in metrics)
    if total_weight <= 0:
        score = 0.0
    else:
        score = sum(metrics[k] * weights.get(k, 0) for k in metrics) / total_weight

    components = {}
    for key in metrics:
        if key in weights and total_weight > 0:
            components[key] = {
                "value": metrics[key],
                "weight": weights[key],
                "contribution": metrics[key] * weights[key] / total_weight,
            }

    result = {
        "score": score,
        "components": components,
    }

    if console:
        table = Table(title="Weighted Score Breakdown")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_column("Weight", justify="right")
        table.add_column("Contribution", justify="right")

        for key, comp in components.items():
            table.add_row(
                key,
                f"{comp['value']:.4f}",
                f"{comp['weight']:.2f}",
                f"{comp['contribution']:.4f}",
            )
        table.add_row("[bold]Total[/bold]", "", "", f"[bold]{score:.4f}[/bold]")
        console.print(table)

    logger.info(f"Weighted score: {score:.4f}")
    return result


def daf_allocate_agents(
    proportions: list[float],
    total: int,
) -> list[int]:
    """Allocate agent counts according to proportions.

    Wraps metta_alo.scoring.allocate_counts(total, weights).

    Args:
        proportions: List of desired proportions (used as weights)
        total: Total number of agents to allocate

    Returns:
        List of integer allocations summing to total
    """
    from metta_alo.scoring import allocate_counts

    # allocate_counts signature: (total: int, weights: Sequence[float])
    result = allocate_counts(total, proportions)
    logger.info(f"Allocated {total} agents: {result}")
    return result


def daf_validate_proportions(
    proportions: list[float],
    num_policies: int | None = None,
) -> dict[str, Any]:
    """Validate proportions for agent allocation.

    Wraps metta_alo.scoring.validate_proportions(proportions, num_policies).

    Args:
        proportions: List of proportions to validate
        num_policies: Number of policies (defaults to len(proportions))

    Returns:
        Dict with 'valid', 'errors' keys
    """
    from metta_alo.scoring import validate_proportions

    if num_policies is None:
        num_policies = len(proportions)

    try:
        validate_proportions(proportions, num_policies)
        return {"valid": True, "errors": []}
    except ValueError as e:
        return {"valid": False, "errors": [str(e)]}


def daf_scoring_summary(
    policy_scores: dict[str, list[float]],
    pool_scores: list[float],
    console: Optional[Console] = None,
) -> dict[str, dict[str, float]]:
    """Generate comprehensive scoring summary for multiple policies.

    Args:
        policy_scores: Dict mapping policy name to list of scores
        pool_scores: Replacement pool scores for VOR computation
        console: Optional Rich console for output

    Returns:
        Dict mapping policy name to scoring metrics
    """
    import numpy as np

    results = {}

    for name, scores in policy_scores.items():
        vor_result = daf_compute_vor(scores, pool_scores)
        results[name] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "vor": vor_result["vor"],
        }

    if console:
        table = Table(title="Policy Scoring Summary")
        table.add_column("Policy")
        table.add_column("Mean", justify="right")
        table.add_column("Std", justify="right")
        table.add_column("VOR", justify="right")

        for name, metrics in results.items():
            table.add_row(
                name,
                f"{metrics['mean']:.4f}",
                f"{metrics['std']:.4f}",
                f"{metrics['vor']:+.4f}",
            )
        console.print(table)

    return results
