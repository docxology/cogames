"""Tests for daf.src.eval.scoring_analysis module.

Tests the scoring analysis wrapper functions that integrate with
metta_alo.scoring for VOR computation, weighted scoring, and agent allocation.
"""

import logging

import pytest

logger = logging.getLogger(__name__)

# Import mettagrid eagerly — if not installed, skip entire module
pytest.importorskip("mettagrid", reason="mettagrid not installed — scoring tests require metta_alo.scoring")

from daf.src.eval.scoring_analysis import (
    daf_allocate_agents,
    daf_compute_vor,
    daf_compute_weighted_score,
    daf_scoring_summary,
    daf_validate_proportions,
)


class TestScoringAnalysis:
    """Tests for scoring analysis functions."""

    def test_compute_vor_basic(self):
        """daf_compute_vor returns VOR dict with correct keys."""
        result = daf_compute_vor(
            candidate_scores=[10.0, 12.0, 8.0],
            pool_scores=[5.0, 6.0, 7.0],
        )

        assert isinstance(result, dict)
        assert "vor" in result
        assert "candidate_mean" in result
        assert "pool_mean" in result
        assert result["vor"] == result["candidate_mean"] - result["pool_mean"]
        assert result["vor"] > 0  # candidate is better

    def test_compute_weighted_score(self):
        """daf_compute_weighted_score returns weighted average."""
        result = daf_compute_weighted_score(
            metrics={"mission_a": 10.0, "mission_b": 20.0},
            weights={"mission_a": 0.3, "mission_b": 0.7},
        )

        assert isinstance(result, dict)
        assert "score" in result
        assert "components" in result
        # Weighted average: (10*0.3 + 20*0.7) / (0.3+0.7) = 17.0
        assert abs(result["score"] - 17.0) < 0.001

    def test_allocate_agents(self):
        """daf_allocate_agents returns allocation list summing to total."""
        result = daf_allocate_agents(
            proportions=[0.5, 0.3, 0.2],
            total=10,
        )

        assert isinstance(result, list)
        assert len(result) == 3
        assert sum(result) == 10
        # Largest proportion should get most agents
        assert result[0] >= result[1] >= result[2]

    def test_validate_proportions_valid(self):
        """daf_validate_proportions accepts valid proportions."""
        result = daf_validate_proportions([0.5, 0.3, 0.2])

        assert isinstance(result, dict)
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_validate_proportions_invalid(self):
        """daf_validate_proportions rejects wrong-length proportions."""
        # num_policies defaults to len(proportions), so passing explicit mismatch
        result = daf_validate_proportions([0.5, 0.3], num_policies=3)

        assert isinstance(result, dict)
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_scoring_summary(self):
        """daf_scoring_summary returns comprehensive summary."""
        result = daf_scoring_summary(
            policy_scores={"agent_a": [10.0, 12.0, 8.0]},
            pool_scores=[5.0, 6.0, 7.0],
        )

        assert isinstance(result, dict)
        assert "agent_a" in result
        assert "mean" in result["agent_a"]
        assert "std" in result["agent_a"]
        assert "vor" in result["agent_a"]
        assert result["agent_a"]["vor"] > 0
