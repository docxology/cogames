"""Tests for daf.src.policy_analysis module.

Tests the policy analysis wrapper functions that integrate with
cogames.policy for discovering and comparing policy architectures.

NOTE: These tests require cogames policy classes to be importable.
They gracefully handle missing policies by checking availability.
"""

import logging

import pytest

logger = logging.getLogger(__name__)


class TestPolicyAnalysis:
    """Tests for policy analysis functions."""

    def test_list_available_policies(self):
        """daf_list_available_policies returns a list of policy info dicts."""
        from daf.src.eval.policy_analysis import daf_list_available_policies

        result = daf_list_available_policies()
        assert isinstance(result, list)
        # On systems without cogames policy packages, list may be empty
        if len(result) == 0:
            pytest.skip("No cogames policy classes available, skipping validation")

        for policy_info in result:
            assert isinstance(policy_info, dict)
            assert "name" in policy_info
        logger.info(f"Discovered {len(result)} policies")

    def test_analyze_policy_by_name(self):
        """daf_analyze_policy returns architecture details for a named policy."""
        from daf.src.eval.policy_analysis import daf_analyze_policy

        result = daf_analyze_policy("cogames.policy.starter_agent.StarterPolicy")
        assert isinstance(result, dict)
        assert "class_path" in result
        # If the policy isn't importable, the analysis should still return a dict
        if not result["importable"]:
            pytest.skip("StarterPolicy not importable, skipping detailed assertions")
        assert result["class_name"] == "StarterPolicy"
        logger.info(f"Analysis keys: {list(result.keys())}")

    def test_analyze_policy_nonexistent(self):
        """daf_analyze_policy handles nonexistent policy gracefully."""
        from daf.src.eval.policy_analysis import daf_analyze_policy

        result = daf_analyze_policy("com.nonexistent.NonExistentPolicy_xyz")
        assert isinstance(result, dict)
        assert result["importable"] is False
        assert len(result["errors"]) > 0

    def test_compare_policy_architectures(self):
        """daf_compare_policy_architectures compares multiple policies."""
        from daf.src.eval.policy_analysis import daf_compare_policy_architectures

        result = daf_compare_policy_architectures(
            policy_specs=["cogames.policy.starter_agent.StarterPolicy",
                          "com.nonexistent.TutorialPolicy"]
        )
        assert isinstance(result, list)
        assert len(result) == 2
        logger.info(f"Comparison returned {len(result)} analyses")
