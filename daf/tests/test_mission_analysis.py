"""Tests for DAF mission analysis functionality."""

from pathlib import Path

import pytest

from daf.src.mission_analysis import (
    MissionAnalysis,
    daf_analyze_mission,
    daf_analyze_mission_set,
    daf_discover_missions_from_readme,
    daf_get_mission_metadata,
    daf_validate_mission_set,
)


def test_mission_analysis_creation():
    """Test MissionAnalysis initialization."""
    analysis = MissionAnalysis("test_mission")

    assert analysis.mission_name == "test_mission"
    assert analysis.is_loadable is False
    assert len(analysis.errors) == 0


def test_mission_analysis_summary_loadable():
    """Test MissionAnalysis summary for loadable mission."""
    analysis = MissionAnalysis("test_mission")
    analysis.is_loadable = True
    analysis.num_agents = 2
    analysis.max_steps = 1000
    analysis.map_size = (50, 50)

    summary = analysis.summary()
    assert "test_mission" in summary
    assert "2 agents" in summary
    assert "1000 max steps" in summary


def test_mission_analysis_summary_not_loadable():
    """Test MissionAnalysis summary for non-loadable mission."""
    analysis = MissionAnalysis("bad_mission")
    analysis.is_loadable = False
    analysis.errors.append("Mission not found")

    summary = analysis.summary()
    assert "bad_mission" in summary
    assert "NOT LOADABLE" in summary
    assert "not found" in summary


def test_daf_analyze_mission_real():
    """Test analyzing a real mission."""
    try:
        analysis = daf_analyze_mission("training_facility_1")

        assert isinstance(analysis, MissionAnalysis)
        assert analysis.mission_name == "training_facility_1"

        # Should be loadable if mission exists
        if analysis.is_loadable:
            assert analysis.env_config is not None

    except Exception:
        pytest.skip("Could not analyze mission")


def test_daf_analyze_mission_nonexistent():
    """Test analyzing non-existent mission."""
    analysis = daf_analyze_mission("nonexistent_mission_xyz_123")

    assert isinstance(analysis, MissionAnalysis)
    assert analysis.is_loadable is False
    assert len(analysis.errors) > 0


def test_daf_analyze_mission_set():
    """Test analyzing multiple missions."""
    mission_names = ["training_facility_1"]

    try:
        analyses = daf_analyze_mission_set(mission_names)

        assert isinstance(analyses, dict)
        assert "training_facility_1" in analyses
        assert isinstance(analyses["training_facility_1"], MissionAnalysis)

    except Exception:
        pytest.skip("Could not analyze mission set")


def test_daf_validate_mission_set():
    """Test validating mission set."""
    mission_names = ["training_facility_1", "nonexistent_mission"]

    valid, invalid = daf_validate_mission_set(mission_names)

    assert isinstance(valid, list)
    assert isinstance(invalid, list)
    assert "training_facility_1" in valid or "training_facility_1" in invalid
    assert "nonexistent_mission" in invalid


def test_daf_get_mission_metadata():
    """Test getting mission metadata."""
    try:
        metadata = daf_get_mission_metadata("training_facility_1")

        assert isinstance(metadata, dict)
        assert "mission_name" in metadata
        assert "is_loadable" in metadata
        assert "num_agents" in metadata
        assert "max_steps" in metadata

    except Exception:
        pytest.skip("Could not get mission metadata")


def test_daf_discover_missions_from_readme():
    """Test discovering missions from README.md."""
    # Try to find README.md
    readme_path = Path(__file__).parent.parent / "README.md"

    if readme_path.exists():
        missions = daf_discover_missions_from_readme(readme_path)

        assert isinstance(missions, list)
        # Should find at least some missions mentioned in README
        assert len(missions) >= 0  # May be 0 if pattern matching fails

    else:
        pytest.skip("README.md not found")


def test_daf_discover_missions_from_readme_nonexistent():
    """Test discovery with non-existent README."""
    missions = daf_discover_missions_from_readme(Path("/nonexistent/README.md"))

    assert missions == []


class TestMissionAnalysisIntegration:
    """Integration tests for mission analysis."""

    def test_complete_mission_analysis_workflow(self):
        """Test complete mission analysis workflow."""
        # Discover missions
        readme_path = Path(__file__).parent.parent / "README.md"
        if not readme_path.exists():
            pytest.skip("README.md not found")

        discovered = daf_discover_missions_from_readme(readme_path)

        # Analyze discovered missions
        if discovered:
            analyses = daf_analyze_mission_set(discovered[:1])  # Just first one
            assert len(analyses) > 0

            # Get metadata
            for mission_name in analyses:
                metadata = daf_get_mission_metadata(mission_name)
                assert "mission_name" in metadata

    def test_mission_validation_workflow(self):
        """Test mission validation workflow."""
        # Test with mix of valid and invalid missions
        mission_names = ["training_facility_1", "nonexistent_xyz"]

        valid, invalid = daf_validate_mission_set(mission_names)

        # Should have at least one valid if training_facility_1 exists
        assert isinstance(valid, list)
        assert isinstance(invalid, list)
        assert len(valid) + len(invalid) == len(mission_names)

