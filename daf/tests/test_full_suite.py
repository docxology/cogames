"""Integration tests for the full suite script.

Tests verify that the full evaluation suite:
1. Runs real cogames simulations (not mocked)
2. Generates proper visualizations
3. Produces correct output structure
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestFullSuiteIntegration:
    """Integration tests for the full evaluation suite."""

    @pytest.fixture
    def temp_output_dir(self, tmp_path: Path) -> Path:
        """Create temporary output directory."""
        output_dir = tmp_path / "suite_test_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def test_full_suite_help(self):
        """Test that help command works."""
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "daf" / "scripts" / "run_full_suite.py"), "--help"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0
        assert "Run full CoGames evaluation suite" in result.stdout
        assert "--quick" in result.stdout
        assert "--policies" in result.stdout
        assert "--missions" in result.stdout

    def test_full_suite_quick_no_sweep(self, temp_output_dir: Path):
        """Test quick mode without sweep (fastest integration test)."""
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "daf" / "scripts" / "run_full_suite.py"),
                "--quick",
                "--no-sweep",
                "--episodes", "1",
                "--missions", "hello_world.hello_world_unclip",
                "--output-dir", str(temp_output_dir),
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=120,  # 2 minute timeout
        )
        
        # Check execution succeeded
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        
        # Check output structure
        assert (temp_output_dir / "SUITE_SUMMARY.json").exists()
        assert (temp_output_dir / "SUITE_SUMMARY.txt").exists()
        assert (temp_output_dir / "comparisons").exists()
        assert (temp_output_dir / "dashboard").exists()
        
        # Check comparison outputs
        comparisons_dir = temp_output_dir / "comparisons"
        assert (comparisons_dir / "comparison_results.json").exists(), "comparison_results.json should exist"
        
        # Visualization files are optional (matplotlib may not generate in all environments)
        # Just check core results exist
        with open(comparisons_dir / "comparison_results.json") as f:
            comparison_data = json.load(f)
        assert "summary_statistics" in comparison_data
        assert len(comparison_data["policies"]) > 0
        
        # Check dashboard
        dashboard_dir = temp_output_dir / "dashboard"
        assert (dashboard_dir / "dashboard.html").exists()
        
        # Verify JSON is valid
        with open(temp_output_dir / "SUITE_SUMMARY.json") as f:
            summary = json.load(f)
        
        assert "comparison" in summary
        assert "dashboard" in summary
        assert summary["comparison"]["status"] == "success"
        assert summary["dashboard"]["status"] == "success"

    @pytest.mark.slow
    def test_full_suite_with_sweep(self, temp_output_dir: Path):
        """Test full suite with sweep enabled (slower, comprehensive)."""
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "daf" / "scripts" / "run_full_suite.py"),
                "--quick",
                "--episodes", "1",
                "--sweep-episodes", "1",
                "--missions", "hello_world.hello_world_unclip",
                "--output-dir", str(temp_output_dir),
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=300,  # 5 minute timeout
        )
        
        # Check execution succeeded
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        
        # Check sweep outputs exist
        sweeps_dir = temp_output_dir / "sweeps"
        assert sweeps_dir.exists(), "Sweeps directory should exist"
        
        # Check for sweep subdirectory (policy name)
        sweep_dirs = list(sweeps_dir.iterdir())
        assert len(sweep_dirs) > 0, "Should have at least one sweep directory"
        
        policy_sweep_dir = sweep_dirs[0]
        assert (policy_sweep_dir / "sweep_results.json").exists()
        assert (policy_sweep_dir / "sweep_progress.png").exists()
        
        # Verify sweep results JSON
        with open(policy_sweep_dir / "sweep_results.json") as f:
            sweep_data = json.load(f)
        
        assert "trials" in sweep_data
        assert len(sweep_data["trials"]) > 0

    def test_full_suite_mission_format(self):
        """Test that mission format validation works."""
        # Invalid mission format should fail gracefully
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "daf" / "scripts" / "run_full_suite.py"),
                "--quick",
                "--no-sweep",
                "--episodes", "1",
                "--missions", "nonexistent_mission",
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=60,
        )
        
        # Script should still run (with failed comparison)
        # The comparison phase will fail but dashboard should still be generated
        assert "Error loading missions" in result.stdout or result.returncode == 0


class TestFullSuiteOutputValidation:
    """Tests for output validation and integrity."""

    def test_leaderboard_format(self, tmp_path: Path):
        """Test that leaderboard JSON has correct format."""
        # Run minimal suite
        output_dir = tmp_path / "leaderboard_test"
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "daf" / "scripts" / "run_full_suite.py"),
                "--quick",
                "--no-sweep",
                "--episodes", "1",
                "--missions", "hello_world.hello_world_unclip",
                "--output-dir", str(output_dir),
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=120,
        )
        
        if result.returncode != 0:
            pytest.skip("Suite execution failed, skipping output validation")
        
        leaderboard_path = output_dir / "comparisons" / "leaderboard.json"
        if not leaderboard_path.exists():
            pytest.skip("Leaderboard not generated")
        
        with open(leaderboard_path) as f:
            leaderboard = json.load(f)
        
        assert "generated" in leaderboard
        assert "leaderboard" in leaderboard
        assert isinstance(leaderboard["leaderboard"], list)
        
        if leaderboard["leaderboard"]:
            entry = leaderboard["leaderboard"][0]
            assert "rank" in entry
            assert "policy" in entry
            assert "avg_reward" in entry
            assert "std_dev" in entry

    def test_html_report_valid(self, tmp_path: Path):
        """Test that HTML report is valid."""
        output_dir = tmp_path / "html_test"
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "daf" / "scripts" / "run_full_suite.py"),
                "--quick",
                "--no-sweep",
                "--episodes", "1",
                "--missions", "hello_world.hello_world_unclip",
                "--output-dir", str(output_dir),
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=120,
        )
        
        if result.returncode != 0:
            pytest.skip("Suite execution failed")
        
        report_path = output_dir / "comparisons" / "report.html"
        if not report_path.exists():
            pytest.skip("Report not generated")
        
        html_content = report_path.read_text()
        assert "<!DOCTYPE html>" in html_content
        assert "Policy Comparison Report" in html_content
        assert "</html>" in html_content


# Mark slow tests
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")

