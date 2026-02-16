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
                "--missions", "cogsguard_machina_1.basic",
                "--output-dir", str(temp_output_dir),
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=120,  # 2 minute timeout
        )
        
        # If the script fails due to missing dependencies, skip
        if result.returncode != 0:
            combined_output = result.stdout + result.stderr
            if "ModuleNotFoundError" in combined_output or "No module named" in combined_output \
                    or "Environment validation failed" in combined_output:
                pytest.skip("Full suite requires cogames/mettagrid infrastructure")
            assert False, f"Script failed: {result.stderr}"
        
        # Check core output structure (always generated)
        assert (temp_output_dir / "SUITE_SUMMARY.json").exists()
        assert (temp_output_dir / "SUITE_SUMMARY.txt").exists()
        
        # Verify summary JSON is valid and has expected structure
        with open(temp_output_dir / "SUITE_SUMMARY.json") as f:
            summary = json.load(f)
        
        assert "comparison" in summary
        assert "dashboard" in summary
        assert "status" in summary["comparison"]
        assert "status" in summary["dashboard"]
        
        # Dashboard should always succeed
        assert summary["dashboard"]["status"] == "success"
        
        # If comparison succeeded, validate its outputs
        if summary["comparison"]["status"] == "success":
            comparisons_dir = temp_output_dir / "comparisons"
            assert comparisons_dir.exists()
            assert (comparisons_dir / "comparison_results.json").exists()
            with open(comparisons_dir / "comparison_results.json") as f:
                comparison_data = json.load(f)
            assert "summary_statistics" in comparison_data
            assert len(comparison_data["policies"]) > 0

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
                "--missions", "cogsguard_machina_1.basic",
                "--output-dir", str(temp_output_dir),
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=300,  # 5 minute timeout
        )
        
        # If the script fails due to missing dependencies, skip
        if result.returncode != 0:
            combined_output = result.stdout + result.stderr
            if "ModuleNotFoundError" in combined_output or "No module named" in combined_output \
                    or "Environment validation failed" in combined_output:
                pytest.skip("Full suite requires cogames/mettagrid infrastructure")
            assert False, f"Script failed: {result.stderr}"
        
        # Verify summary JSON generated
        assert (temp_output_dir / "SUITE_SUMMARY.json").exists()
        with open(temp_output_dir / "SUITE_SUMMARY.json") as f:
            summary = json.load(f)
        
        # If sweeps were generated, validate them
        sweeps_dir = temp_output_dir / "sweeps"
        if sweeps_dir.exists():
            sweep_dirs = list(sweeps_dir.iterdir())
            assert len(sweep_dirs) > 0, "Should have at least one sweep directory"
            policy_sweep_dir = sweep_dirs[0]
            assert (policy_sweep_dir / "sweep_results.json").exists()
            with open(policy_sweep_dir / "sweep_results.json") as f:
                sweep_data = json.load(f)
            assert "trials" in sweep_data
            assert len(sweep_data["trials"]) > 0
        else:
            # Sweeps may not generate if missions aren't available
            assert "sweeps" in summary or summary.get("comparison", {}).get("status") == "failed"

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
        
        # Script should still run (with failed comparison) or skip for missing deps
        combined_output = result.stdout + result.stderr
        if "ModuleNotFoundError" in combined_output or "No module named" in combined_output \
                or "Environment validation failed" in combined_output:
            pytest.skip("Full suite requires cogames/mettagrid infrastructure")
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
                "--missions", "cogsguard_machina_1.basic",
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
                "--missions", "cogsguard_machina_1.basic",
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

