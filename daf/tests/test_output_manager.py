"""Tests for DAF output manager and directory structure."""

import json
from pathlib import Path

import pytest

from daf.src.core.output_manager import (
    OutputDirectories,
    OutputManager,
    get_output_manager,
    reset_output_manager,
)


class TestOutputDirectories:
    """Tests for OutputDirectories dataclass."""

    def test_create_all_directories(self, tmp_path):
        """Verify all subdirectories are created."""
        dirs = OutputDirectories.create(tmp_path / "output")

        assert dirs.base_dir.exists()
        assert dirs.sweeps.exists()
        assert dirs.comparisons.exists()
        assert dirs.training.exists()
        assert dirs.deployment.exists()
        assert dirs.evaluations.exists()
        assert dirs.visualizations.exists()
        assert dirs.logs.exists()
        assert dirs.reports.exists()
        assert dirs.artifacts.exists()
        assert dirs.temp.exists()

    def test_directory_names(self, tmp_path):
        """Verify directory naming convention."""
        dirs = OutputDirectories.create(tmp_path / "output")

        assert dirs.sweeps.name == "sweeps"
        assert dirs.comparisons.name == "comparisons"
        assert dirs.training.name == "training"
        assert dirs.deployment.name == "deployment"
        assert dirs.evaluations.name == "evaluations"
        assert dirs.visualizations.name == "visualizations"
        assert dirs.logs.name == "logs"
        assert dirs.reports.name == "reports"
        assert dirs.artifacts.name == "artifacts"
        assert dirs.temp.name == ".temp"

    def test_idempotent_creation(self, tmp_path):
        """Verify calling create twice does not error."""
        dirs1 = OutputDirectories.create(tmp_path / "output")
        dirs2 = OutputDirectories.create(tmp_path / "output")

        assert dirs1.base_dir == dirs2.base_dir

    def test_nested_path_creation(self, tmp_path):
        """Verify deeply nested base_dir is created."""
        deep = tmp_path / "a" / "b" / "c" / "output"
        dirs = OutputDirectories.create(deep)

        assert dirs.base_dir.exists()
        assert dirs.sweeps.exists()


class TestOutputManager:
    """Tests for OutputManager class."""

    def test_init_creates_directories(self, tmp_path):
        """Verify __init__ creates directory structure."""
        mgr = OutputManager(base_dir=tmp_path / "daf_out", log_to_file=False)

        assert mgr.dirs.base_dir.exists()
        assert mgr.dirs.sweeps.exists()
        assert mgr.session_id  # Non-empty session ID

    def test_session_metadata_initialized(self, tmp_path):
        """Verify session metadata structure."""
        mgr = OutputManager(base_dir=tmp_path / "daf_out", log_to_file=False)

        assert "session_id" in mgr.session_metadata
        assert "created" in mgr.session_metadata
        assert "operations" in mgr.session_metadata
        assert isinstance(mgr.session_metadata["operations"], list)

    def test_get_operation_dir_valid(self, tmp_path):
        """Test getting directory for valid operations."""
        mgr = OutputManager(base_dir=tmp_path / "daf_out", log_to_file=False)

        for op in ["sweep", "comparison", "training", "deployment", "evaluation", "visualization"]:
            op_dir = mgr.get_operation_dir(op)
            assert op_dir.exists()

    def test_get_operation_dir_invalid(self, tmp_path):
        """Test ValueError for invalid operation type."""
        mgr = OutputManager(base_dir=tmp_path / "daf_out", log_to_file=False)

        with pytest.raises(ValueError, match="Unknown operation type"):
            mgr.get_operation_dir("nonexistent")

    def test_get_operation_dir_with_subdir(self, tmp_path):
        """Test subdirectory creation within operation."""
        mgr = OutputManager(base_dir=tmp_path / "daf_out", log_to_file=False)

        op_dir = mgr.get_operation_dir("sweep", subdir="trial_001")
        assert op_dir.exists()
        assert op_dir.name == "trial_001"

    def test_save_json_results_roundtrip(self, tmp_path):
        """Test JSON save and read roundtrip."""
        mgr = OutputManager(base_dir=tmp_path / "daf_out", log_to_file=False)
        data = {"accuracy": 0.95, "loss": 0.05, "epochs": 10}

        result_path = mgr.save_json_results(data, "sweep", "results")

        assert result_path.exists()
        assert result_path.suffix == ".json"

        with open(result_path) as f:
            loaded = json.load(f)

        assert loaded["accuracy"] == 0.95
        assert loaded["loss"] == 0.05

    def test_save_text_results(self, tmp_path):
        """Test text file saving."""
        mgr = OutputManager(base_dir=tmp_path / "daf_out", log_to_file=False)
        text = "Training completed successfully.\nFinal reward: 42.0"

        result_path = mgr.save_text_results(text, "training", "report")

        assert result_path.exists()
        assert result_path.suffix == ".txt"
        assert result_path.read_text() == text

    def test_save_summary_report(self, tmp_path):
        """Test summary report with enhanced metadata."""
        mgr = OutputManager(base_dir=tmp_path / "daf_out", log_to_file=False)
        summary = {"status": "success", "total_trials": 50, "best_score": 0.99}

        report_path = mgr.save_summary_report("sweep", summary)

        assert report_path.exists()

        with open(report_path) as f:
            data = json.load(f)

        assert data["operation"] == "sweep"
        assert data["session_id"] == mgr.session_id
        assert data["status"] == "success"
        assert "timestamp" in data

    def test_save_session_metadata(self, tmp_path):
        """Test session metadata persistence."""
        mgr = OutputManager(base_dir=tmp_path / "daf_out", log_to_file=False)

        # Log an operation
        mgr.log_operation_start("sweep", {"name": "test_sweep"})
        mgr.log_operation_complete("sweep", status="success")

        meta_path = mgr.save_session_metadata()

        assert meta_path.exists()

        with open(meta_path) as f:
            data = json.load(f)

        assert data["session_id"] == mgr.session_id
        assert len(data["operations"]) >= 1

    def test_log_operation_start_and_complete(self, tmp_path):
        """Test operation lifecycle logging."""
        mgr = OutputManager(base_dir=tmp_path / "daf_out", log_to_file=False)

        mgr.log_operation_start("comparison", {"policies": 3})
        mgr.log_operation_complete("comparison", status="success", details={"time": 5.2})

        assert len(mgr.session_metadata["operations"]) == 1
        assert mgr.session_metadata["operations"][0]["operation"] == "comparison"

    def test_get_output_structure_info(self, tmp_path):
        """Test output structure info string."""
        mgr = OutputManager(base_dir=tmp_path / "daf_out", log_to_file=False)

        info = mgr.get_output_structure_info()

        assert "DAF Output Structure" in info
        assert mgr.session_id in info
        assert "Sweeps" in info
        assert "Comparisons" in info

    def test_cleanup_temp_files(self, tmp_path):
        """Test temp directory cleanup."""
        mgr = OutputManager(base_dir=tmp_path / "daf_out", log_to_file=False)

        # Create a temp file
        temp_file = mgr.dirs.temp / "scratch.txt"
        temp_file.write_text("temporary data")
        assert temp_file.exists()

        mgr.cleanup_temp_files()

        assert not mgr.dirs.temp.exists()


class TestGetOutputManager:
    """Tests for singleton get_output_manager."""

    def test_singleton_pattern(self, tmp_path):
        """Test that get_output_manager returns same instance."""
        reset_output_manager()

        mgr1 = get_output_manager(base_dir=tmp_path / "singleton", log_to_file=False)
        mgr2 = get_output_manager(base_dir=tmp_path / "singleton", log_to_file=False)

        assert mgr1 is mgr2

        reset_output_manager()

    def test_reset_clears_instance(self, tmp_path):
        """Test that reset creates fresh instance."""
        reset_output_manager()

        mgr1 = get_output_manager(base_dir=tmp_path / "reset1", log_to_file=False)
        reset_output_manager()
        mgr2 = get_output_manager(base_dir=tmp_path / "reset2", log_to_file=False)

        assert mgr1 is not mgr2

        reset_output_manager()
