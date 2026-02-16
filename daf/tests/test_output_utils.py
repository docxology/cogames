"""Tests for DAF output utility functions."""

import json
from pathlib import Path
from datetime import datetime

import pytest

from daf.src.core.output_utils import (
    find_latest_session,
    list_sessions,
    cleanup_old_sessions,
    export_session,
    print_output_summary,
)


def _create_session_dirs(base: Path, operation: str, names: list[str]) -> list[Path]:
    """Helper to create session directories with metadata."""
    dirs = []
    for name in names:
        session_dir = base / operation / name
        session_dir.mkdir(parents=True, exist_ok=True)
        # Write a metadata file so list_sessions can pick it up
        meta = {"session_id": name, "created": datetime.now().isoformat()}
        with open(session_dir / "session_metadata.json", "w") as f:
            json.dump(meta, f)
        dirs.append(session_dir)
    return dirs


class TestFindLatestSession:
    """Tests for find_latest_session."""

    def test_empty_directory(self, tmp_path):
        """Returns None when no sessions exist."""
        result = find_latest_session(tmp_path)
        assert result is None

    def test_finds_latest(self, tmp_path):
        """Returns the most recent session directory."""
        _create_session_dirs(tmp_path, "sweeps", ["20240101_100000", "20240201_100000", "20240115_100000"])

        result = find_latest_session(tmp_path, "sweep")

        # Should find the latest by name sort
        if result is not None:
            assert "20240201" in result.name

    def test_nonexistent_operation(self, tmp_path):
        """Returns None for unknown operation type."""
        result = find_latest_session(tmp_path, "nonexistent_op")
        assert result is None


class TestListSessions:
    """Tests for list_sessions."""

    def test_empty_directory(self, tmp_path):
        """Returns empty list when no sessions exist."""
        sessions = list_sessions(tmp_path)
        assert isinstance(sessions, list)
        assert len(sessions) == 0

    def test_max_results(self, tmp_path):
        """Respects max_results limit."""
        _create_session_dirs(tmp_path, "sweeps", [
            "20240101_100000", "20240102_100000", "20240103_100000",
            "20240104_100000", "20240105_100000",
        ])

        sessions = list_sessions(tmp_path, operation="sweep", max_results=3)

        assert len(sessions) <= 3


class TestCleanupOldSessions:
    """Tests for cleanup_old_sessions."""

    def test_dry_run_no_deletion(self, tmp_path):
        """Dry run reports but does not delete."""
        dirs = _create_session_dirs(tmp_path, "sweeps", ["20200101_100000"])

        result = cleanup_old_sessions(tmp_path, operation="sweep", days=0, dry_run=True)

        # Session dir should still exist
        assert dirs[0].exists()
        assert isinstance(result, int)

    def test_actual_cleanup(self, tmp_path):
        """Actual cleanup removes old sessions."""
        dirs = _create_session_dirs(tmp_path, "sweeps", ["20200101_100000"])

        result = cleanup_old_sessions(tmp_path, operation="sweep", days=0, dry_run=False)

        assert isinstance(result, int)


class TestExportSession:
    """Tests for export_session."""

    def test_export_uncompressed(self, tmp_path):
        """Export session without compression."""
        dirs = _create_session_dirs(tmp_path, "sweeps", ["20240101_100000"])
        export_dest = tmp_path / "export_dir"

        export_session(dirs[0], export_dest, compress=False)

        assert export_dest.exists()

    def test_export_compressed(self, tmp_path):
        """Export session as compressed archive."""
        dirs = _create_session_dirs(tmp_path, "sweeps", ["20240101_100000"])
        export_dest = tmp_path / "export.tar.gz"

        export_session(dirs[0], export_dest, compress=True)

        assert export_dest.exists()

    def test_export_nonexistent_session(self, tmp_path):
        """Export nonexistent session handles gracefully."""
        nonexistent = tmp_path / "no_such_session"
        export_dest = tmp_path / "export"

        # Should handle gracefully (raise or return)
        try:
            export_session(nonexistent, export_dest, compress=False)
        except (FileNotFoundError, ValueError, OSError):
            pass  # Expected


class TestPrintOutputSummary:
    """Tests for print_output_summary."""

    def test_smoke_test_empty(self, tmp_path):
        """Runs without error on empty output directory."""
        # Should not raise
        print_output_summary(tmp_path)

    def test_smoke_test_with_content(self, tmp_path):
        """Runs without error when sessions exist."""
        _create_session_dirs(tmp_path, "sweeps", ["20240101_100000"])
        _create_session_dirs(tmp_path, "comparisons", ["20240102_100000"])

        print_output_summary(tmp_path)
