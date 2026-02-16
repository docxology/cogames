"""Tests for DAF environment checks."""

from pathlib import Path

import pytest


from daf.src.environment_checks import (
    EnvironmentCheckResult,
    daf_check_cuda_availability,
    daf_check_dependencies,
    daf_check_disk_space,
    daf_check_environment,
    daf_get_recommended_device,
)


def test_environment_check_result():
    """Test EnvironmentCheckResult initialization and methods."""
    result = EnvironmentCheckResult()

    # Add some checks
    result.add_check("check1", True)
    result.add_check("check2", False, warning="Test warning")
    result.add_error("check3", "Test error")

    assert len(result.checks) == 3
    assert result.checks["check1"] is True
    assert result.checks["check2"] is False
    assert result.checks["check3"] is False
    assert len(result.warnings) == 1
    assert len(result.errors) == 1
    assert not result.is_healthy()


def test_environment_check_result_is_healthy():
    """Test healthy environment detection."""
    result = EnvironmentCheckResult()
    result.add_check("check1", True)
    result.add_check("check2", True)

    assert result.is_healthy()

    result.add_error("check3", "error")
    assert not result.is_healthy()


def test_environment_check_result_summary():
    """Test summary string generation."""
    result = EnvironmentCheckResult()
    result.add_check("check1", True)
    result.add_check("check2", True)

    summary = result.summary()
    assert "HEALTHY" in summary
    assert "2/2" in summary


def test_daf_check_cuda_availability():
    """Test GPU availability check (CUDA on Linux, MPS on macOS)."""
    result = daf_check_cuda_availability()

    assert isinstance(result, EnvironmentCheckResult)
    # On macOS the backward-compat alias routes to MPS checks
    gpu_checks = {"CUDA Available", "CUDA Backend", "MPS Available", "MPS Backend"}
    assert gpu_checks & set(result.checks), f"Expected GPU checks, got: {result.checks}"

    # If CUDA available, should have device info
    if result.checks.get("CUDA Available", False):
        assert "device_count" in result.info


def test_daf_check_dependencies():
    """Test dependency checking."""
    result = daf_check_dependencies()

    assert isinstance(result, EnvironmentCheckResult)
    # Should have checked required packages
    assert any("Required:" in check for check in result.checks)

    # All required packages should be available
    required_ok = all(
        result.checks[check] for check in result.checks if "Required:" in check
    )
    assert required_ok


def test_daf_check_disk_space(tmp_path):
    """Test disk space checking."""
    result = daf_check_disk_space(
        checkpoint_dir=tmp_path,
        min_available_gb=0.001,  # Very small requirement
    )

    assert isinstance(result, EnvironmentCheckResult)
    # Should have at least 1KB available
    assert "Disk Space" in result.checks


def test_daf_check_disk_space_insufficient(tmp_path):
    """Test disk space check with impossible requirement."""
    result = daf_check_disk_space(
        checkpoint_dir=tmp_path,
        min_available_gb=10000.0,  # Impossible to have 10TB available
    )

    assert isinstance(result, EnvironmentCheckResult)
    # Should fail due to insufficient space (unless system has 10TB)
    # Most systems will fail this check
    disk_check = result.checks.get("Disk Space")
    if disk_check is False:
        assert "Low disk space" in "\n".join(result.warnings)


def test_daf_get_recommended_device():
    """Test recommended device resolution."""
    device = daf_get_recommended_device()

    import torch
    assert isinstance(device, torch.device)
    assert device.type in ("cpu", "cuda", "mps")


def test_daf_check_environment_all_checks(tmp_path):
    """Test comprehensive environment check."""
    result = daf_check_environment(
        checkout_dir=tmp_path,
        missions=None,  # Skip mission checks
        check_cuda=True,
        check_disk=True,
        check_dependencies=True,
        check_missions=False,
    )

    assert isinstance(result, EnvironmentCheckResult)
    assert len(result.checks) > 0


def test_daf_check_environment_selective(tmp_path):
    """Test selective environment checks."""
    # Only check CUDA
    result = daf_check_environment(
        check_cuda=True,
        check_disk=False,
        check_dependencies=False,
        check_missions=False,
    )

    # Should have minimal checks
    assert len(result.checks) <= 5  # Only CUDA-related checks


def test_daf_check_environment_no_checks():
    """Test with all checks disabled."""
    result = daf_check_environment(
        check_cuda=False,
        check_disk=False,
        check_dependencies=False,
        check_missions=False,
    )

    assert len(result.checks) == 0
    assert result.is_healthy()


class TestEnvironmentCheckIntegration:
    """Integration tests for environment checks."""

    def test_full_environment_validation(self, tmp_path):
        """Test full environment validation workflow."""
        # This mimics what a user would do before training
        result = daf_check_environment(
            checkout_dir=tmp_path,
            missions=None,
            check_cuda=True,
            check_disk=True,
            check_dependencies=True,
        )

        # Should complete without exceptions
        assert result is not None
        assert isinstance(result, EnvironmentCheckResult)

    def test_environment_suitable_for_training(self):
        """Test that environment is suitable for training."""
        # Run all checks
        result = daf_check_environment(
            check_cuda=True,
            check_disk=True,
            check_dependencies=True,
            check_missions=False,
        )

        # Required packages should be available
        required_checks = [c for c in result.checks if "Required:" in c]
        required_ok = all(result.checks[c] for c in required_checks)
        assert required_ok, f"Missing required packages: {result.errors}"

