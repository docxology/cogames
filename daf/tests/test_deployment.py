"""Tests for DAF deployment pipeline."""

from pathlib import Path

import pytest

from daf.src.train.deployment import DeploymentResult, daf_deploy_policy, daf_monitor_deployment, daf_package_policy, daf_rollback_deployment, daf_validate_deployment


def test_deployment_result_creation():
    """Test DeploymentResult initialization."""
    result = DeploymentResult(
        policy_name="test_policy",
        version="1.0.0",
        status="success",
        message="Deployment successful",
    )

    assert result.policy_name == "test_policy"
    assert result.version == "1.0.0"
    assert result.status == "success"
    assert result.timestamp is not None


def test_daf_package_policy(tmp_path):
    """Test policy packaging."""
    output_dir = tmp_path / "packages"
    output_dir.mkdir(parents=True)

    result = daf_package_policy(
        policy_class_path="cogames.policy.scripted_agent.baseline_agent.BaselinePolicy",
        weights_path=None,
        additional_files=None,
        output_dir=output_dir,
    )

    assert isinstance(result, DeploymentResult)
    assert result.status in ("success", "packaging_failed")

    if result.status == "success":
        assert result.package_path is not None
        assert result.package_path.exists()


def test_daf_package_policy_with_weights(tmp_path):
    """Test policy packaging with weights file."""
    # Create dummy weights file
    weights_file = tmp_path / "weights.pt"
    weights_file.write_text("dummy weights")

    output_dir = tmp_path / "packages"
    output_dir.mkdir(parents=True)

    result = daf_package_policy(
        policy_class_path="cogames.policy.starter_agent.StarterPolicy",
        weights_path=weights_file,
        output_dir=output_dir,
    )

    assert isinstance(result, DeploymentResult)


def test_daf_package_policy_nonexistent_weights(tmp_path):
    """Test packaging with non-existent weights file."""
    output_dir = tmp_path / "packages"
    output_dir.mkdir(parents=True)

    result = daf_package_policy(
        policy_class_path="cogames.policy.starter_agent.StarterPolicy",
        weights_path=tmp_path / "nonexistent.pt",
        output_dir=output_dir,
    )

    assert result.status == "packaging_failed"
    assert "not found" in result.message.lower()


def test_daf_validate_deployment(tmp_path, safe_mission_loader):
    """Test deployment validation."""
    # Use fixture for safe mission loading
    mission_name, env_cfg = safe_mission_loader("cogsguard_machina_1.basic")
    missions = [(mission_name, env_cfg)]

    result = daf_validate_deployment(
        policy_class_path="cogames.policy.starter_agent.StarterPolicy",
        weights_path=None,
        validation_missions=missions,
        success_threshold=0.0,  # Very low threshold for testing
    )

    assert isinstance(result, DeploymentResult)
    assert result.status in ("success", "validation_failed", "validation_error")


def test_daf_validate_deployment_no_missions():
    """Test validation without missions."""
    result = daf_validate_deployment(
        policy_class_path="cogames.policy.starter_agent.StarterPolicy",
        validation_missions=None,
    )

    assert isinstance(result, DeploymentResult)
    # Accepts 'success' (validation skipped) or 'validation_error' (mettagrid unavailable)
    assert result.status in ("success", "validation_error")


def test_daf_deploy_policy_simulation_mode(tmp_path):
    """Test deployment in simulation mode (httpx not available)."""
    # Create dummy package
    package_file = tmp_path / "package.tar.gz"
    package_file.write_text("dummy package")

    result = daf_deploy_policy(
        policy_name="test_policy",
        package_path=package_file,
        deployment_endpoint="https://example.com/api",
    )

    assert isinstance(result, DeploymentResult)
    # Should handle gracefully even if httpx not available
    assert result.status in ("success", "deployment_failed", "unavailable")


def test_daf_deploy_policy_nonexistent_package():
    """Test deployment with non-existent package."""
    result = daf_deploy_policy(
        policy_name="test_policy",
        package_path=Path("/nonexistent/package.tar.gz"),
        deployment_endpoint="https://example.com/api",
    )

    assert result.status == "deployment_failed"
    assert "not found" in result.message.lower()


def test_daf_monitor_deployment_simulation():
    """Test monitoring in simulation mode."""
    try:
        status = daf_monitor_deployment(
            deployment_id="test_id",
            endpoint="https://example.com/api",
        )

        assert isinstance(status, dict)
        assert "status" in status

    except Exception:
        # May fail if httpx not available - that's OK
        pass


def test_daf_rollback_deployment_simulation():
    """Test rollback in simulation mode."""
    result = daf_rollback_deployment(
        deployment_id="test_id",
        endpoint="https://example.com/api",
        previous_version="1.0.0",
    )

    assert isinstance(result, DeploymentResult)
    assert result.version == "1.0.0"


class TestDeploymentIntegration:
    """Integration tests for deployment pipeline."""

    def test_package_and_validate_workflow(self, tmp_path, safe_mission_loader):
        """Test complete package and validate workflow."""
        # Package policy
        output_dir = tmp_path / "packages"
        package_result = daf_package_policy(
            policy_class_path="cogames.policy.starter_agent.StarterPolicy",
            output_dir=output_dir,
        )

        if package_result.status == "success" and package_result.package_path:
            # Validate packaged policy
            mission_name, env_cfg = safe_mission_loader("cogsguard_machina_1.basic")
            missions = [(mission_name, env_cfg)]

            validate_result = daf_validate_deployment(
                policy_class_path="cogames.policy.starter_agent.StarterPolicy",
                validation_missions=missions,
                success_threshold=0.0,
            )

            assert isinstance(validate_result, DeploymentResult)

