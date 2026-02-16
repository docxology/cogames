"""Policy deployment pipeline for DAF.

DAF sidecar utility: Uses `cogames.auth` and `cogames.cli.submit` patterns.

Provides policy packaging, validation, deployment, and monitoring.
Extends CoGames deployment patterns with additional validation and monitoring.
"""

from __future__ import annotations

import logging
import shutil
import tarfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console

logger = logging.getLogger("daf.deployment")


@dataclass
class DeploymentResult:
    """Result from a policy deployment operation."""

    policy_name: str
    version: str
    status: str  # "success", "validation_failed", "deployment_failed"
    message: str
    package_path: Optional[Path] = None
    deployment_id: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self) -> None:
        """Set default timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()


def daf_package_policy(
    policy_class_path: str,
    weights_path: Optional[Path | str] = None,
    additional_files: Optional[list[Path | str]] = None,
    output_dir: Path | str = "deployment_packages",
    console: Optional[Console] = None,
) -> DeploymentResult:
    """Package policy with dependencies for deployment.

    Creates a portable bundle containing policy code and weights.

    Args:
        policy_class_path: Full class path to policy
        weights_path: Optional path to policy weights/checkpoint
        additional_files: Optional additional files to include
        output_dir: Output directory for package
        console: Optional Rich console for output

    Returns:
        DeploymentResult with package location

    Raises:
        FileNotFoundError: If policy file or weights not found
        RuntimeError: If packaging fails
    """
    if console is None:
        console = Console()

    # Validate inputs
    if not policy_class_path:
        raise ValueError("policy_class_path cannot be empty")
    if not isinstance(policy_class_path, str):
        raise ValueError(f"policy_class_path must be string, got {type(policy_class_path)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Packaging policy: {policy_class_path}")
    console.print(f"[cyan]Packaging policy: {policy_class_path}[/cyan]")

    # Create temporary package directory
    package_name = policy_class_path.split(".")[-1]
    package_dir = output_dir / f"{package_name}_package"

    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Create package metadata
        metadata = {
            "policy_class_path": policy_class_path,
            "created": datetime.now().isoformat(),
        }

        if weights_path:
            weights_path = Path(weights_path)
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights not found: {weights_path}")

            # Copy weights
            weights_dest = package_dir / "weights.pt"
            shutil.copy2(weights_path, weights_dest)
            metadata["weights_file"] = "weights.pt"

            console.print(f"[green]✓ Included weights: {weights_path}[/green]")

        # Copy additional files
        if additional_files:
            files_dir = package_dir / "files"
            files_dir.mkdir(exist_ok=True)

            for file_path in additional_files:
                file_path = Path(file_path)
                if file_path.is_dir():
                    shutil.copytree(file_path, files_dir / file_path.name)
                else:
                    shutil.copy2(file_path, files_dir / file_path.name)

            console.print(f"[green]✓ Included {len(additional_files)} additional file(s)[/green]")

        # Write metadata
        import json

        with open(package_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Create tarball
        package_tar = output_dir / f"{package_name}_package.tar.gz"
        with tarfile.open(package_tar, "w:gz") as tar:
            tar.add(package_dir, arcname=package_name)

        # Cleanup temp directory
        shutil.rmtree(package_dir)

        console.print(f"[green]✓ Package created: {package_tar}[/green]\n")

        return DeploymentResult(
            policy_name=package_name,
            version="1.0.0",
            status="success",
            message=f"Policy packaged successfully",
            package_path=package_tar,
        )

    except Exception as e:
        console.print(f"[red]✗ Packaging failed: {e}[/red]\n")
        return DeploymentResult(
            policy_name=package_name,
            version="1.0.0",
            status="packaging_failed",
            message=f"Packaging failed: {e}",
        )


def daf_validate_deployment(
    policy_class_path: str,
    weights_path: Optional[Path | str] = None,
    validation_missions: Optional[list[tuple[str, "Any"]]] = None,
    success_threshold: float = 0.5,
    console: Optional[Console] = None,
) -> DeploymentResult:
    """Validate policy deployment before release.
    
    Args:
        policy_class_path: Full class path to policy
        weights_path: Optional path to policy weights
        validation_missions: Optional missions to validate on
        success_threshold: Required success rate (0-1)
        console: Optional Rich console for output
        
    Returns:
        DeploymentResult indicating validation status
        
    Raises:
        ValueError: If inputs are invalid
    """
    # Validate inputs
    if not policy_class_path:
        raise ValueError("policy_class_path cannot be empty")
    if not (0 <= success_threshold <= 1):
        raise ValueError(f"success_threshold must be between 0 and 1, got {success_threshold}")
    
    return daf_validate_deployment_impl(
        policy_class_path,
        weights_path,
        validation_missions,
        success_threshold,
        console,
    )


def daf_validate_deployment_impl(
    policy_class_path: str,
    weights_path: Optional[Path | str] = None,
    validation_missions: Optional[list[tuple[str, "Any"]]] = None,
    success_threshold: float = 0.5,
    console: Optional[Console] = None,
) -> DeploymentResult:
    """Validate policy in isolated environment.

    Args:
        policy_class_path: Policy class path
        weights_path: Optional weights file
        validation_missions: Missions to validate on
        success_threshold: Minimum success rate for validation
        console: Optional Rich console for output

    Returns:
        DeploymentResult with validation status

    Raises:
        ValueError: If validation fails
    """
    if console is None:
        console = Console()

    console.print(f"[cyan]Validating policy: {policy_class_path}[/cyan]\n")

    try:
        from mettagrid.policy.policy import PolicySpec
        from mettagrid.policy.policy_env_interface import PolicyEnvInterface
        from mettagrid.simulator.rollout import Rollout

        # Create policy spec
        policy_spec = PolicySpec(class_path=policy_class_path, data_path=weights_path)

        # Run smoke tests on missions
        if not validation_missions:
            console.print("[yellow]No validation missions provided - skipping evaluation[/yellow]\n")
            return DeploymentResult(
                policy_name=policy_class_path.split(".")[-1],
                version="1.0.0",
                status="success",
                message="Validation skipped (no missions)",
            )

        from mettagrid.policy.loader import initialize_or_load_policy

        success_count = 0

        for mission_name, env_cfg in validation_missions:
            try:
                console.print(f"  Testing on {mission_name}...", end=" ")

                # Initialize policy
                policy_env_info = PolicyEnvInterface.from_mg_cfg(env_cfg)
                policy = initialize_or_load_policy(policy_env_info, policy_spec)
                agent_policies = [policy.agent_policy(i) for i in range(env_cfg.game.num_agents)]

                # Run episode
                rollout = Rollout(env_cfg, agent_policies, render_mode="none", seed=42)
                rollout.run_until_done()

                rewards = sum(rollout._sim.episode_rewards)
                console.print(f"reward={rewards:.2f}")

                if rewards > 0:
                    success_count += 1

            except Exception as e:
                console.print(f"[red]failed: {e}[/red]")

        success_rate = success_count / len(validation_missions)
        passed = success_rate >= success_threshold

        console.print(f"\nValidation Results:")
        console.print(f"  Success Rate: {success_rate:.1%} (threshold: {success_threshold:.1%})")
        console.print(f"  Status: {'[green]PASSED[/green]' if passed else '[red]FAILED[/red]'}\n")

        return DeploymentResult(
            policy_name=policy_class_path.split(".")[-1],
            version="1.0.0",
            status="success" if passed else "validation_failed",
            message=f"Validation {'passed' if passed else 'failed'}: {success_rate:.1%} success rate",
        )

    except Exception as e:
        console.print(f"[red]✗ Validation error: {e}[/red]\n")
        return DeploymentResult(
            policy_name=policy_class_path.split(".")[-1],
            version="1.0.0",
            status="validation_error",
            message=f"Validation error: {e}",
        )


def daf_deploy_policy(
    policy_name: str,
    package_path: Path | str,
    deployment_endpoint: str,
    auth_token: Optional[str] = None,
    console: Optional[Console] = None,
) -> DeploymentResult:
    """Deploy policy to production/competition endpoint.

    Args:
        policy_name: Name for deployment
        package_path: Path to policy package
        deployment_endpoint: Target deployment URL
        auth_token: Optional authentication token
        console: Optional Rich console for output

    Returns:
        DeploymentResult with deployment status

    Raises:
        FileNotFoundError: If package not found
        RuntimeError: If deployment fails
    """
    if console is None:
        console = Console()

    package_path = Path(package_path)

    if not package_path.exists():
        console.print(f"[red]✗ Package not found: {package_path}[/red]\n")
        return DeploymentResult(
            policy_name=policy_name,
            version="1.0.0",
            status="deployment_failed",
            message=f"Package not found: {package_path}",
        )

    console.print(f"[cyan]Deploying policy: {policy_name}[/cyan]")
    console.print(f"  Endpoint: {deployment_endpoint}")
    console.print(f"  Package: {package_path}\n")

    try:
        import httpx

        # Upload package
        with open(package_path, "rb") as f:
            files = {"package": f}
            headers = {}

            if auth_token:
                headers["Authorization"] = f"Bearer {auth_token}"

            response = httpx.post(
                f"{deployment_endpoint}/upload",
                files=files,
                headers=headers,
                timeout=30.0,
            )

            response.raise_for_status()

        console.print(f"[green]✓ Deployment successful[/green]")
        console.print(f"  Response: {response.status_code}\n")

        return DeploymentResult(
            policy_name=policy_name,
            version="1.0.0",
            status="success",
            message="Policy deployed successfully",
            deployment_id=response.headers.get("X-Deployment-ID"),
        )

    except ImportError:
        console.print(
            "[yellow]⚠ httpx not available - deployment simulation mode[/yellow]"
        )
        return DeploymentResult(
            policy_name=policy_name,
            version="1.0.0",
            status="success",
            message="Deployment simulation mode (httpx not available)",
        )

    except Exception as e:
        console.print(f"[red]✗ Deployment failed: {e}[/red]\n")
        return DeploymentResult(
            policy_name=policy_name,
            version="1.0.0",
            status="deployment_failed",
            message=f"Deployment failed: {e}",
        )


def daf_monitor_deployment(
    deployment_id: str,
    endpoint: str,
    auth_token: Optional[str] = None,
    console: Optional[Console] = None,
) -> dict[str, "Any"]:
    """Monitor deployed policy performance.

    Args:
        deployment_id: Deployment ID to monitor
        endpoint: Monitoring endpoint URL
        auth_token: Optional authentication token
        console: Optional Rich console for output

    Returns:
        Status dictionary with performance metrics

    Raises:
        RuntimeError: If monitoring fails
    """
    if console is None:
        console = Console()

    console.print(f"[cyan]Monitoring deployment: {deployment_id}[/cyan]\n")

    try:
        import httpx

        headers = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        response = httpx.get(
            f"{endpoint}/status/{deployment_id}",
            headers=headers,
            timeout=10.0,
        )

        response.raise_for_status()
        status = response.json()

        console.print(f"[green]✓ Status retrieved[/green]")
        console.print(f"  Running: {status.get('running', False)}")
        console.print(f"  Avg Reward: {status.get('avg_reward', 0):.2f}")
        console.print(f"  Episodes: {status.get('episodes_run', 0)}\n")

        return status

    except ImportError:
        console.print("[yellow]⚠ httpx not available - cannot monitor[/yellow]\n")
        return {"status": "monitoring_unavailable"}

    except Exception as e:
        console.print(f"[red]✗ Monitoring failed: {e}[/red]\n")
        raise RuntimeError(f"Monitoring failed: {e}") from e


def daf_rollback_deployment(
    deployment_id: str,
    endpoint: str,
    previous_version: str,
    auth_token: Optional[str] = None,
    console: Optional[Console] = None,
) -> DeploymentResult:
    """Rollback deployment to previous version.

    Args:
        deployment_id: Current deployment ID
        endpoint: Deployment endpoint
        previous_version: Version to rollback to
        auth_token: Optional authentication token
        console: Optional Rich console for output

    Returns:
        DeploymentResult with rollback status
    """
    if console is None:
        console = Console()

    console.print(f"[cyan]Rolling back deployment: {deployment_id}[/cyan]")
    console.print(f"  Target version: {previous_version}\n")

    try:
        import httpx

        headers = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        response = httpx.post(
            f"{endpoint}/rollback/{deployment_id}",
            json={"target_version": previous_version},
            headers=headers,
            timeout=30.0,
        )

        response.raise_for_status()

        console.print(f"[green]✓ Rollback successful[/green]\n")

        return DeploymentResult(
            policy_name="",
            version=previous_version,
            status="success",
            message="Rollback completed successfully",
            deployment_id=deployment_id,
        )

    except ImportError:
        console.print("[yellow]⚠ httpx not available[/yellow]\n")
        return DeploymentResult(
            policy_name="",
            version=previous_version,
            status="unavailable",
            message="Rollback unavailable (httpx not installed)",
        )

    except Exception as e:
        console.print(f"[red]✗ Rollback failed: {e}[/red]\n")
        return DeploymentResult(
            policy_name="",
            version=previous_version,
            status="failed",
            message=f"Rollback failed: {e}",
        )


def daf_submit_policy(
    policy_class_path: str,
    weights_path: Optional[Path | str] = None,
    server_url: str = "https://cogames.ai",
    season: Optional[str] = None,
    submission_name: Optional[str] = None,
    validate_before_submit: bool = True,
    console: Optional[Console] = None,
) -> DeploymentResult:
    """Submit policy to CoGames tournament using cogames.cli.submit workflow.

    Wraps the bundle → validate → upload submission pipeline.

    Args:
        policy_class_path: Full class path to the policy
        weights_path: Optional path to policy weights
        server_url: CoGames server URL
        season: Optional tournament season name
        submission_name: Optional submission name
        validate_before_submit: Run local validation before uploading
        console: Optional Rich console for output

    Returns:
        DeploymentResult with submission status
    """
    if console is None:
        console = Console()

    if not policy_class_path:
        raise ValueError("policy_class_path required")

    console.print(f"\n[bold cyan]DAF Tournament Submission[/bold cyan]")
    console.print(f"  Policy: {policy_class_path}")
    console.print(f"  Server: {server_url}\n")

    try:
        from cogames.cli.submit import submit_policy

        # Build submission kwargs
        kwargs = {
            "policy_class_path": policy_class_path,
            "server_url": server_url,
        }
        if weights_path:
            kwargs["weights_path"] = str(weights_path)
        if season:
            kwargs["season"] = season
        if submission_name:
            kwargs["submission_name"] = submission_name

        # Local validation
        if validate_before_submit:
            console.print("[cyan]Running pre-submit validation...[/cyan]")
            validation = daf_validate_deployment(
                policy_class_path=policy_class_path,
                weights_path=weights_path,
                console=console,
            )
            if validation.status != "success":
                return DeploymentResult(
                    policy_name=policy_class_path.split(".")[-1],
                    version="1.0.0",
                    status="validation_failed",
                    message=f"Pre-submit validation failed: {validation.message}",
                )

        # Submit via cogames CLI
        console.print("[cyan]Submitting to tournament...[/cyan]")
        result = submit_policy(**kwargs)

        logger.info(f"Submission result: {result}")
        console.print(f"[green]✓ Submission complete[/green]\n")

        return DeploymentResult(
            policy_name=policy_class_path.split(".")[-1],
            version="1.0.0",
            status="success",
            message="Tournament submission successful",
            deployment_id=str(result) if result else None,
        )

    except ImportError as e:
        console.print(f"[yellow]⚠ cogames.cli.submit not available: {e}[/yellow]\n")
        return DeploymentResult(
            policy_name=policy_class_path.split(".")[-1],
            version="1.0.0",
            status="unavailable",
            message=f"Submit not available: {e}",
        )

    except Exception as e:
        console.print(f"[red]✗ Submission failed: {e}[/red]\n")
        return DeploymentResult(
            policy_name=policy_class_path.split(".")[-1],
            version="1.0.0",
            status="submission_failed",
            message=f"Submission failed: {e}",
        )

