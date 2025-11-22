"""Environment validation utilities for DAF.

DAF sidecar utility: Uses `cogames.device.resolve_training_device()` for device resolution.

Provides comprehensive checks before training, sweeps, and deployments.
Extends CoGames device resolution with additional environment validation.
"""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path
from typing import Optional

from rich.console import Console

import torch

# Check for optional dependencies
try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class EnvironmentCheckResult:
    """Results from environment validation checks."""

    def __init__(self):
        """Initialize check result container."""
        self.checks: dict[str, bool] = {}
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.info: dict[str, str] = {}

    def add_check(self, name: str, passed: bool, warning: Optional[str] = None, info: Optional[str] = None) -> None:
        """Record a check result.

        Args:
            name: Check name
            passed: Whether check passed
            warning: Optional warning message
            info: Optional info message
        """
        self.checks[name] = passed
        if warning:
            self.warnings.append(warning)
        if info:
            self.info[name] = info

    def add_error(self, name: str, error_msg: str) -> None:
        """Record a check error.

        Args:
            name: Check name
            error_msg: Error message
        """
        self.checks[name] = False
        self.errors.append(f"{name}: {error_msg}")

    def is_healthy(self) -> bool:
        """Check if environment is healthy (all checks passed).

        Returns:
            True if all checks passed, False otherwise
        """
        return all(self.checks.values()) and not self.errors

    def summary(self) -> str:
        """Generate summary string of check results.

        Returns:
            Human-readable summary
        """
        passed = sum(1 for v in self.checks.values() if v)
        total = len(self.checks)
        status = "HEALTHY" if self.is_healthy() else "ISSUES DETECTED"
        return f"Environment Check: {status} ({passed}/{total} checks passed)"
    
    def get_recommendations(self) -> list[str]:
        """Get recommendations for fixing detected issues.
        
        Returns:
            List of actionable recommendations
        """
        recommendations = []
        
        # CUDA recommendations
        if "CUDA Available" in self.checks and not self.checks["CUDA Available"]:
            recommendations.append(
                "CUDA not available. For GPU training, ensure PyTorch CUDA support is installed: "
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
            )
        
        # Disk space recommendations
        if "Disk Space" in self.checks and not self.checks["Disk Space"]:
            recommendations.append(
                "Low disk space detected. Free up space or configure checkpoint directory to different location: "
                "export CHECKPOINT_DIR=/path/with/more/space"
            )
        
        # Dependency recommendations
        for check_name in self.checks:
            if check_name.startswith("Required:") and not self.checks[check_name]:
                package = check_name.replace("Required: ", "")
                recommendations.append(f"Install missing dependency: pip install {package}")
        
        return recommendations


def daf_check_cuda_availability(console: Optional[Console] = None) -> EnvironmentCheckResult:
    """Verify GPU/CUDA setup using cogames.device patterns.

    Args:
        console: Optional Rich console for output

    Returns:
        EnvironmentCheckResult with CUDA availability status
    """
    if console is None:
        console = Console()

    result = EnvironmentCheckResult()

    # Check if CUDA backend is available
    cuda_backend = getattr(torch.backends, "cuda", None)
    if cuda_backend is None or not cuda_backend.is_built():
        result.add_check("CUDA Backend", False, warning="CUDA backend not built into PyTorch")
        result.info["device"] = "CPU only"
        return result

    # Check if cuda device count function exists
    if not hasattr(torch._C, "_cuda_getDeviceCount"):
        result.add_check("CUDA Support", False, warning="CUDA device count function not available")
        return result

    # Check if CUDA is available
    try:
        cuda_available = torch.cuda.is_available()
        result.add_check("CUDA Available", cuda_available)

        if cuda_available:
            device_count = torch.cuda.device_count()
            result.info["device_count"] = str(device_count)

            # Check each device
            for i in range(device_count):
                try:
                    device_name = torch.cuda.get_device_name(i)
                    device_props = torch.cuda.get_device_properties(i)
                    result.info[f"device_{i}_name"] = device_name
                    result.info[f"device_{i}_memory"] = f"{device_props.total_memory / 1e9:.1f} GB"
                except Exception as e:
                    result.add_error(f"Device {i}", str(e))

            console.print("[green]✓ CUDA available with {} device(s)[/green]".format(device_count))
        else:
            console.print("[yellow]⚠ CUDA not available - will use CPU[/yellow]")

    except Exception as e:
        result.add_error("CUDA Check", str(e))

    return result


def daf_check_disk_space(
    checkpoint_dir: Path = Path("./daf_output/checkpoints"),
    min_available_gb: float = 10.0,
    console: Optional[Console] = None,
) -> EnvironmentCheckResult:
    """Ensure sufficient disk space for checkpoints.

    Args:
        checkpoint_dir: Directory for checkpoints
        min_available_gb: Minimum required free space in GB
        console: Optional Rich console for output

    Returns:
        EnvironmentCheckResult with disk space status
    """
    if console is None:
        console = Console()

    result = EnvironmentCheckResult()

    if not HAS_PSUTIL:
        result.add_check("Disk Space Check", False, warning="psutil not available - skipping disk space check")
        return result

    try:
        # Create checkpoint directory if needed
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Get disk usage for directory's filesystem
        usage = psutil.disk_usage(str(checkpoint_dir))
        available_gb = usage.free / 1e9

        has_space = available_gb >= min_available_gb
        result.add_check(
            "Disk Space",
            has_space,
            warning=f"Low disk space: {available_gb:.1f} GB available, {min_available_gb} GB recommended"
            if not has_space
            else None,
            info=f"{available_gb:.1f} GB available",
        )

        if has_space:
            console.print(f"[green]✓ Disk space available: {available_gb:.1f} GB[/green]")
        else:
            console.print(f"[yellow]⚠ Low disk space: {available_gb:.1f} GB (recommended: {min_available_gb} GB)[/yellow]")

    except Exception as e:
        result.add_error("Disk Space Check", str(e))

    return result


def daf_check_dependencies(console: Optional[Console] = None) -> EnvironmentCheckResult:
    """Validate all required packages are installed and importable.

    Args:
        console: Optional Rich console for output

    Returns:
        EnvironmentCheckResult with dependency status
    """
    if console is None:
        console = Console()

    result = EnvironmentCheckResult()

    required_packages = [
        "torch",
        "numpy",
        "pydantic",
        "pyyaml",
        "typer",
        "rich",
        "mettagrid",
        "pufferlib",
    ]

    optional_packages = [
        "ray",  # For distributed training
        "matplotlib",  # For visualization
        "scipy",  # For statistical tests
        "psutil",  # For system monitoring
    ]

    # Check required packages
    for package_name in required_packages:
        try:
            importlib.import_module(package_name)
            result.add_check(f"Required: {package_name}", True, info="Available")
        except ImportError:
            result.add_error(f"Required Package: {package_name}", "Not installed or not importable")

    # Check optional packages
    for package_name in optional_packages:
        try:
            importlib.import_module(package_name)
            result.add_check(f"Optional: {package_name}", True, info="Available")
        except ImportError:
            result.add_check(
                f"Optional: {package_name}",
                False,
                warning=f"Optional package '{package_name}' not available - some features may be limited",
            )

    if result.is_healthy():
        console.print("[green]✓ All required packages available[/green]")
    else:
        console.print("[red]✗ Missing required dependencies[/red]")

    return result


def daf_check_mission_configs(missions: list[str], console: Optional[Console] = None) -> EnvironmentCheckResult:
    """Validate mission configurations can be loaded.

    Args:
        missions: List of mission names or paths
        console: Optional Rich console for output

    Returns:
        EnvironmentCheckResult with mission validation status
    """
    if console is None:
        console = Console()

    from cogames.cli.mission import get_mission_name_and_config

    result = EnvironmentCheckResult()

    for mission_name in missions:
        try:
            # Try to load the mission
            from typer import Context

            ctx = Context(lambda: None)
            _ = get_mission_name_and_config(ctx, mission_name)
            result.add_check(f"Mission: {mission_name}", True, info="Loadable")
        except Exception as e:
            result.add_error(f"Mission: {mission_name}", str(e))

    if result.is_healthy():
        console.print(f"[green]✓ All {len(missions)} mission(s) loadable[/green]")
    else:
        console.print(f"[red]✗ Some missions failed to load[/red]")

    return result


def daf_check_environment(
    checkout_dir: Optional[Path] = None,
    missions: Optional[list[str]] = None,
    check_cuda: bool = True,
    check_disk: bool = True,
    check_dependencies: bool = True,
    check_missions: bool = True,
    console: Optional[Console] = None,
) -> EnvironmentCheckResult:
    """Comprehensive environment check before training/sweeps.

    Runs all checks and reports any issues or warnings.

    Args:
        checkout_dir: Checkpoint directory for disk space check
        missions: Mission names to validate
        check_cuda: Whether to check CUDA availability
        check_disk: Whether to check disk space
        check_dependencies: Whether to check package dependencies
        check_missions: Whether to validate mission configs
        console: Optional Rich console for output

    Returns:
        EnvironmentCheckResult with combined results
    """
    if console is None:
        console = Console()

    console.print("\n[bold cyan]DAF Environment Check[/bold cyan]\n")

    combined_result = EnvironmentCheckResult()

    # Check CUDA
    if check_cuda:
        cuda_result = daf_check_cuda_availability(console)
        for name, passed in cuda_result.checks.items():
            combined_result.checks[name] = passed
        combined_result.warnings.extend(cuda_result.warnings)
        combined_result.errors.extend(cuda_result.errors)

    # Check disk space
    if check_disk:
        checkout_dir = checkout_dir or Path("./daf_output/checkpoints")
        disk_result = daf_check_disk_space(checkout_dir, console=console)
        for name, passed in disk_result.checks.items():
            combined_result.checks[name] = passed
        combined_result.warnings.extend(disk_result.warnings)
        combined_result.errors.extend(disk_result.errors)

    # Check dependencies
    if check_dependencies:
        dep_result = daf_check_dependencies(console)
        for name, passed in dep_result.checks.items():
            combined_result.checks[name] = passed
        combined_result.warnings.extend(dep_result.warnings)
        combined_result.errors.extend(dep_result.errors)

    # Check missions
    if check_missions and missions:
        mission_result = daf_check_mission_configs(missions, console)
        for name, passed in mission_result.checks.items():
            combined_result.checks[name] = passed
        combined_result.warnings.extend(mission_result.warnings)
        combined_result.errors.extend(mission_result.errors)

    # Print summary
    console.print()
    console.print(combined_result.summary())

    if combined_result.warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warning in combined_result.warnings:
            console.print(f"  [yellow]⚠ {warning}[/yellow]")

    if combined_result.errors:
        console.print("\n[red]Errors:[/red]")
        for error in combined_result.errors:
            console.print(f"  [red]✗ {error}[/red]")
    
    # Print recommendations if there are issues
    recommendations = combined_result.get_recommendations()
    if recommendations:
        console.print("\n[cyan]Recommendations:[/cyan]")
        for i, rec in enumerate(recommendations, 1):
            console.print(f"  {i}. {rec}")

    console.print()

    return combined_result


def daf_get_recommended_device(console: Optional[Console] = None) -> torch.device:
    """Get recommended device for training (CUDA if available, else CPU).

    Uses cogames.device.resolve_training_device pattern but with DAF-specific
    environment checks.

    Args:
        console: Optional Rich console for output

    Returns:
        Recommended torch.device
    """
    from cogames.device import resolve_training_device

    if console is None:
        console = Console()

    return resolve_training_device(console, "auto")

