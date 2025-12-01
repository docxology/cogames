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
        """Check if environment is healthy (no critical errors).

        Returns:
            True if no critical errors, False otherwise.
            Warnings (CUDA, optional packages) don't affect health status.
        """
        return not self.errors

    def is_fully_optimal(self) -> bool:
        """Check if environment is fully optimal (all checks passed).

        Returns:
            True if all checks passed and no warnings/errors
        """
        return all(self.checks.values()) and not self.errors and not self.warnings

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
        import platform
        is_macos = platform.system() == "Darwin"
        recommendations = []
        
        # GPU recommendations - platform-specific
        if is_macos:
            # MPS recommendations for macOS
            if "MPS Backend" in self.checks and not self.checks["MPS Backend"]:
                recommendations.append(
                    "MPS backend not built. Reinstall PyTorch with MPS support: "
                    "pip install torch torchvision torchaudio"
                )
            elif "MPS Available" in self.checks and not self.checks["MPS Available"]:
                recommendations.append(
                    "MPS not available. Ensure you're on Apple Silicon (M1/M2/M3) Mac with macOS 12.3+."
                )
        else:
            # CUDA recommendations for Linux/Windows
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


def _is_macos() -> bool:
    """Check if running on macOS."""
    import platform
    return platform.system() == "Darwin"


def daf_check_gpu_availability(console: Optional[Console] = None) -> EnvironmentCheckResult:
    """Verify GPU setup: CUDA on Linux/Windows, MPS on macOS.

    Args:
        console: Optional Rich console for output

    Returns:
        EnvironmentCheckResult with GPU availability status
    """
    if console is None:
        console = Console()

    result = EnvironmentCheckResult()
    is_macos = _is_macos()

    if is_macos:
        # macOS: Check MPS (Metal Performance Shaders) backend
        return _check_mps_availability(result, console)
    else:
        # Linux/Windows: Check CUDA backend
        return _check_cuda_availability(result, console)


def _check_mps_availability(result: EnvironmentCheckResult, console: Console) -> EnvironmentCheckResult:
    """Check MPS (Metal) GPU availability on macOS.

    Args:
        result: EnvironmentCheckResult to populate
        console: Rich console for output

    Returns:
        Updated EnvironmentCheckResult
    """
    try:
        mps_built = torch.backends.mps.is_built()
        mps_available = torch.backends.mps.is_available()

        if not mps_built:
            result.add_check("MPS Backend", False, warning="MPS backend not built into PyTorch")
            result.info["device"] = "CPU only"
            console.print("[yellow]⚠ MPS not built - will use CPU[/yellow]")
            return result

        result.add_check("MPS Backend", True, info="Built")

        if mps_available:
            result.add_check("MPS Available", True, info="Apple Silicon GPU ready")
            result.info["device"] = "MPS (Apple Silicon)"
            result.info["gpu_backend"] = "Metal Performance Shaders"
            console.print("[green]✓ MPS (Apple Silicon GPU) available[/green]")
        else:
            result.add_check("MPS Available", False, warning="MPS built but not available on this device")
            result.info["device"] = "CPU"
            console.print("[yellow]⚠ MPS not available on this Mac - will use CPU[/yellow]")

    except Exception as e:
        result.add_error("MPS Check", str(e))

    return result


def _check_cuda_availability(result: EnvironmentCheckResult, console: Console) -> EnvironmentCheckResult:
    """Check CUDA GPU availability on Linux/Windows.

    Args:
        result: EnvironmentCheckResult to populate
        console: Rich console for output

    Returns:
        Updated EnvironmentCheckResult
    """
    # Check if CUDA backend is available
    cuda_backend = getattr(torch.backends, "cuda", None)
    if cuda_backend is None or not cuda_backend.is_built():
        result.add_check("CUDA Backend", False, warning="CUDA backend not built into PyTorch")
        result.info["device"] = "CPU only"
        console.print("[yellow]⚠ CUDA not built - will use CPU[/yellow]")
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
            result.info["gpu_backend"] = "CUDA"

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


# Backward compatibility alias
def daf_check_cuda_availability(console: Optional[Console] = None) -> EnvironmentCheckResult:
    """Verify GPU setup (CUDA or MPS depending on platform).

    Deprecated: Use daf_check_gpu_availability() instead.

    Args:
        console: Optional Rich console for output

    Returns:
        EnvironmentCheckResult with GPU availability status
    """
    return daf_check_gpu_availability(console)


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

    # Package name -> import name mapping (for packages where they differ)
    required_packages = {
        "torch": "torch",
        "numpy": "numpy",
        "pydantic": "pydantic",
        "pyyaml": "yaml",  # PyYAML is imported as yaml
        "typer": "typer",
        "rich": "rich",
        "mettagrid": "mettagrid",
        "pufferlib": "pufferlib",
    }

    optional_packages = [
        "ray",  # For distributed training
        "matplotlib",  # For visualization
        "scipy",  # For statistical tests
        "psutil",  # For system monitoring
    ]

    # Check required packages
    for package_name, import_name in required_packages.items():
        try:
            importlib.import_module(import_name)
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
        missions: List of mission names in format 'site.mission' (e.g., 'training_facility.open_world')
        console: Optional Rich console for output

    Returns:
        EnvironmentCheckResult with mission validation status
    """
    if console is None:
        console = Console()

    result = EnvironmentCheckResult()

    for mission_name in missions:
        try:
            # Parse site.mission format and load directly from mission module
            if "." in mission_name:
                from cogames.cogs_vs_clips.missions import MISSIONS
                
                # Find mission by checking both name and full_name
                site_name, mission_part = mission_name.split(".", 1)
                found = False
                for m in MISSIONS:
                    # Check various name formats
                    if hasattr(m, 'full_name') and m.full_name == mission_name:
                        found = True
                        break
                    if hasattr(m, 'name') and m.name == mission_part:
                        # Check if site matches
                        if hasattr(m, 'site') and hasattr(m.site, 'name') and m.site.name == site_name:
                            found = True
                            break
                
                if found:
                    result.add_check(f"Mission: {mission_name}", True, info="Loadable")
                else:
                    result.add_error(f"Mission: {mission_name}", f"Not found in MISSIONS")
            else:
                # Legacy format - try direct name match
                from cogames.cogs_vs_clips.missions import MISSIONS
                
                found = any(
                    (hasattr(m, 'name') and m.name == mission_name) or
                    (hasattr(m, 'full_name') and m.full_name == mission_name)
                    for m in MISSIONS
                )
                if found:
                    result.add_check(f"Mission: {mission_name}", True, info="Loadable")
                else:
                    result.add_error(f"Mission: {mission_name}", f"Not found in MISSIONS")
                    
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

    # Check GPU (CUDA on Linux/Windows, MPS on macOS)
    if check_cuda:
        gpu_result = daf_check_gpu_availability(console)
        for name, passed in gpu_result.checks.items():
            combined_result.checks[name] = passed
        combined_result.warnings.extend(gpu_result.warnings)
        combined_result.errors.extend(gpu_result.errors)

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
    """Get recommended device for training (CUDA/MPS if available, else CPU).

    Uses cogames.device.resolve_training_device pattern but with DAF-specific
    environment checks. On macOS, prefers MPS (Metal). On Linux/Windows, prefers CUDA.

    Args:
        console: Optional Rich console for output

    Returns:
        Recommended torch.device
    """
    from cogames.device import resolve_training_device

    if console is None:
        console = Console()

    device = resolve_training_device(console, "auto")
    
    # Provide informative message about the selected device
    if _is_macos() and device.type == "mps":
        console.print("[green]✓ Using MPS (Apple Silicon GPU) for acceleration[/green]")
    elif device.type == "cuda":
        console.print(f"[green]✓ Using CUDA GPU for acceleration[/green]")
    else:
        console.print("[yellow]ℹ Using CPU (no GPU acceleration)[/yellow]")
    
    return device

