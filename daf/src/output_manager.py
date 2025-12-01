"""Centralized output and logging management for DAF.

Provides unified output structure with organized subfolders for all DAF operations
(sweeps, comparisons, training, deployment, etc.). Implements structured logging with
both file and console output.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.logging import RichHandler


@dataclass
class OutputDirectories:
    """Structured output directories for all DAF operations."""

    base_dir: Path
    sweeps: Path
    comparisons: Path
    training: Path
    deployment: Path
    evaluations: Path
    visualizations: Path
    logs: Path
    reports: Path
    artifacts: Path
    temp: Path

    @classmethod
    def create(cls, base_dir: Path | str) -> OutputDirectories:
        """Create and initialize all output directories.

        Args:
            base_dir: Root output directory for all DAF operations

        Returns:
            OutputDirectories instance with all paths created
        """
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        dirs = cls(
            base_dir=base_dir,
            sweeps=base_dir / "sweeps",
            comparisons=base_dir / "comparisons",
            training=base_dir / "training",
            deployment=base_dir / "deployment",
            evaluations=base_dir / "evaluations",
            visualizations=base_dir / "visualizations",
            logs=base_dir / "logs",
            reports=base_dir / "reports",
            artifacts=base_dir / "artifacts",
            temp=base_dir / ".temp",
        )

        # Create all directories
        for path in [
            dirs.sweeps,
            dirs.comparisons,
            dirs.training,
            dirs.deployment,
            dirs.evaluations,
            dirs.visualizations,
            dirs.logs,
            dirs.reports,
            dirs.artifacts,
            dirs.temp,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        return dirs


class OutputManager:
    """Centralized manager for all DAF outputs and logging.

    Provides:
    - Unified output directory structure
    - Structured logging with file and console handlers
    - JSON results tracking
    - Automatic directory creation and cleanup
    - Performance metrics recording
    """

    def __init__(
        self,
        base_dir: Path | str = "./daf_output",
        verbose: bool = False,
        log_to_file: bool = True,
    ):
        """Initialize output manager.

        Args:
            base_dir: Root output directory
            verbose: Enable verbose logging
            log_to_file: Write logs to files
        """
        self.base_dir = Path(base_dir)
        self.verbose = verbose
        self.log_to_file = log_to_file

        # Create output directory structure
        self.dirs = OutputDirectories.create(self.base_dir)

        # Setup logging
        self.logger = self._setup_logging()
        self.console = Console(force_terminal=not verbose)

        # Track session metadata
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_metadata: Dict[str, Any] = {
            "session_id": self.session_id,
            "created": datetime.now().isoformat(),
            "operations": [],
        }

    def _setup_logging(self) -> logging.Logger:
        """Configure structured logging with file and console handlers.

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger("daf")
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)

        # Clear existing handlers
        logger.handlers.clear()

        # Console handler with Rich formatting
        console_handler = RichHandler(
            console=Console(),
            show_time=True,
            show_level=True,
            show_path=False,
        )
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

        # File handler
        if self.log_to_file:
            log_file = self.dirs.logs / f"daf_{self.session_id}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
            )
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            self.logger = logger
            logger.info(f"Logging to: {log_file}")

        return logger

    def get_operation_dir(self, operation: str, subdir: Optional[str] = None) -> Path:
        """Get output directory for specific operation type.

        Args:
            operation: Operation type (sweep, comparison, training, etc)
            subdir: Optional subdirectory within operation folder

        Returns:
            Path to operation output directory

        Raises:
            ValueError: If operation type not recognized
        """
        operation_map = {
            "sweep": self.dirs.sweeps,
            "comparison": self.dirs.comparisons,
            "training": self.dirs.training,
            "deployment": self.dirs.deployment,
            "evaluation": self.dirs.evaluations,
            "visualization": self.dirs.visualizations,
        }

        if operation not in operation_map:
            raise ValueError(f"Unknown operation type: {operation}")

        base_path = operation_map[operation]

        if subdir:
            path = base_path / subdir
        else:
            path = base_path / self.session_id

        path.mkdir(parents=True, exist_ok=True)
        return path

    def log_operation_start(
        self,
        operation: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log start of DAF operation.

        Args:
            operation: Operation type (sweep, comparison, etc)
            details: Optional operation-specific details
        """
        msg = f"Starting {operation}"
        self.logger.info(msg)

        if details:
            for key, value in details.items():
                self.logger.debug(f"  {key}: {value}")

        self.session_metadata["operations"].append(
            {
                "operation": operation,
                "timestamp": datetime.now().isoformat(),
                "details": details or {},
            }
        )

    def log_operation_complete(
        self,
        operation: str,
        status: str = "success",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log completion of DAF operation.

        Args:
            operation: Operation type
            status: Completion status (success, warning, error)
            details: Optional completion-specific details
        """
        level = {
            "success": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
        }.get(status, logging.INFO)

        msg = f"Completed {operation}: {status}"
        self.logger.log(level, msg)

        if details:
            for key, value in details.items():
                self.logger.debug(f"  {key}: {value}")

    def save_json_results(
        self,
        data: Dict[str, Any] | List[Any],
        operation: str,
        filename: str,
        subdir: Optional[str] = None,
    ) -> Path:
        """Save results to JSON file in operation directory.

        Args:
            data: Data to save
            operation: Operation type
            filename: Output filename (without .json extension)
            subdir: Optional subdirectory

        Returns:
            Path to saved file
        """
        output_dir = self.get_operation_dir(operation, subdir)
        output_file = output_dir / f"{filename}.json"

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

        self.logger.info(f"Saved results: {output_file}")
        return output_file

    def save_text_results(
        self,
        text: str,
        operation: str,
        filename: str,
        subdir: Optional[str] = None,
    ) -> Path:
        """Save text results to file.

        Args:
            text: Text content to save
            operation: Operation type
            filename: Output filename (without extension)
            subdir: Optional subdirectory

        Returns:
            Path to saved file
        """
        output_dir = self.get_operation_dir(operation, subdir)
        output_file = output_dir / f"{filename}.txt"

        with open(output_file, "w") as f:
            f.write(text)

        self.logger.info(f"Saved report: {output_file}")
        return output_file

    def save_summary_report(
        self,
        operation: str,
        summary: Dict[str, Any],
        subdir: Optional[str] = None,
    ) -> Path:
        """Save standardized summary report.

        Args:
            operation: Operation type
            summary: Summary data with standard fields
            subdir: Optional subdirectory

        Returns:
            Path to saved report
        """
        output_dir = self.get_operation_dir(operation, subdir)
        report_file = output_dir / "summary_report.json"

        # Enhance summary with metadata
        enhanced_summary = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "session_id": self.session_id,
            **summary,
        }

        with open(report_file, "w") as f:
            json.dump(enhanced_summary, f, indent=2, default=str)

        self.logger.info(f"Saved summary report: {report_file}")
        return report_file

    def save_session_metadata(self) -> Path:
        """Save session metadata to file.

        Returns:
            Path to metadata file
        """
        metadata_file = self.dirs.logs / f"session_{self.session_id}.json"

        with open(metadata_file, "w") as f:
            json.dump(self.session_metadata, f, indent=2)

        self.logger.info(f"Saved session metadata: {metadata_file}")
        return metadata_file

    def get_output_structure_info(self) -> str:
        """Get human-readable output directory structure information.

        Returns:
            Formatted string describing output organization
        """
        info = f"""
DAF Output Structure (Session: {self.session_id})
{'=' * 60}
Base Directory: {self.dirs.base_dir}

Organized by operation type:
  - Sweeps:         {self.dirs.sweeps}
  - Comparisons:    {self.dirs.comparisons}
  - Training:       {self.dirs.training}
  - Deployment:     {self.dirs.deployment}
  - Evaluations:    {self.dirs.evaluations}
  - Visualizations: {self.dirs.visualizations}
  - Logs:           {self.dirs.logs}
  - Reports:        {self.dirs.reports}
  - Artifacts:      {self.dirs.artifacts}
{'=' * 60}
"""
        return info

    def print_output_info(self) -> None:
        """Print output structure information to console."""
        self.console.print(self.get_output_structure_info())

    def cleanup_temp_files(self) -> None:
        """Clean up temporary files from current session."""
        import shutil

        if self.dirs.temp.exists():
            try:
                shutil.rmtree(self.dirs.temp)
                self.logger.info("Cleaned up temporary files")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup temp files: {e}")


# Global output manager instance
_output_manager: Optional[OutputManager] = None


def get_output_manager(
    base_dir: Path | str = "./daf_output",
    verbose: bool = False,
    log_to_file: bool = True,
) -> OutputManager:
    """Get or create global output manager instance.

    Args:
        base_dir: Root output directory
        verbose: Enable verbose logging
        log_to_file: Write logs to files

    Returns:
        Singleton OutputManager instance
    """
    global _output_manager

    if _output_manager is None:
        _output_manager = OutputManager(
            base_dir=base_dir,
            verbose=verbose,
            log_to_file=log_to_file,
        )

    return _output_manager


def reset_output_manager() -> None:
    """Reset the global output manager instance."""
    global _output_manager
    _output_manager = None







