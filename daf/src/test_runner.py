"""Unified test runner with organized output collection and reporting.

Provides infrastructure for running test suites with structured output,
comprehensive logging, and organized results collection.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from daf.logging_config import DAFLogger, create_daf_logger
from daf.output_manager import OutputManager, get_output_manager


@dataclass
class TestResult:
    """Result of a single test suite execution."""

    test_name: str
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float
    timestamp: str

    @property
    def passed(self) -> bool:
        """Check if test passed."""
        return self.returncode == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class TestRunner:
    """Unified test runner with organized output management.

    Executes test suites and collects results with:
    - Structured logging
    - Output organization by test category
    - Comprehensive reporting
    - Performance metrics
    """

    def __init__(
        self,
        output_base_dir: Path | str = "./daf_output",
        verbose: bool = False,
    ):
        """Initialize test runner.

        Args:
            output_base_dir: Base directory for test outputs
            verbose: Enable verbose output
        """
        self.output_manager = OutputManager(
            base_dir=output_base_dir,
            verbose=verbose,
            log_to_file=True,
        )
        self.logger = create_daf_logger(
            name="daf.tests",
            log_dir=self.output_manager.dirs.logs,
            verbose=verbose,
        )
        self.console = Console()
        self.results: List[TestResult] = []
        self.categories: Dict[str, List[TestResult]] = {}

    def run_test_suite(
        self,
        test_path: str,
        category: str = "core",
        suite_name: Optional[str] = None,
    ) -> TestResult:
        """Run a single test suite with pytest.

        Args:
            test_path: Path to test file or directory
            category: Test category (cogames, daf, etc)
            suite_name: Optional friendly name for the suite

        Returns:
            TestResult instance
        """
        suite_name = suite_name or Path(test_path).stem
        display_name = f"{category}/{suite_name}"

        self.logger.info(f"Running test suite: {display_name}")

        start_time = datetime.now()

        try:
            result = subprocess.run(
                ["python", "-m", "pytest", test_path, "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=600,
            )

            duration = (datetime.now() - start_time).total_seconds()

            test_result = TestResult(
                test_name=display_name,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                duration_seconds=duration,
                timestamp=start_time.isoformat(),
            )

            # Track result
            self.results.append(test_result)
            if category not in self.categories:
                self.categories[category] = []
            self.categories[category].append(test_result)

            # Log result
            status = "✓ PASSED" if test_result.passed else "✗ FAILED"
            msg = f"{status}: {display_name} ({duration:.2f}s)"
            if test_result.passed:
                self.logger.info(msg)
            else:
                self.logger.error(msg)

            return test_result

        except subprocess.TimeoutExpired:
            duration = (datetime.now() - start_time).total_seconds()
            test_result = TestResult(
                test_name=display_name,
                returncode=-1,
                stdout="",
                stderr="Test suite timed out after 600 seconds",
                duration_seconds=duration,
                timestamp=start_time.isoformat(),
            )
            self.results.append(test_result)
            self.logger.error(f"TIMEOUT: {display_name}")
            return test_result

    def run_test_batch(
        self,
        tests: List[tuple[str, str, Optional[str]]],
    ) -> Dict[str, List[TestResult]]:
        """Run multiple test suites.

        Args:
            tests: List of (test_path, category, suite_name) tuples

        Returns:
            Dictionary of results organized by category
        """
        self.logger.print_section("Running Test Suite", level=1)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task(f"Running {len(tests)} test suites...", total=len(tests))

            for test_path, category, suite_name in tests:
                progress.update(task, description=f"Running {category}/{suite_name or Path(test_path).stem}...")
                self.run_test_suite(test_path, category, suite_name)
                progress.advance(task)

        return self.categories

    def save_test_outputs(self) -> Path:
        """Save all test outputs to organized folders.

        Returns:
            Path to test outputs directory
        """
        test_output_dir = self.output_manager.get_operation_dir("evaluation", "tests")

        for category, results in self.categories.items():
            category_dir = test_output_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)

            for result in results:
                test_name = result.test_name.replace("/", "_")

                # Save stdout
                stdout_file = category_dir / f"{test_name}_stdout.txt"
                with open(stdout_file, "w") as f:
                    f.write(result.stdout)

                # Save stderr if present
                if result.stderr:
                    stderr_file = category_dir / f"{test_name}_stderr.txt"
                    with open(stderr_file, "w") as f:
                        f.write(result.stderr)

        self.logger.info(f"Saved test outputs to: {test_output_dir}")
        return test_output_dir

    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report.

        Returns:
            Report dictionary
        """
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        total_duration = sum(r.duration_seconds for r in self.results)

        report = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.output_manager.session_id,
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "pass_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                "total_duration_seconds": total_duration,
            },
            "by_category": {
                category: {
                    "total": len(results),
                    "passed": sum(1 for r in results if r.passed),
                    "failed": sum(1 for r in results if not r.passed),
                    "duration_seconds": sum(r.duration_seconds for r in results),
                }
                for category, results in self.categories.items()
            },
            "tests": [r.to_dict() for r in self.results],
        }

        return report

    def save_test_report(self) -> Path:
        """Save comprehensive test report.

        Returns:
            Path to report file
        """
        report = self.generate_test_report()

        report_file = self.output_manager.save_json_results(
            report,
            operation="evaluation",
            filename="test_report",
            subdir="tests",
        )

        return report_file

    def print_test_summary(self) -> None:
        """Print formatted test summary to console."""
        report = self.generate_test_report()
        summary = report["summary"]

        self.logger.print_section("Test Results Summary", level=1)

        # Summary metrics
        summary_data = [
            {"metric": "Total Tests", "value": summary["total_tests"]},
            {"metric": "Passed", "value": summary["passed"]},
            {"metric": "Failed", "value": summary["failed"]},
            {"metric": "Pass Rate", "value": f"{summary['pass_rate']:.1f}%"},
            {"metric": "Total Duration", "value": f"{summary['total_duration_seconds']:.2f}s"},
        ]

        for item in summary_data:
            status_color = "cyan"
            if item["metric"] == "Failed" and item["value"] > 0:
                status_color = "red"
            elif item["metric"] == "Passed":
                status_color = "green"

            self.console.print(
                f"  {item['metric']:.<20} [{status_color}]{item['value']}[/{status_color}]"
            )

        # By category breakdown
        if report["by_category"]:
            self.logger.print_section("Results by Category", level=2)

            category_data = [
                {
                    "Category": cat,
                    "Total": stats["total"],
                    "Passed": stats["passed"],
                    "Failed": stats["failed"],
                    "Duration (s)": f"{stats['duration_seconds']:.2f}",
                }
                for cat, stats in report["by_category"].items()
            ]

            self.logger.print_table("Test Results by Category", category_data)

    def print_failed_tests(self) -> None:
        """Print details of failed tests."""
        failed = [r for r in self.results if not r.passed]

        if not failed:
            self.console.print("[green]✓ All tests passed![/green]\n")
            return

        self.logger.print_section(f"Failed Tests ({len(failed)})", level=2)

        for result in failed:
            self.console.print(f"\n[red]✗ {result.test_name}[/red]")
            self.console.print(f"  Return code: {result.returncode}")
            if result.stderr:
                self.console.print(f"[yellow]  Error output:[/yellow]")
                # Print last 10 lines of stderr
                error_lines = result.stderr.split("\n")[-10:]
                for line in error_lines:
                    if line.strip():
                        self.console.print(f"    {line}")

    def cleanup(self) -> None:
        """Cleanup test runner resources."""
        self.output_manager.cleanup_temp_files()
        self.logger.info("Test runner cleanup complete")






