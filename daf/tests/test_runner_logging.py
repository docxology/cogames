#!/usr/bin/env python3
"""
Structured logging wrapper for test execution with OutputManager and DAFLogger.

Provides centralized logging for the test runner with consistent formatting,
performance metrics tracking, and detailed operation logging.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from daf.src.output_manager import get_output_manager
from daf.src.logging_config import create_daf_logger


class TestRunnerLogger:
    """Structured logging for test suite execution."""

    def __init__(self, output_base: str = "./daf_output", verbose: bool = False):
        """Initialize test runner logger.
        
        Args:
            output_base: Base output directory for DAF operations
            verbose: Enable verbose logging
        """
        self.output_manager = get_output_manager(
            base_dir=output_base,
            verbose=verbose,
            log_to_file=True
        )
        self.logger = create_daf_logger(
            name="test_runner",
            log_dir=Path(output_base) / "logs",
            verbose=verbose,
            track_calls=False
        )
        self.test_results: Dict[str, Dict] = {}

    def log_test_collection_phase(self, phase1_total: int, phase2_total: int, grand_total: int) -> None:
        """Log test collection phase results."""
        self.logger.info("=" * 70)
        self.logger.info("TEST COLLECTION PHASE: Counting Tests Before Execution")
        self.logger.info("=" * 70)
        
        with self.logger.track_operation("test_collection", metadata={
            "phase1_tests": phase1_total,
            "phase2_tests": phase2_total,
            "grand_total": grand_total,
        }):
            self.logger.info(f"Phase 1 (CoGames): {phase1_total} tests across 10 suites")
            self.logger.info(f"Phase 2 (DAF):     {phase2_total} tests across 8 suites")
            self.logger.info(f"Total to execute:  {grand_total} tests")

    def log_phase_start(self, phase: int, phase_name: str) -> None:
        """Log start of test phase."""
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info(f"PHASE {phase}: {phase_name}")
        self.logger.info("=" * 70)

    def log_suite_start(self, suite_name: str, test_file: str) -> None:
        """Log start of test suite execution."""
        self.logger.debug(f"Starting: {suite_name}")
        self.logger.debug(f"  File: {test_file}")

    def log_suite_result(self, suite_name: str, passed: bool, exit_code: int, duration: float) -> None:
        """Log test suite result."""
        status = "PASSED" if passed else "FAILED"
        status_icon = "✓" if passed else "✗"
        self.logger.info(f"{status_icon} {suite_name}: {status} (exit code: {exit_code}, {duration:.2f}s)")
        
        self.test_results[suite_name] = {
            "status": status,
            "exit_code": exit_code,
            "duration": duration,
            "passed": passed,
        }

    def log_phase_complete(self, phase: int, all_passed: bool) -> None:
        """Log phase completion."""
        self.logger.info("")
        self.logger.info("=" * 70)
        if all_passed:
            self.logger.info(f"PHASE {phase} Complete: All Tests Passed")
        else:
            self.logger.info(f"PHASE {phase} Complete: Some Tests Failed")
        self.logger.info("=" * 70)

    def log_full_test_complete(self, phase1_passed: bool, phase2_passed: bool) -> None:
        """Log full test suite completion."""
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("FULL TEST SUITE COMPLETE")
        self.logger.info("=" * 70)
        
        if phase1_passed:
            self.logger.info("✓ Phase 1: All CoGames Tests Passed")
        else:
            self.logger.info("✗ Phase 1: Some CoGames Tests Failed")
        
        if phase2_passed:
            self.logger.info("✓ Phase 2: All DAF Tests Passed")
        else:
            self.logger.info("✗ Phase 2: Some DAF Tests Failed")

    def log_report_generation(self, test_output_dir: str) -> None:
        """Log report generation."""
        self.logger.info("")
        self.logger.info("Generating test reports...")
        
        with self.logger.track_operation("report_generation"):
            self.logger.info(f"Test outputs: {test_output_dir}")
            self.logger.info("Generating Markdown report...")
            self.logger.info("Generating JSON report...")
            self.logger.info("Generating plain text summary...")

    def save_metrics(self) -> Path:
        """Save test metrics to JSON."""
        with self.logger.track_operation("save_metrics"):
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "test_results": self.test_results,
                "total_suites": len(self.test_results),
                "suites_passed": sum(1 for r in self.test_results.values() if r["passed"]),
                "suites_failed": sum(1 for r in self.test_results.values() if not r["passed"]),
                "total_duration": sum(r["duration"] for r in self.test_results.values()),
            }
            
            output_file = self.output_manager.dirs.logs / f"test_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, "w") as f:
                json.dump(metrics, f, indent=2)
            
            self.logger.info(f"Saved metrics: {output_file}")
            return output_file

    def print_summary(self) -> None:
        """Print test execution summary."""
        self.logger.print_section("Test Execution Summary", level=1)
        
        summary_data = [
            {
                "Suite": name,
                "Status": result["status"],
                "Duration (s)": f"{result['duration']:.2f}",
            }
            for name, result in self.test_results.items()
        ]
        
        self.logger.print_table("Test Suites", summary_data)
        
        total_duration = sum(r["duration"] for r in self.test_results.values())
        passed = sum(1 for r in self.test_results.values() if r["passed"])
        total = len(self.test_results)
        
        self.logger.print_section("Overall Metrics", level=2)
        metrics_data = [
            {
                "Metric": "Total Suites",
                "Value": str(total),
            },
            {
                "Metric": "Suites Passed",
                "Value": str(passed),
            },
            {
                "Metric": "Suites Failed",
                "Value": str(total - passed),
            },
            {
                "Metric": "Total Duration (s)",
                "Value": f"{total_duration:.2f}",
            },
        ]
        
        for item in metrics_data:
            self.logger.info(f"  {item['Metric']}: {item['Value']}")


if __name__ == "__main__":
    # Example usage
    runner_logger = TestRunnerLogger(verbose=True)
    
    runner_logger.log_test_collection_phase(185, 100, 285)
    
    runner_logger.log_phase_start(1, "CoGames Core Tests")
    runner_logger.log_suite_start("CLI Tests", "tests/test_cli.py")
    runner_logger.log_suite_result("CLI Tests", True, 0, 7.70)
    runner_logger.log_phase_complete(1, True)
    
    runner_logger.log_phase_start(2, "DAF Module Tests")
    runner_logger.log_suite_start("Configuration Tests", "daf/tests/test_config.py")
    runner_logger.log_suite_result("Configuration Tests", True, 0, 0.11)
    runner_logger.log_phase_complete(2, True)
    
    runner_logger.log_full_test_complete(True, True)
    runner_logger.print_summary()
    runner_logger.save_metrics()

