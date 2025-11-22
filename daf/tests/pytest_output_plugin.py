"""
Pytest plugin for enhanced test output and reporting.

Captures detailed information about test execution for better reporting and
diagnosis of test failures, including function calls and execution traces.
"""

import pytest
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional


class TestExecutionTracer:
    """Traces and records test execution details."""

    def __init__(self):
        """Initialize tracer."""
        self.execution_log: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("pytest_test_tracer")

    def log_event(self, event_type: str, test_name: str, details: Optional[Dict] = None) -> None:
        """Log an execution event."""
        self.execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "test": test_name,
            "details": details or {},
        })


class TestOutputCapture:
    """Captures detailed test execution data."""
    
    def __init__(self, output_dir: Path):
        """Initialize capture system."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tests: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
        self.tracer = TestExecutionTracer()
    
    def add_test(
        self,
        test_name: str,
        status: str,
        duration: float,
        passed: bool,
        failed: bool = False,
        skipped: bool = False,
        error_message: Optional[str] = None,
    ) -> None:
        """Record test execution data."""
        test_entry = {
            "name": test_name,
            "status": status,
            "duration": duration,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "timestamp": datetime.now().isoformat(),
        }
        if error_message:
            test_entry["error"] = error_message
        
        self.tests.append(test_entry)
        self.tracer.log_event(status, test_name)
    
    def save_json_report(self, filename: str = "test_results.json") -> Path:
        """Save results as JSON for programmatic access."""
        output_file = self.output_dir / filename
        
        # Build detailed test data with full error information
        tests_detailed = []
        for test in self.tests:
            test_detail = test.copy()
            # Find full traceback if available
            for log_entry in self.execution_log:
                if (log_entry.get("event") == "test_failure_details" and 
                    log_entry.get("test") == test["name"]):
                    test_detail["full_traceback"] = log_entry.get("full_traceback")
                    break
            tests_detailed.append(test_detail)
        
        data = {
            "generated": datetime.now().isoformat(),
            "start_time": self.start_time.isoformat(),
            "total_tests": len(self.tests),
            "passed": sum(1 for t in self.tests if t["passed"]),
            "failed": sum(1 for t in self.tests if t["failed"]),
            "skipped": sum(1 for t in self.tests if t["skipped"]),
            "total_duration": sum(t["duration"] for t in self.tests),
            "tests": tests_detailed,
            "execution_log": self.execution_log,
        }
        
        output_file.write_text(json.dumps(data, indent=2))
        return output_file


class TestOutputPlugin:
    """Pytest plugin for enhanced test reporting."""
    
    def __init__(self, output_dir: Path):
        """Initialize plugin."""
        self.output_dir = Path(output_dir)
        self.capture = TestOutputCapture(self.output_dir)
        self.current_test = None
    
    def pytest_runtest_logreport(self, report):
        """Called after each test phase (setup, call, teardown)."""
        if report.when == "call":
            # This is the actual test execution
            test_name = report.nodeid
            duration = report.duration
            error_message = None
            
            if report.passed:
                self.capture.add_test(
                    test_name,
                    "PASSED",
                    duration,
                    passed=True,
                )
            elif report.failed:
                # Capture failure reason with full traceback
                error_message = None
                if report.longrepr:
                    # Capture full traceback without truncation
                    error_message = str(report.longrepr)
                    # For storage, keep first 1000 chars in summary, full in details
                    summary = error_message[:200]
                    
                    self.capture.add_test(
                        test_name,
                        "FAILED",
                        duration,
                        passed=False,
                        failed=True,
                        error_message=summary,
                    )
                    
                    # Store full traceback separately
                    self.capture.execution_log.append({
                        "timestamp": datetime.now().isoformat(),
                        "event": "test_failure_details",
                        "test": test_name,
                        "full_traceback": error_message,
                    })
                else:
                    self.capture.add_test(
                        test_name,
                        "FAILED",
                        duration,
                        passed=False,
                        failed=True,
                    )
            elif report.skipped:
                # Capture skip reason
                if report.wasxfail:
                    error_message = "Expected failure"
                
                self.capture.add_test(
                    test_name,
                    "SKIPPED",
                    duration,
                    passed=False,
                    skipped=True,
                    error_message=error_message,
                )
    
    def pytest_sessionfinish(self, session):
        """Called at end of test session."""
        # Save JSON report
        report_file = self.capture.save_json_report()
        print(f"\n✓ Test results saved to: {report_file}")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--test-output-dir",
        action="store",
        default="./daf_output/logs",
        help="Directory to save detailed test output files",
    )


def pytest_configure(config):
    """Configure pytest with our plugin."""
    output_dir = config.getoption("--test-output-dir")
    plugin = TestOutputPlugin(output_dir)
    config.pluginmanager.register(plugin, "test_output_plugin")

