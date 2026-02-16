"""
Test execution logging configuration.

Provides comprehensive logging setup for test runs including:
- Detailed test execution logs
- Performance metrics
- Test output capture
- Result aggregation
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


class DAFLogFormatter(logging.Formatter):
    """Custom formatter for test logs with timestamps and levels."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with timestamp and level."""
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        
        level_colors = {
            "DEBUG": "\033[36m",  # Cyan
            "INFO": "\033[32m",   # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",  # Red
            "CRITICAL": "\033[35m",  # Magenta
        }
        reset_color = "\033[0m"
        
        level = record.levelname
        color = level_colors.get(level, "")
        
        # Format: [timestamp] [LEVEL] message
        message = f"[{timestamp}] [{color}{level}{reset_color}] {record.getMessage()}"
        
        # Add exception info if present
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"
        
        return message


def setup_test_logging(
    output_dir: Path,
    log_level: int = logging.INFO,
    also_print: bool = True,
) -> logging.Logger:
    """
    Configure logging for test runs.
    
    Args:
        output_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        also_print: Whether to also print to stdout
    
    Returns:
        Configured logger instance
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("cogames_tests")
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler - detailed log
    log_file = output_dir / f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = DAFLogFormatter()
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler - optional
    if also_print:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = DAFLogFormatter()
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger


class DAFExecutionTracker:
    """Track test execution metrics and timings."""
    
    def __init__(self, log_dir: Path):
        """Initialize tracker."""
        self.log_dir = log_dir
        self.suite_results = {}
        self.start_time = None
        self.end_time = None
        self.logger = setup_test_logging(log_dir)
    
    def start_suite(self, suite_name: str) -> None:
        """Mark start of test suite."""
        if suite_name not in self.suite_results:
            self.suite_results[suite_name] = {
                "start_time": datetime.now(),
                "end_time": None,
                "duration": None,
                "status": None,
            }
        self.logger.info(f"Starting test suite: {suite_name}")
    
    def end_suite(self, suite_name: str, status: str = "PASSED") -> None:
        """Mark end of test suite."""
        if suite_name in self.suite_results:
            self.suite_results[suite_name]["end_time"] = datetime.now()
            duration = (
                self.suite_results[suite_name]["end_time"]
                - self.suite_results[suite_name]["start_time"]
            ).total_seconds()
            self.suite_results[suite_name]["duration"] = duration
            self.suite_results[suite_name]["status"] = status
            
            self.logger.info(
                f"Completed test suite: {suite_name} - {status} ({duration:.2f}s)"
            )
    
    def log_test_result(
        self,
        suite_name: str,
        test_name: str,
        status: str,
        duration: float = 0.0,
        message: str = "",
    ) -> None:
        """Log individual test result."""
        status_symbol = "✓" if status == "PASSED" else "✗" if status == "FAILED" else "⏭"
        msg = f"{status_symbol} {suite_name}::{test_name} - {status}"
        
        if duration > 0:
            msg += f" ({duration:.2f}s)"
        
        if message:
            msg += f" - {message}"
        
        level = (
            logging.WARNING
            if status == "FAILED"
            else logging.DEBUG
            if status == "SKIPPED"
            else logging.INFO
        )
        self.logger.log(level, msg)
    
    def summary(self) -> str:
        """Generate execution summary."""
        lines = [
            "Test Execution Summary",
            "=" * 50,
        ]
        
        total_duration = 0
        total_suites = len(self.suite_results)
        passed_suites = 0
        
        for suite_name, result in self.suite_results.items():
            duration = result.get("duration", 0)
            status = result.get("status", "UNKNOWN")
            total_duration += duration
            
            if status == "PASSED":
                passed_suites += 1
            
            symbol = "✓" if status == "PASSED" else "✗" if status == "FAILED" else "?"
            lines.append(
                f"{symbol} {suite_name}: {status} ({duration:.2f}s)"
            )
        
        lines.extend([
            "=" * 50,
            f"Total Duration: {total_duration:.2f}s",
            f"Suites Passed: {passed_suites}/{total_suites}",
            f"Overall Status: {'PASSED' if passed_suites == total_suites else 'FAILED'}",
        ])
        
        return "\n".join(lines)


def create_test_log_file(
    output_dir: Path,
    test_file: str,
    test_name: str,
) -> Path:
    """Create timestamped log file for test execution."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"{test_file}_{test_name}_{timestamp}.log"
    
    return log_file


if __name__ == "__main__":
    # Example usage
    log_dir = Path("./test_logs")
    tracker = DAFExecutionTracker(log_dir)
    
    # Simulate test execution
    tracker.start_suite("test_config")
    tracker.log_test_result("test_config", "test_defaults", "PASSED", 0.15)
    tracker.log_test_result("test_config", "test_custom", "PASSED", 0.08)
    tracker.end_suite("test_config", "PASSED")
    
    tracker.start_suite("test_sweeps")
    tracker.log_test_result("test_sweeps", "test_grid_search", "PASSED", 0.42)
    tracker.log_test_result("test_sweeps", "test_random", "PASSED", 0.31)
    tracker.log_test_result("test_sweeps", "test_best_config", "FAILED", 0.05, "AssertionError")
    tracker.end_suite("test_sweeps", "FAILED")
    
    # Print summary
    print(tracker.summary())
    print(f"\nLog file: {log_dir}/cogames_tests_*.log")

