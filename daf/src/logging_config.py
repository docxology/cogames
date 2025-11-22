"""Structured logging configuration for DAF operations.

Provides enhanced logging utilities with consistent formatting, structured output,
and performance metrics tracking. Includes functional call tracing for debugging.
"""

from __future__ import annotations

import functools
import json
import logging
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Generator, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table


@dataclass
class OperationMetrics:
    """Metrics for a single operation execution."""

    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"
    error_message: Optional[str] = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def duration_seconds(self) -> float:
        """Get operation duration in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "operation_name": self.operation_name,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": (
                datetime.fromtimestamp(self.end_time).isoformat()
                if self.end_time
                else None
            ),
            "duration_seconds": self.duration_seconds,
            "status": self.status,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class FunctionCallRecord:
    """Record of a single function call for tracing."""

    function_name: str
    module: str
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None
    start_time: float = 0.0
    end_time: Optional[float] = None
    duration_ms: float = 0.0
    call_stack: list = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "function": f"{self.module}.{self.function_name}",
            "args": str(self.args)[:100],  # Truncate for readability
            "result": str(self.result)[:100] if self.result else None,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "timestamp": datetime.fromtimestamp(self.start_time).isoformat(),
        }


class CallTracker:
    """Track function calls for debugging and performance analysis."""

    def __init__(self, max_records: int = 1000):
        """Initialize call tracker."""
        self.max_records = max_records
        self.calls: list[FunctionCallRecord] = []

    def record_call(
        self,
        function_name: str,
        module: str,
        args: tuple,
        kwargs: dict,
        result: Any = None,
        error: Optional[str] = None,
        duration_ms: float = 0.0,
    ) -> None:
        """Record a function call."""
        # Get call stack (skip first few frames)
        stack = traceback.extract_stack()[:-3]
        call_stack = [f"{frame.filename}:{frame.lineno} in {frame.name}" for frame in stack[-3:]]

        record = FunctionCallRecord(
            function_name=function_name,
            module=module,
            args=args,
            kwargs=kwargs,
            result=result,
            error=error,
            start_time=time.time(),
            duration_ms=duration_ms,
            call_stack=call_stack,
        )

        self.calls.append(record)

        # Keep only max_records
        if len(self.calls) > self.max_records:
            self.calls = self.calls[-self.max_records :]

    def get_summary(self) -> dict[str, Any]:
        """Get summary of tracked calls."""
        return {
            "total_calls": len(self.calls),
            "calls": [call.to_dict() for call in self.calls],
        }

    def clear(self) -> None:
        """Clear all recorded calls."""
        self.calls.clear()


def track_function_calls(logger: Optional[logging.Logger] = None) -> Callable:
    """Decorator to track function calls with logging.

    Args:
        logger: Optional logger instance for detailed logging

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            module = func.__module__
            func_name = func.__name__

            if logger:
                logger.debug(f"CALL: {module}.{func_name}({args[:2]}, ...)")

            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000

                if logger:
                    logger.debug(f"RETURN: {func_name} ({duration_ms:.2f}ms)")

                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                error_msg = f"{type(e).__name__}: {str(e)}"

                if logger:
                    logger.error(f"ERROR: {func_name} ({duration_ms:.2f}ms) - {error_msg}")

                raise

        return wrapper

    return decorator


class OperationTracker:
    """Track metrics and timing for DAF operations."""

    def __init__(self):
        """Initialize operation tracker."""
        self.operations: list[OperationMetrics] = []
        self.current_operation: Optional[OperationMetrics] = None

    @contextmanager
    def track(
        self,
        operation_name: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Generator[OperationMetrics, None, None]:
        """Context manager to track operation execution.

        Args:
            operation_name: Name of operation being tracked
            metadata: Optional operation-specific metadata

        Yields:
            OperationMetrics instance for this operation
        """
        metrics = OperationMetrics(
            operation_name=operation_name,
            start_time=time.time(),
            metadata=metadata or {},
        )

        previous_operation = self.current_operation
        self.current_operation = metrics

        try:
            yield metrics
            metrics.status = "success"
            metrics.end_time = time.time()
        except Exception as e:
            metrics.status = "failed"
            metrics.error_message = str(e)
            metrics.end_time = time.time()
            raise
        finally:
            self.operations.append(metrics)
            self.current_operation = previous_operation

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all tracked operations.

        Returns:
            Dictionary with operation counts and statistics
        """
        total = len(self.operations)
        successful = sum(1 for op in self.operations if op.status == "success")
        failed = sum(1 for op in self.operations if op.status == "failed")
        total_time = sum(op.duration_seconds for op in self.operations)

        return {
            "total_operations": total,
            "successful": successful,
            "failed": failed,
            "total_duration_seconds": total_time,
            "operations": [op.to_dict() for op in self.operations],
        }


class DAFLogger:
    """Enhanced logger for DAF operations with structured output.

    Provides:
    - Consistent formatting across operations
    - Performance metrics tracking
    - Rich console output
    - Structured JSON logging
    """

    def __init__(
        self,
        name: str = "daf",
        log_file: Optional[Path] = None,
        verbose: bool = False,
        track_calls: bool = False,
    ):
        """Initialize DAF logger.

        Args:
            name: Logger name
            log_file: Optional file to write logs to
            verbose: Enable verbose output
            track_calls: Enable function call tracking
        """
        self.name = name
        self.log_file = log_file
        self.verbose = verbose
        self.console = Console()
        self.tracker = OperationTracker()
        self.call_tracker = CallTracker() if track_calls else None

        # Get or create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)

        # Clear existing handlers
        self.logger.handlers.clear()

        # Setup handlers
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Setup logging handlers."""
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO if not self.verbose else logging.DEBUG)
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler if specified
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, msg: str, *args, **kwargs) -> None:
        """Log info message."""
        self.logger.info(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        """Log error message."""
        self.logger.error(msg, *args, **kwargs)

    def start_operation(
        self,
        operation_name: str,
        details: Optional[dict[str, Any]] = None,
    ) -> OperationMetrics:
        """Log operation start.

        Args:
            operation_name: Name of operation
            details: Optional operation details

        Returns:
            OperationMetrics instance
        """
        self.info(f"Starting: {operation_name}")
        if details:
            for key, value in details.items():
                self.debug(f"  {key}: {value}")

        metrics = OperationMetrics(
            operation_name=operation_name,
            start_time=time.time(),
            metadata=details or {},
        )
        self.tracker.current_operation = metrics
        return metrics

    def end_operation(
        self,
        metrics: OperationMetrics,
        status: str = "success",
        summary: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log operation completion.

        Args:
            metrics: OperationMetrics from start_operation
            status: Completion status
            summary: Optional completion summary
        """
        metrics.status = status
        metrics.end_time = time.time()

        msg = f"Completed {metrics.operation_name}: {status} ({metrics.duration_seconds:.2f}s)"
        if status == "success":
            self.info(msg)
        else:
            self.warning(msg)

        if summary:
            for key, value in summary.items():
                self.debug(f"  {key}: {value}")

    def print_section(self, title: str, level: int = 1) -> None:
        """Print formatted section header.

        Args:
            title: Section title
            level: Header level (1, 2, 3)
        """
        if level == 1:
            self.console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
            self.console.print(f"[bold cyan]{title.center(60)}[/bold cyan]")
            self.console.print(f"[bold cyan]{'=' * 60}[/bold cyan]\n")
        elif level == 2:
            self.console.print(f"\n[bold]{title}[/bold]")
            self.console.print("[cyan]" + "-" * len(title) + "[/cyan]\n")
        else:
            self.console.print(f"[cyan]• {title}[/cyan]")

    def print_table(
        self,
        title: str,
        data: list[dict[str, Any]],
        columns: Optional[list[str]] = None,
    ) -> None:
        """Print formatted table.

        Args:
            title: Table title
            data: List of dictionaries to display
            columns: Optional list of column names to include
        """
        if not data:
            self.console.print(f"[yellow]{title}: No data[/yellow]\n")
            return

        # Use provided columns or all keys from first row
        if columns is None:
            columns = list(data[0].keys())

        table = Table(title=title, show_header=True, header_style="bold magenta")

        for col in columns:
            table.add_column(col)

        for row in data:
            table.add_row(*[str(row.get(col, "")) for col in columns])

        self.console.print(table)

    def print_metrics_summary(self) -> None:
        """Print summary of tracked operations."""
        summary = self.tracker.get_summary()

        self.print_section("Operation Metrics Summary", level=1)

        metrics_data = [
            {
                "Total Operations": summary["total_operations"],
                "Successful": summary["successful"],
                "Failed": summary["failed"],
                "Total Time (s)": f"{summary['total_duration_seconds']:.2f}",
            }
        ]

        self.console.print(
            Table.grid(
                *[
                    f"[cyan]{k}[/cyan]: {v}"
                    for k, v in metrics_data[0].items()
                ]
            )
        )

        if summary["operations"]:
            ops_table_data = [
                {
                    "Operation": op["operation_name"],
                    "Status": op["status"],
                    "Duration (s)": f"{op['duration_seconds']:.2f}",
                }
                for op in summary["operations"]
            ]
            self.print_table("Individual Operations", ops_table_data)

    def save_metrics_json(self, output_path: Path | str) -> Path:
        """Save operation metrics to JSON file.

        Args:
            output_path: Path to save metrics

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = self.tracker.get_summary()
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        self.info(f"Saved metrics: {output_path}")
        return output_path

    def record_call(
        self,
        function_name: str,
        module: str,
        args: tuple,
        kwargs: dict,
        result: Any = None,
        error: Optional[str] = None,
        duration_ms: float = 0.0,
    ) -> None:
        """Record a function call if tracking is enabled.

        Args:
            function_name: Name of function
            module: Module name
            args: Function arguments
            kwargs: Function keyword arguments
            result: Function return value
            error: Error message if function failed
            duration_ms: Duration in milliseconds
        """
        if self.call_tracker:
            self.call_tracker.record_call(
                function_name, module, args, kwargs, result, error, duration_ms
            )

    def save_call_traces(self, output_path: Path | str) -> Path:
        """Save recorded function calls to file.

        Args:
            output_path: Path to save traces

        Returns:
            Path to saved file
        """
        if not self.call_tracker:
            self.warning("Call tracking not enabled")
            return Path(output_path)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = self.call_tracker.get_summary()
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        self.info(f"Saved call traces: {output_path}")
        return output_path

    @contextmanager
    def track_operation(
        self,
        operation_name: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Generator[OperationMetrics, None, None]:
        """Context manager for tracking operations.

        Args:
            operation_name: Name of operation
            metadata: Optional metadata

        Yields:
            OperationMetrics instance
        """
        metrics = self.start_operation(operation_name, metadata)
        try:
            yield metrics
            self.end_operation(metrics, status="success")
        except Exception as e:
            self.end_operation(metrics, status="error", summary={"error": str(e)})
            raise


def create_daf_logger(
    name: str = "daf",
    log_dir: Optional[Path] = None,
    verbose: bool = False,
    track_calls: bool = False,
) -> DAFLogger:
    """Create and configure DAF logger.

    Args:
        name: Logger name
        log_dir: Optional directory for log files
        verbose: Enable verbose output
        track_calls: Enable function call tracking

    Returns:
        Configured DAFLogger instance
    """
    log_file = None
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    return DAFLogger(name=name, log_file=log_file, verbose=verbose, track_calls=track_calls)

