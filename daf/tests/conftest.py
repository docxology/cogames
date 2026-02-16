"""
Pytest configuration and fixtures for DAF and CoGames tests.

Provides:
- Custom test markers
- Logging configuration
- Shared fixtures
- Output directory management
"""

# Pre-import torch with faulthandler disabled to prevent SIGABRT from
# torch's C extension conflicting with pytest's signal handlers on
# macOS ARM + Python 3.13. Once cached in sys.modules, subsequent
# imports are safe.
import faulthandler as _fh
_fh_was_enabled = _fh.is_enabled()
if _fh_was_enabled:
    _fh.disable()
try:
    import torch  # noqa: F401
except ImportError:
    pass  # torch not installed — GPU tests will be skipped
if _fh_was_enabled:
    _fh.enable()
del _fh, _fh_was_enabled

import pytest
import logging
from pathlib import Path
from datetime import datetime


# Register fixtures from fixtures.py module
pytest_plugins = ["daf.tests.fixtures"]


# ============================================================================
# Custom Markers
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests",
    )
    config.addinivalue_line(
        "markers",
        "unit: marks tests as unit tests",
    )
    config.addinivalue_line(
        "markers",
        "performance: marks tests as performance benchmarks",
    )
    config.addinivalue_line(
        "markers",
        "auth: marks tests requiring authentication",
    )


# ============================================================================
# Logging Configuration
# ============================================================================

@pytest.fixture(scope="session")
def test_log_dir():
    """Create and return test output directory."""
    log_dir = Path("./daf_output/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


@pytest.fixture(scope="session", autouse=True)
def configure_logging(test_log_dir):
    """Configure logging for test session."""
    # Create session log file
    log_file = test_log_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(formatter)
    
    logger.addHandler(fh)
    
    yield
    
    # Cleanup
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


# ============================================================================
# Output Fixtures
# ============================================================================

@pytest.fixture
def tmp_test_dir(tmp_path):
    """Provide a temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def output_dir():
    """Provide the main output directory."""
    output_dir = Path("./daf_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ============================================================================
# Timing Fixtures
# ============================================================================

@pytest.fixture
def timer():
    """Provide a simple timer for performance tracking."""
    class Timer:
        def __init__(self):
            self.times = {}
            self.start_time = None
        
        def start(self, label: str = "default"):
            """Start timing."""
            self.start_time = datetime.now()
            self.times[label] = {"start": self.start_time}
        
        def stop(self, label: str = "default"):
            """Stop timing and return duration."""
            if label not in self.times:
                self.times[label] = {}
            
            end_time = datetime.now()
            self.times[label]["end"] = end_time
            
            if "start" in self.times[label]:
                duration = (end_time - self.times[label]["start"]).total_seconds()
                self.times[label]["duration"] = duration
                return duration
            
            return 0.0
        
        def get(self, label: str = "default"):
            """Get duration for labeled timer."""
            if label in self.times and "duration" in self.times[label]:
                return self.times[label]["duration"]
            return None
    
    return Timer()


# ============================================================================
# Hooks for Enhanced Reporting
# ============================================================================

def pytest_runtest_logreport(report):
    """Capture test execution details."""
    if report.when == "call":
        # Log test result with timing
        test_name = report.nodeid.split("::")[-1]
        duration = report.duration
        
        if report.passed:
            logging.info(f"✓ {test_name} PASSED ({duration:.3f}s)")
        elif report.failed:
            logging.warning(f"✗ {test_name} FAILED ({duration:.3f}s)")
        elif report.skipped:
            logging.info(f"⏭ {test_name} SKIPPED ({duration:.3f}s)")


def pytest_collection_modifyitems(config, items):
    """Modify test collection for custom behavior."""
    # Check if pytest-timeout plugin is available
    has_timeout_plugin = config.pluginmanager.has_plugin("timeout")
    
    for item in items:
        # Add timeout marker if not already present and plugin is available
        if has_timeout_plugin and "timeout" not in [marker.name for marker in item.iter_markers()]:
            # Default: 5 minutes per test
            item.add_marker(pytest.mark.timeout(300))


# ============================================================================
# Assertion Helpers
# ============================================================================

@pytest.fixture
def assert_performance():
    """Provide performance assertion helper."""
    def check_performance(actual: float, expected_max: float, metric_name: str = "metric"):
        """Assert performance metric is within expected bounds."""
        if actual > expected_max:
            pytest.fail(
                f"Performance degradation: {metric_name} took {actual:.3f}s "
                f"(expected max: {expected_max:.3f}s)"
            )
        return True
    
    return check_performance


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def mission_names():
    """Provide common test mission names using real CogsGuard sites."""
    return [
        "cogsguard_machina_1",
        "cogsguard_arena",
        "training_facility.open_world",
        "hello_world",
    ]


@pytest.fixture
def cogsguard_missions():
    """Provide CogsGuard-specific mission names."""
    return [
        "cogsguard_machina_1",
        "cogsguard_arena",
    ]


@pytest.fixture
def variant_names():
    """Provide common variant names for testing."""
    return [
        "dark_side",
        "energized",
        "dense",
    ]


@pytest.fixture
def policy_names():
    """Provide common test policy names."""
    return [
        "random",
        "baseline",
        "lstm",
        "stateless",
    ]


# ============================================================================
# Test Execution Tracker
# ============================================================================

@pytest.fixture
def execution_tracker():
    """Fixture for tracking test execution details."""
    class ExecutionTracker:
        def __init__(self):
            self.events = []
            self.start_time = datetime.now()
        
        def record_event(self, event_type: str, details: dict = None):
            """Record an execution event."""
            self.events.append({
                "timestamp": datetime.now().isoformat(),
                "type": event_type,
                "details": details or {},
            })
        
        def get_summary(self) -> dict:
            """Get summary of tracked events."""
            return {
                "start_time": self.start_time.isoformat(),
                "event_count": len(self.events),
                "events": self.events,
            }
    
    return ExecutionTracker()

