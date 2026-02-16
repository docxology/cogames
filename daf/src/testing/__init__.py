"""DAF testing infrastructure: test runner and report generation.

Submodules:
    test_runner          Unified test execution with structured output
    generate_test_report HTML/JSON report generation from test results
"""

from __future__ import annotations

__all__ = [
    "TestRunner",
    "TestResult",
    "TestReportGenerator",
    "generate_report_from_outputs",
]

_MODULE_MAP = {
    "TestRunner": "daf.src.testing.test_runner",
    "TestResult": "daf.src.testing.test_runner",
    "TestReportGenerator": "daf.src.testing.generate_test_report",
    "generate_report_from_outputs": "daf.src.testing.generate_test_report",
}


def __getattr__(name: str):
    """Lazy import of testing submodule attributes."""
    if name in _MODULE_MAP:
        import importlib
        mod = importlib.import_module(_MODULE_MAP[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'daf.src.testing' has no attribute '{name}'")
