#!/usr/bin/env python3
"""
Reliable pytest JSON report parser.

Uses pytest's built-in JSON output instead of fragile regex parsing.
Provides structured access to test results with full details.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class TestDetails:
    """Details for a single test."""
    name: str
    status: str  # passed, failed, skipped
    duration: float
    outcome: str  # passed, failed, skipped
    file: str
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None


@dataclass
class SuiteReport:
    """Report for a test suite."""
    suite_name: str
    file: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration: float
    status: str  # PASSED or FAILED
    tests: List[TestDetails]
    
    @property
    def pass_rate(self) -> float:
        """Calculate pass rate as percentage."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed / self.total_tests) * 100


def parse_pytest_json_output(json_file: Path) -> Optional[Dict[str, Any]]:
    """
    Parse pytest JSON output file.
    
    Args:
        json_file: Path to pytest report JSON file
        
    Returns:
        Parsed JSON data or None if file not found/invalid
    """
    if not json_file.exists():
        return None
    
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not parse {json_file}: {e}")
        return None


def extract_suite_report(suite_name: str, test_file: str, json_data: Dict) -> SuiteReport:
    """
    Extract suite report from pytest JSON output.
    
    Args:
        suite_name: Human-readable suite name
        test_file: Path to test file
        json_data: Parsed pytest JSON output
        
    Returns:
        SuiteReport with extracted data
    """
    tests: List[TestDetails] = []
    passed = failed = skipped = errors = 0
    total_duration = 0.0
    
    for test_item in json_data.get("tests", []):
        outcome = test_item.get("outcome", "unknown")
        duration = test_item.get("duration", 0.0)
        total_duration += duration
        
        error_msg = None
        error_trace = None
        if "call" in test_item:
            call_data = test_item["call"]
            if call_data.get("outcome") == "failed":
                error_msg = call_data.get("longrepr", "")[:200]
                error_trace = call_data.get("longrepr", "")
        
        test_detail = TestDetails(
            name=test_item.get("nodeid", "unknown"),
            status=outcome,
            duration=duration,
            outcome=outcome,
            file=test_item.get("file", test_file),
            error_message=error_msg,
            error_traceback=error_trace,
        )
        tests.append(test_detail)
        
        if outcome == "passed":
            passed += 1
        elif outcome == "failed":
            failed += 1
            errors += 1
        elif outcome == "skipped":
            skipped += 1
    
    total_tests = passed + failed + skipped
    status = "FAILED" if failed > 0 else "PASSED"
    
    return SuiteReport(
        suite_name=suite_name,
        file=test_file,
        total_tests=total_tests,
        passed=passed,
        failed=failed,
        skipped=skipped,
        errors=errors,
        duration=total_duration,
        status=status,
        tests=tests,
    )


def parse_pytest_output_directory(output_dir: Path, json_suffix: str = "_output.json") -> Dict[str, SuiteReport]:
    """
    Parse all pytest JSON output files in a directory.
    
    Args:
        output_dir: Directory containing pytest JSON output files
        json_suffix: Suffix pattern for JSON files
        
    Returns:
        Dictionary mapping suite names to SuiteReport objects
    """
    reports = {}
    
    json_files = sorted(output_dir.glob(f"*{json_suffix}"))
    
    for json_file in json_files:
        # Extract suite name from filename
        suite_name = json_file.stem.replace("_output", "").replace("cogames_", "").replace("daf_", "")
        suite_name = suite_name.replace("test_", "").replace("_", " ").title()
        
        # Parse JSON
        json_data = parse_pytest_json_output(json_file)
        if not json_data:
            continue
        
        # Extract report
        test_file = json_data.get("summary", {}).get("file", "")
        report = extract_suite_report(suite_name, test_file, json_data)
        reports[suite_name] = report
    
    return reports


def merge_reports(reports: Dict[str, SuiteReport]) -> Dict[str, Any]:
    """
    Merge individual suite reports into summary statistics.
    
    Args:
        reports: Dictionary of SuiteReport objects
        
    Returns:
        Aggregated statistics dictionary
    """
    total_tests = sum(r.total_tests for r in reports.values())
    total_passed = sum(r.passed for r in reports.values())
    total_failed = sum(r.failed for r in reports.values())
    total_skipped = sum(r.skipped for r in reports.values())
    total_duration = sum(r.duration for r in reports.values())
    
    pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0.0
    
    return {
        "total_tests": total_tests,
        "passed": total_passed,
        "failed": total_failed,
        "skipped": total_skipped,
        "duration": total_duration,
        "pass_rate": pass_rate,
        "suites": len(reports),
    }


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: pytest_json_parser.py <output_dir>")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    reports = parse_pytest_output_directory(output_dir)
    
    for suite_name, report in reports.items():
        print(f"{suite_name}: {report.passed}/{report.total_tests} ({report.pass_rate:.1f}%)")

