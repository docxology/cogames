#!/usr/bin/env python3
"""
Generate comprehensive Markdown test report from pytest output.

This script parses pytest output files and generates a detailed Markdown report
with test statistics, timing information, and status summaries.
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from datetime import datetime


def load_test_plan(test_plan_file: Path) -> Optional[Dict]:
    """Load test plan JSON containing expected test counts."""
    if test_plan_file.exists():
        try:
            with open(test_plan_file, 'r') as f:
                data = json.load(f)
                print(f"✓ Loaded test plan: {test_plan_file}")
                return data
        except Exception as e:
            print(f"Warning: Could not load test plan from {test_plan_file}: {e}")
    return None


@dataclass
class TestResult:
    """Represents a single test result."""
    name: str
    status: str  # PASSED, FAILED, SKIPPED
    duration: float = 0.0
    file: str = ""


@dataclass
class SuiteResult:
    """Represents results from a test suite."""
    suite_name: str
    test_file: str
    status: str  # PASSED, FAILED
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration: float = 0.0
    expected_tests: int = 0
    tests: List[TestResult] = None
    
    def __post_init__(self):
        if self.tests is None:
            self.tests = []
    
    @property
    def pass_rate(self) -> float:
        """Calculate pass rate as percentage."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed / self.total_tests) * 100
    
    @property
    def test_count_match(self) -> bool:
        """Check if actual test count matches expected."""
        if self.expected_tests == 0:
            return True  # No expectation set
        return self.total_tests == self.expected_tests


def parse_pytest_output(output_file: Path) -> Tuple[SuiteResult, str]:
    """
    Parse pytest output file and extract test results.
    
    Returns:
        Tuple of (SuiteResult, raw_summary_line)
    """
    if not output_file.exists():
        return None, ""
    
    content = output_file.read_text()
    lines = content.split("\n")
    
    # Extract suite info from filename
    suite_name = output_file.stem.replace("_output", "").replace("cogames_", "").replace("daf_", "")
    suite_name = suite_name.replace("test_", "").replace("_", " ").title()
    
    # Look for pytest summary line
    summary_line = ""
    for line in lines:
        if " passed" in line or " failed" in line or " skipped" in line or " error" in line:
            if "warning" in line or "error" in line or "==" in line:
                summary_line = line
    
    # Parse counts from summary
    passed = failed = skipped = errors = 0
    total = 0
    duration = 0.0
    
    if summary_line:
        # Extract numbers using regex
        passed_match = re.search(r"(\d+) passed", summary_line)
        failed_match = re.search(r"(\d+) failed", summary_line)
        skipped_match = re.search(r"(\d+) skipped", summary_line)
        error_match = re.search(r"(\d+) error", summary_line)
        time_match = re.search(r"in ([\d.]+)s", summary_line)
        
        if passed_match:
            passed = int(passed_match.group(1))
        if failed_match:
            failed = int(failed_match.group(1))
        if skipped_match:
            skipped = int(skipped_match.group(1))
        if error_match:
            errors = int(error_match.group(1))
        if time_match:
            duration = float(time_match.group(1))
    
    # For reporting purposes, treat errors as failures
    total_failures = failed + errors
    total = passed + total_failures + skipped
    status = "FAILED" if total_failures > 0 else "PASSED"
    
    result = SuiteResult(
        suite_name=suite_name,
        test_file=output_file.name,
        status=status,
        total_tests=total,
        passed=passed,
        failed=failed,
        skipped=skipped,
        errors=errors,
        duration=duration,
    )
    
    return result, summary_line


def generate_markdown_report(
    phase1_results: List[SuiteResult],
    phase2_results: List[SuiteResult],
    output_file: Path,
    test_plan: Optional[Dict] = None,
) -> None:
    """Generate comprehensive Markdown test report with expected test counts."""
    
    # Calculate phase statistics
    phase1_total = sum(r.total_tests for r in phase1_results)
    phase1_passed = sum(r.passed for r in phase1_results)
    phase1_failed = sum(r.failed for r in phase1_results)
    phase1_skipped = sum(r.skipped for r in phase1_results)
    phase1_duration = sum(r.duration for r in phase1_results)
    phase1_status = "✅ PASSED" if all(r.status == "PASSED" for r in phase1_results) else "❌ FAILED"
    
    phase2_total = sum(r.total_tests for r in phase2_results)
    phase2_passed = sum(r.passed for r in phase2_results)
    phase2_failed = sum(r.failed for r in phase2_results)
    phase2_skipped = sum(r.skipped for r in phase2_results)
    phase2_duration = sum(r.duration for r in phase2_results)
    phase2_status = "✅ PASSED" if all(r.status == "PASSED" for r in phase2_results) else "❌ FAILED"
    
    total_tests = phase1_total + phase2_total
    total_passed = phase1_passed + phase2_passed
    total_failed = phase1_failed + phase2_failed
    total_skipped = phase1_skipped + phase2_skipped
    total_duration = phase1_duration + phase2_duration
    overall_status = "✅ SUCCESS" if (phase1_status.startswith("✅") and phase2_status.startswith("✅")) else "❌ FAILURE"
    
    # Calculate pass rates safely
    phase1_pass_rate = (phase1_passed / phase1_total * 100) if phase1_total > 0 else 0.0
    phase2_pass_rate = (phase2_passed / phase2_total * 100) if phase2_total > 0 else 0.0
    total_pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0.0
    
    # Generate markdown with proper f-strings - FIX: All variables now properly interpolated
    markdown = f"""# CoGames Full Test Suite Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Overall Status:** {overall_status}

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total Tests** | {total_tests} |
| **Passed** | {total_passed} ✅ |
| **Failed** | {total_failed} ❌ |
| **Skipped** | {total_skipped} ⏭️ |
| **Total Duration** | {total_duration:.2f}s |
| **Pass Rate** | {total_pass_rate:.1f}% |

---

## Test Plan

"""
    
    if test_plan:
        plan_phase1 = test_plan.get("phase1", {})
        plan_phase2 = test_plan.get("phase2", {})
        plan_total = test_plan.get("grand_total", 0)
        
        markdown += f"""### Expected vs Actual Test Counts

| Phase | Expected | Actual | Variance |
|-------|----------|--------|----------|
| Phase 1 (CoGames) | {plan_phase1.get('total', 0)} | {phase1_total} | {'+' if phase1_total >= plan_phase1.get('total', 0) else ''}{phase1_total - plan_phase1.get('total', 0)} |
| Phase 2 (DAF) | {plan_phase2.get('total', 0)} | {phase2_total} | {'+' if phase2_total >= plan_phase2.get('total', 0) else ''}{phase2_total - plan_phase2.get('total', 0)} |
| **Total** | **{plan_total}** | **{total_tests}** | **{'+' if total_tests >= plan_total else ''}{total_tests - plan_total}** |

**Note:** Positive variance indicates more tests collected than planned, negative indicates fewer.

"""
    else:
        markdown += """### Expected vs Actual Test Counts

No test plan data available. To enable test count validation:
- Run tests using `./daf/tests/run_daf_tests.sh` which generates test_plan.json
- Or pass test plan path as second argument: `generate_test_report.py <output_dir> <test_plan_file>`

"""
    
    markdown += f"""
---

## Phase 1: CoGames Core Tests

**Status:** {phase1_status}

### Summary

| Metric | Value |
|--------|-------|
| Tests Run | {phase1_total} |
| Passed | {phase1_passed} ✅ |
| Failed | {phase1_failed} ❌ |
| Skipped | {phase1_skipped} ⏭️ |
| Duration | {phase1_duration:.2f}s |
| Pass Rate | {phase1_pass_rate:.1f}% |

### Test Suites

| Status | Suite | Results | Time |
|--------|-------|---------|------|
"""
    
    for result in phase1_results:
        status_icon = "✅" if result.status == "PASSED" else "❌"
        markdown += f"| {status_icon} | {result.suite_name} | {result.passed}/{result.total_tests} | {result.duration:.2f}s |\n"
    
    markdown += f"""
---

## Phase 2: DAF Module Tests

**Status:** {phase2_status}

### Summary

| Metric | Value |
|--------|-------|
| Tests Run | {phase2_total} |
| Passed | {phase2_passed} ✅ |
| Failed | {phase2_failed} ❌ |
| Skipped | {phase2_skipped} ⏭️ |
| Duration | {phase2_duration:.2f}s |
| Pass Rate | {phase2_pass_rate:.1f}% |

### Test Suites

| Status | Suite | Results | Time |
|--------|-------|---------|------|
"""
    
    for result in phase2_results:
        status_icon = "✅" if result.status == "PASSED" else "❌"
        markdown += f"| {status_icon} | {result.suite_name} | {result.passed}/{result.total_tests} | {result.duration:.2f}s |\n"
    
    phase1_pct = (phase1_duration / total_duration * 100) if total_duration > 0 else 0.0
    phase2_pct = (phase2_duration / total_duration * 100) if total_duration > 0 else 0.0
    
    markdown += """
---

## Performance Analysis

### Slowest Test Suites

"""
    
    all_results = phase1_results + phase2_results
    slowest = sorted(all_results, key=lambda r: r.duration, reverse=True)[:5]
    
    for i, result in enumerate(slowest, 1):
        markdown += f"{i}. **{result.suite_name}** - {result.duration:.2f}s\n"
    
    markdown += f"""
### Test Distribution

- **Phase 1 (CoGames):** {phase1_pct:.1f}% of total time ({phase1_duration:.2f}s)
- **Phase 2 (DAF):** {phase2_pct:.1f}% of total time ({phase2_duration:.2f}s)

---

## Detailed Results

### Phase 1: CoGames Test Suites

"""
    
    for result in phase1_results:
        status_icon = "✅" if result.status == "PASSED" else "❌"
        expected_info = ""
        if result.expected_tests > 0:
            test_variance = result.total_tests - result.expected_tests
            variance_icon = "⚠️" if test_variance != 0 else "✓"
            expected_info = f"- **Expected Tests:** {result.expected_tests} (actual: {result.total_tests} {variance_icon})\n"
        
        # Build file links for different output formats
        file_base = result.test_file.replace("_output.txt", "")
        file_links = "- **Output Files:**\n"
        file_links += f"  - [📄 Text]({result.test_file})\n"
        file_links += f"  - [📊 JSON]({file_base}.json)\n"
        file_links += f"  - [✔️ JUnit XML]({file_base}.xml)"
        
        markdown += f"""#### {status_icon} {result.suite_name}

- **Status:** {result.status}
- **Tests:** {result.passed} passed, {result.failed} failed, {result.skipped} skipped ({result.total_tests} total)
{expected_info}- **Pass Rate:** {result.pass_rate:.1f}%
- **Duration:** {result.duration:.2f}s
{file_links}

"""
    
    markdown += """### Phase 2: DAF Test Suites

"""
    
    for result in phase2_results:
        status_icon = "✅" if result.status == "PASSED" else "❌"
        expected_info = ""
        if result.expected_tests > 0:
            test_variance = result.total_tests - result.expected_tests
            variance_icon = "⚠️" if test_variance != 0 else "✓"
            expected_info = f"- **Expected Tests:** {result.expected_tests} (actual: {result.total_tests} {variance_icon})\n"
        
        # Build file links for different output formats
        file_base = result.test_file.replace("_output.txt", "")
        file_links = "- **Output Files:**\n"
        file_links += f"  - [📄 Text]({result.test_file})\n"
        file_links += f"  - [📊 JSON]({file_base}.json)\n"
        file_links += f"  - [✔️ JUnit XML]({file_base}.xml)"
        
        markdown += f"""#### {status_icon} {result.suite_name}

- **Status:** {result.status}
- **Tests:** {result.passed} passed, {result.failed} failed, {result.skipped} skipped ({result.total_tests} total)
{expected_info}- **Pass Rate:** {result.pass_rate:.1f}%
- **Duration:** {result.duration:.2f}s
{file_links}

"""
    
    # Add failure details section if there are failures
    failed_suites = [r for r in phase1_results + phase2_results if r.status == "FAILED"]
    if failed_suites:
        markdown += """---

## Failed Test Details

"""
        for suite in failed_suites:
            markdown += f"""### {suite.suite_name}

- **Total Failures:** {suite.failed}
- **Failed Tests:** See output files for details
- **Output Files:**
  - [Full Text Output]({suite.test_file})
  - [JSON Report]({suite.test_file.replace('_output.txt', '.json')})
  - [JUnit XML]({suite.test_file.replace('_output.txt', '.xml')})

**Recommendations:**
1. Review the full test output for assertion failures
2. Check JSON report for detailed error information
3. Look for error tracebacks in the test output files

"""
    
    markdown += """---

## Recommendations

"""
    
    if overall_status.startswith("✅"):
        markdown += """✅ **All tests passed successfully!**

- Review test outputs for any warnings or performance concerns
- Monitor trends in test execution time
- Consider adding performance regression tests if needed
"""
    else:
        markdown += """❌ **Some tests failed. Action required:**

"""
        if phase1_status.startswith("❌"):
            markdown += """- **Phase 1 Failures:** Review CoGames core test failures in `daf_output/evaluations/tests/`
"""
        if phase2_status.startswith("❌"):
            markdown += """- **Phase 2 Failures:** Review DAF module test failures in `daf_output/evaluations/tests/`

"""
        markdown += """**Next Steps:**
1. Check detailed output files for error messages
2. Run individual failing tests with `-vv` flag for verbose output
3. Review recent code changes that may have introduced failures
4. Fix issues and re-run full test suite
"""
    
    markdown += """

---

## Output Files

Test output files are saved in: `daf_output/evaluations/tests/`

Each test suite produces:
- Individual output file: `*_output.txt`
- Detailed test logs with assertion details
- Full pytest output including timing

## Running Tests

To run the full test suite:

```bash
cd /Users/4d/Documents/GitHub/cogames
./daf/tests/run_daf_tests.sh
```

To run individual test suites:

```bash
# CoGames tests
uv run pytest tests/test_cli.py -v

# DAF tests
uv run pytest daf/tests/test_config.py -v
```

---

*Report generated by CoGames Test Infrastructure*
"""
    
    # Write report
    output_file.write_text(markdown)
    print(f"✓ Markdown report generated: {output_file}")


def main():
    """Main entry point."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: generate_test_report.py <output_dir> [test_plan_file]")
        print("  output_dir: Directory containing test output files")
        print("  test_plan_file: Optional - path to test_plan.json (auto-detected if in parent dir)")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    
    if not output_dir.exists():
        print(f"Error: Directory not found: {output_dir}")
        sys.exit(1)
    
    # Find all test output files
    output_files = sorted(output_dir.glob("*_output.txt"))
    
    if not output_files:
        print(f"No test output files found in: {output_dir}")
        sys.exit(1)
    
    # Load test plan if available - check multiple locations
    test_plan = None
    
    # Check if explicit test plan file provided
    if len(sys.argv) > 2:
        test_plan_file = Path(sys.argv[2])
        test_plan = load_test_plan(test_plan_file)
    else:
        # Try to find test plan in standard locations
        candidates = [
            output_dir.parent / "test_plan.json",  # evaluations/test_plan.json
            output_dir.parent.parent / "test_plan.json",  # daf_output/test_plan.json
            Path("./daf_output/test_plan.json"),  # Project root relative
        ]
        for test_plan_file in candidates:
            if test_plan_file.exists():
                test_plan = load_test_plan(test_plan_file)
                if test_plan:
                    break
    
    # Build maps of expected counts from test plan for quick lookup
    expected_counts = {}
    if test_plan:
        for suite in test_plan.get("phase1", {}).get("suites", []):
            expected_counts[suite["name"]] = suite["expected"]
        for suite in test_plan.get("phase2", {}).get("suites", []):
            expected_counts[suite["name"]] = suite["expected"]
    
    # Parse results
    phase1_results = []
    phase2_results = []
    
    for output_file in output_files:
        result, summary = parse_pytest_output(output_file)
        if result:
            # Set expected count if available from test plan
            if result.suite_name in expected_counts:
                result.expected_tests = expected_counts[result.suite_name]
            
            if "cogames" in output_file.stem or result.suite_name in [
                "Cli Tests",
                "Core Game Tests",
                "Cvc Assembler Hearts Tests",
                "Procedural Maps Tests",
                "Scripted Policies Tests",
                "Train Integration Tests",
                "Train Vector Alignment Tests",
                "All Games Describe Tests",
                "All Games Eval Tests",
                "All Games Play Tests",
            ]:
                phase1_results.append(result)
            else:
                phase2_results.append(result)
    
    # Generate markdown report
    report_file = output_dir.parent / "TEST_RUN_SUMMARY.md"
    generate_markdown_report(phase1_results, phase2_results, report_file, test_plan)
    
    # Also generate JSON for programmatic access
    json_file = output_dir.parent / "TEST_RUN_SUMMARY.json"
    json_data = {
        "generated": datetime.now().isoformat(),
        "phase1": [
            {
                "suite": r.suite_name,
                "status": r.status,
                "tests": r.total_tests,
                "passed": r.passed,
                "failed": r.failed,
                "skipped": r.skipped,
                "errors": r.errors,
                "duration": r.duration,
            }
            for r in phase1_results
        ],
        "phase2": [
            {
                "suite": r.suite_name,
                "status": r.status,
                "tests": r.total_tests,
                "passed": r.passed,
                "failed": r.failed,
                "skipped": r.skipped,
                "errors": r.errors,
                "duration": r.duration,
            }
            for r in phase2_results
        ],
    }
    json_file.write_text(json.dumps(json_data, indent=2))
    print(f"✓ JSON report generated: {json_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())
