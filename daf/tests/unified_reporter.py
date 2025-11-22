#!/usr/bin/env python3
"""
Unified JSON-based reporting system for test results.

Consolidates all test output into single JSON format with embedded metadata.
Generates Markdown and HTML reports from unified JSON data.
"""

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from enum import Enum


class ReportFormat(Enum):
    """Report output format."""
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"


@dataclass
class TestMetrics:
    """Test execution metrics."""
    total_tests: int
    passed: int
    failed: int
    skipped: int
    duration_seconds: float
    pass_rate: float = 0.0
    
    def __post_init__(self):
        if self.total_tests > 0:
            self.pass_rate = (self.passed / self.total_tests) * 100


@dataclass
class SuiteResult:
    """Single test suite result."""
    name: str
    file: str
    phase: int
    status: str  # "passed", "failed", "skipped"
    metrics: TestMetrics
    start_time: str
    end_time: str
    duration_seconds: float
    error_message: Optional[str] = None
    expected_tests: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["metrics"] = asdict(self.metrics)
        return data


@dataclass
class TestRunReport:
    """Complete test run report."""
    timestamp: str
    session_id: str
    total_duration_seconds: float
    suites: List[SuiteResult] = field(default_factory=list)
    
    @property
    def metrics(self) -> TestMetrics:
        """Aggregate metrics across all suites."""
        total = sum(s.metrics.total_tests for s in self.suites)
        passed = sum(s.metrics.passed for s in self.suites)
        failed = sum(s.metrics.failed for s in self.suites)
        skipped = sum(s.metrics.skipped for s in self.suites)
        
        return TestMetrics(
            total_tests=total,
            passed=passed,
            failed=failed,
            skipped=skipped,
            duration_seconds=self.total_duration_seconds,
        )
    
    @property
    def phase1_suites(self) -> List[SuiteResult]:
        """Get Phase 1 suites."""
        return [s for s in self.suites if s.phase == 1]
    
    @property
    def phase2_suites(self) -> List[SuiteResult]:
        """Get Phase 2 suites."""
        return [s for s in self.suites if s.phase == 2]
    
    @property
    def failed_suites(self) -> List[SuiteResult]:
        """Get failed suites."""
        return [s for s in self.suites if s.status == "failed"]
    
    @property
    def slowest_suites(self, top_n: int = 5) -> List[SuiteResult]:
        """Get slowest suites."""
        return sorted(
            self.suites,
            key=lambda s: s.duration_seconds,
            reverse=True,
        )[:top_n]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "total_duration_seconds": self.total_duration_seconds,
            "metrics": asdict(self.metrics),
            "suites": [s.to_dict() for s in self.suites],
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


class UnifiedReporter:
    """Unified reporting system."""
    
    def __init__(self, output_dir: Path = Path("./daf_output")):
        """Initialize reporter.
        
        Args:
            output_dir: Output directory for reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_json_report(
        self,
        report: TestRunReport,
        filename: str = "test_report.json",
    ) -> Path:
        """Save report as JSON.
        
        Args:
            report: Test run report
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_file = self.output_dir / filename
        output_file.write_text(report.to_json())
        return output_file
    
    def generate_markdown_report(
        self,
        report: TestRunReport,
        filename: str = "TEST_RUN_SUMMARY.md",
    ) -> Path:
        """Generate Markdown report from JSON data.
        
        Args:
            report: Test run report
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        metrics = report.metrics
        
        md = f"""# CoGames Test Suite Report

**Generated**: {report.timestamp}

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total Tests** | {metrics.total_tests} |
| **Passed** | {metrics.passed} ✅ |
| **Failed** | {metrics.failed} ❌ |
| **Skipped** | {metrics.skipped} ⏭️ |
| **Total Duration** | {report.total_duration_seconds:.2f}s |
| **Pass Rate** | {metrics.pass_rate:.1f}% |

---

## Phase 1: CoGames Core Tests

### Summary

| Metric | Value |
|--------|-------|
| Tests Run | {sum(s.metrics.total_tests for s in report.phase1_suites)} |
| Passed | {sum(s.metrics.passed for s in report.phase1_suites)} ✅ |
| Failed | {sum(s.metrics.failed for s in report.phase1_suites)} ❌ |
| Skipped | {sum(s.metrics.skipped for s in report.phase1_suites)} ⏭️ |
| Pass Rate | {sum(s.metrics.passed for s in report.phase1_suites) / max(1, sum(s.metrics.total_tests for s in report.phase1_suites)) * 100:.1f}% |

### Test Suites

| Status | Suite | Results | Time |
|--------|-------|---------|------|
"""
        
        for suite in report.phase1_suites:
            status_icon = "✅" if suite.status == "passed" else "❌"
            md += f"| {status_icon} | {suite.name} | {suite.metrics.passed}/{suite.metrics.total_tests} | {suite.duration_seconds:.2f}s |\n"
        
        md += "\n---\n\n## Phase 2: DAF Module Tests\n\n### Summary\n\n| Metric | Value |\n|--------|-------|\n"
        md += f"| Tests Run | {sum(s.metrics.total_tests for s in report.phase2_suites)} |\n"
        md += f"| Passed | {sum(s.metrics.passed for s in report.phase2_suites)} ✅ |\n"
        md += f"| Failed | {sum(s.metrics.failed for s in report.phase2_suites)} ❌ |\n"
        md += f"| Skipped | {sum(s.metrics.skipped for s in report.phase2_suites)} ⏭️ |\n"
        md += f"| Pass Rate | {sum(s.metrics.passed for s in report.phase2_suites) / max(1, sum(s.metrics.total_tests for s in report.phase2_suites)) * 100:.1f}% |\n"
        md += "\n### Test Suites\n\n| Status | Suite | Results | Time |\n|--------|-------|---------|------|\n"
        
        for suite in report.phase2_suites:
            status_icon = "✅" if suite.status == "passed" else "❌"
            md += f"| {status_icon} | {suite.name} | {suite.metrics.passed}/{suite.metrics.total_tests} | {suite.duration_seconds:.2f}s |\n"
        
        if report.failed_suites:
            md += "\n---\n\n## Failed Tests\n\n"
            for suite in report.failed_suites:
                md += f"### ❌ {suite.name}\n\n"
                if suite.error_message:
                    md += f"**Error**: {suite.error_message}\n\n"
                md += f"- **Status**: {suite.status.upper()}\n"
                md += f"- **Tests**: {suite.metrics.passed} passed, {suite.metrics.failed} failed\n"
                md += f"- **Duration**: {suite.duration_seconds:.2f}s\n\n"
        
        md += "\n---\n\n## Performance Analysis\n\n### Slowest Test Suites\n\n"
        for i, suite in enumerate(sorted(
            report.suites,
            key=lambda s: s.duration_seconds,
            reverse=True,
        )[:5], 1):
            md += f"{i}. **{suite.name}** - {suite.duration_seconds:.2f}s\n"
        
        output_file = self.output_dir / filename
        output_file.write_text(md)
        return output_file
    
    def generate_html_report(
        self,
        report: TestRunReport,
        filename: str = "test_report.html",
    ) -> Path:
        """Generate HTML report from JSON data.
        
        Args:
            report: Test run report
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        metrics = report.metrics
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>CoGames Test Report</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
            border-left: 4px solid #007bff;
            padding-left: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #007bff;
            color: white;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .passed {{
            color: green;
            font-weight: bold;
        }}
        .failed {{
            color: red;
            font-weight: bold;
        }}
        .skipped {{
            color: orange;
            font-weight: bold;
        }}
        .metric-box {{
            display: inline-block;
            padding: 15px 25px;
            margin: 10px;
            background: #f8f9fa;
            border-radius: 4px;
            border-left: 4px solid #007bff;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧪 CoGames Test Report</h1>
        <p>Generated: {report.timestamp}</p>
        
        <h2>Executive Summary</h2>
        <div>
            <div class="metric-box">
                <div class="metric-value">{metrics.total_tests}</div>
                <div class="metric-label">Total Tests</div>
            </div>
            <div class="metric-box">
                <div class="metric-value passed">{metrics.passed}</div>
                <div class="metric-label">Passed</div>
            </div>
            <div class="metric-box">
                <div class="metric-value failed">{metrics.failed}</div>
                <div class="metric-label">Failed</div>
            </div>
            <div class="metric-box">
                <div class="metric-value skipped">{metrics.skipped}</div>
                <div class="metric-label">Skipped</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{metrics.pass_rate:.1f}%</div>
                <div class="metric-label">Pass Rate</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{report.total_duration_seconds:.1f}s</div>
                <div class="metric-label">Duration</div>
            </div>
        </div>
        
        <h2>Phase 1: CoGames Core Tests</h2>
        <table>
            <thead>
                <tr>
                    <th>Status</th>
                    <th>Suite</th>
                    <th>Results</th>
                    <th>Time</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for suite in report.phase1_suites:
            status_icon = "✅" if suite.status == "passed" else "❌"
            html += f"""                <tr>
                    <td>{status_icon}</td>
                    <td>{suite.name}</td>
                    <td>{suite.metrics.passed}/{suite.metrics.total_tests}</td>
                    <td>{suite.duration_seconds:.2f}s</td>
                </tr>
"""
        
        html += """            </tbody>
        </table>
        
        <h2>Phase 2: DAF Module Tests</h2>
        <table>
            <thead>
                <tr>
                    <th>Status</th>
                    <th>Suite</th>
                    <th>Results</th>
                    <th>Time</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for suite in report.phase2_suites:
            status_icon = "✅" if suite.status == "passed" else "❌"
            html += f"""                <tr>
                    <td>{status_icon}</td>
                    <td>{suite.name}</td>
                    <td>{suite.metrics.passed}/{suite.metrics.total_tests}</td>
                    <td>{suite.duration_seconds:.2f}s</td>
                </tr>
"""
        
        html += """            </tbody>
        </table>
    </div>
</body>
</html>"""
        
        output_file = self.output_dir / filename
        output_file.write_text(html)
        return output_file


if __name__ == "__main__":
    # Example usage
    report = TestRunReport(
        timestamp=datetime.now().isoformat(),
        session_id="test_session",
        total_duration_seconds=353.54,
        suites=[
            SuiteResult(
                name="CLI Tests",
                file="tests/test_cli.py",
                phase=1,
                status="passed",
                metrics=TestMetrics(6, 6, 0, 0, 7.70),
                start_time=datetime.now().isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=7.70,
                expected_tests=6,
            ),
        ],
    )
    
    reporter = UnifiedReporter()
    reporter.save_json_report(report)
    reporter.generate_markdown_report(report)
    reporter.generate_html_report(report)
    print("Reports generated successfully!")

