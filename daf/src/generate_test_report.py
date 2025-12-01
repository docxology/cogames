"""Generate comprehensive test reports from DAF test runs.

Collects test outputs from organized folders and generates summary reports,
statistics, and recommendations.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table


class TestReportGenerator:
    """Generate comprehensive test reports from saved test outputs."""

    def __init__(self, test_output_dir: Path | str):
        """Initialize report generator.

        Args:
            test_output_dir: Directory containing test output files
        """
        self.test_dir = Path(test_output_dir)
        self.console = Console()
        self.test_files = list(self.test_dir.glob("*_output.txt"))

    def parse_pytest_output(self, output: str) -> Dict[str, Any]:
        """Parse pytest output to extract statistics.

        Args:
            output: Raw pytest output

        Returns:
            Dictionary with parsed statistics
        """
        stats = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "duration": 0.0,
            "warnings": 0,
        }

        # Extract pass/fail summary line
        summary_pattern = r"(\d+)\s+passed"
        match = re.search(summary_pattern, output)
        if match:
            stats["passed"] = int(match.group(1))

        summary_pattern = r"(\d+)\s+failed"
        match = re.search(summary_pattern, output)
        if match:
            stats["failed"] = int(match.group(1))

        summary_pattern = r"(\d+)\s+skipped"
        match = re.search(summary_pattern, output)
        if match:
            stats["skipped"] = int(match.group(1))

        summary_pattern = r"(\d+)\s+warning"
        match = re.search(summary_pattern, output)
        if match:
            stats["warnings"] = int(match.group(1))

        # Extract duration
        duration_pattern = r"in\s+([\d.]+)s"
        match = re.search(duration_pattern, output)
        if match:
            stats["duration"] = float(match.group(1))

        stats["total"] = stats["passed"] + stats["failed"] + stats["skipped"]

        return stats

    def collect_all_results(self) -> Dict[str, Dict[str, Any]]:
        """Collect results from all test output files.

        Returns:
            Dictionary of results organized by test category
        """
        results = {}

        for test_file in sorted(self.test_files):
            test_name = test_file.stem.replace("_output", "")

            with open(test_file, "r") as f:
                output = f.read()

            stats = self.parse_pytest_output(output)
            results[test_name] = {
                "file": str(test_file),
                "stats": stats,
                "output_size_kb": test_file.stat().st_size / 1024,
            }

        return results

    def generate_summary(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics.

        Args:
            results: Collected test results

        Returns:
            Summary report
        """
        total_tests = sum(r["stats"]["total"] for r in results.values())
        total_passed = sum(r["stats"]["passed"] for r in results.values())
        total_failed = sum(r["stats"]["failed"] for r in results.values())
        total_skipped = sum(r["stats"]["skipped"] for r in results.values())
        total_warnings = sum(r["stats"]["warnings"] for r in results.values())
        total_duration = sum(r["stats"]["duration"] for r in results.values())

        return {
            "timestamp": datetime.now().isoformat(),
            "test_output_dir": str(self.test_dir),
            "summary": {
                "total_test_suites": len(results),
                "total_tests": total_tests,
                "total_passed": total_passed,
                "total_failed": total_failed,
                "total_skipped": total_skipped,
                "total_warnings": total_warnings,
                "total_duration_seconds": total_duration,
                "pass_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0,
            },
            "by_suite": results,
        }

    def print_summary(self, summary: Dict[str, Any]) -> None:
        """Print formatted summary to console.

        Args:
            summary: Summary report
        """
        s = summary["summary"]

        self.console.print("\n[bold cyan]" + "=" * 70 + "[/bold cyan]")
        self.console.print("[bold cyan]Test Execution Summary Report[/bold cyan]")
        self.console.print("[bold cyan]" + "=" * 70 + "[/bold cyan]\n")

        # Overall metrics
        self.console.print("[bold]Overall Metrics:[/bold]")
        metrics = [
            ("Test Suites", s["total_test_suites"]),
            ("Total Tests", s["total_tests"]),
            ("Passed", s["total_passed"]),
            ("Failed", s["total_failed"]),
            ("Skipped", s["total_skipped"]),
            ("Warnings", s["total_warnings"]),
            ("Pass Rate", f"{s['pass_rate']:.1f}%"),
            ("Total Duration", f"{s['total_duration_seconds']:.2f}s"),
        ]

        for label, value in metrics:
            if label == "Failed" and value > 0:
                color = "red"
            elif label == "Passed":
                color = "green"
            elif label in ["Skipped", "Warnings"] and value > 0:
                color = "yellow"
            else:
                color = "cyan"

            self.console.print(f"  {label:.<30} [{color}]{value}[/{color}]")

        # By suite breakdown
        if summary["by_suite"]:
            self.console.print("\n[bold]Results by Test Suite:[/bold]")

            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Test Suite")
            table.add_column("Passed", justify="right")
            table.add_column("Failed", justify="right")
            table.add_column("Skipped", justify="right")
            table.add_column("Duration (s)", justify="right")

            for suite_name, suite_result in sorted(summary["by_suite"].items()):
                stats = suite_result["stats"]
                status_color = "green" if stats["failed"] == 0 else "red"

                table.add_row(
                    f"[{status_color}]{suite_name}[/{status_color}]",
                    str(stats["passed"]),
                    str(stats["failed"]),
                    str(stats["skipped"]),
                    f"{stats['duration']:.2f}",
                )

            self.console.print(table)

    def save_summary(
        self,
        summary: Dict[str, Any],
        output_path: Path | str,
    ) -> Path:
        """Save summary report to JSON file.

        Args:
            summary: Summary report
            output_path: Output file path

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        return output_path

    def generate_html_report(
        self,
        summary: Dict[str, Any],
        output_path: Path | str,
    ) -> Path:
        """Generate HTML report.

        Args:
            summary: Summary report
            output_path: Output HTML file path

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        s = summary["summary"]

        # Determine color for pass rate
        pass_rate = s["pass_rate"]
        if pass_rate >= 95:
            rate_color = "#10b981"  # green
        elif pass_rate >= 80:
            rate_color = "#f59e0b"  # amber
        else:
            rate_color = "#ef4444"  # red

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>DAF Test Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f9fafb;
        }}
        .header {{
            background: #0f172a;
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2em;
        }}
        .header p {{
            margin: 5px 0 0 0;
            opacity: 0.8;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border-left: 4px solid #3b82f6;
        }}
        .metric-card.success {{
            border-left-color: #10b981;
        }}
        .metric-card.warning {{
            border-left-color: #f59e0b;
        }}
        .metric-card.error {{
            border-left-color: #ef4444;
        }}
        .metric-label {{
            font-size: 0.875rem;
            color: #6b7280;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #1f2937;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th {{
            background: #0f172a;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e5e7eb;
        }}
        tr:last-child td {{
            border-bottom: none;
        }}
        tr:hover {{
            background: #f3f4f6;
        }}
        .status-passed {{
            color: #10b981;
            font-weight: 600;
        }}
        .status-failed {{
            color: #ef4444;
            font-weight: 600;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #6b7280;
            font-size: 0.875rem;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>DAF Test Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="metrics">
        <div class="metric-card success">
            <div class="metric-label">Total Tests</div>
            <div class="metric-value">{s['total_tests']}</div>
        </div>
        <div class="metric-card success">
            <div class="metric-label">Passed</div>
            <div class="metric-value" style="color: #10b981;">{s['total_passed']}</div>
        </div>
        <div class="metric-card {'error' if s['total_failed'] > 0 else 'success'}">
            <div class="metric-label">Failed</div>
            <div class="metric-value" style="color: {'#ef4444' if s['total_failed'] > 0 else '#10b981'};">{s['total_failed']}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Pass Rate</div>
            <div class="metric-value" style="color: {rate_color};">{s['pass_rate']:.1f}%</div>
        </div>
    </div>

    <h2>Results by Test Suite</h2>
    <table>
        <thead>
            <tr>
                <th>Test Suite</th>
                <th>Passed</th>
                <th>Failed</th>
                <th>Skipped</th>
                <th>Duration (s)</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
"""

        for suite_name, suite_result in sorted(summary["by_suite"].items()):
            stats = suite_result["stats"]
            status = "✓ PASS" if stats["failed"] == 0 else "✗ FAIL"
            status_class = "status-passed" if stats["failed"] == 0 else "status-failed"

            html += f"""            <tr>
                <td>{suite_name}</td>
                <td>{stats['passed']}</td>
                <td>{stats['failed']}</td>
                <td>{stats['skipped']}</td>
                <td>{stats['duration']:.2f}</td>
                <td class="{status_class}">{status}</td>
            </tr>
"""

        html += """        </tbody>
    </table>

    <div class="footer">
        <p>DAF Test Report | Generated by test_runner.py</p>
    </div>
</body>
</html>"""

        with open(output_path, "w") as f:
            f.write(html)

        return output_path


def generate_report_from_outputs(
    test_output_dir: Path | str = "./daf_output/evaluations/tests",
    output_json: Optional[Path | str] = None,
    output_html: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Generate comprehensive test report from saved outputs.

    Args:
        test_output_dir: Directory with test output files
        output_json: Optional path to save JSON report
        output_html: Optional path to save HTML report

    Returns:
        Summary report dictionary
    """
    generator = TestReportGenerator(test_output_dir)

    # Collect results
    results = generator.collect_all_results()

    # Generate summary
    summary = generator.generate_summary(results)

    # Print to console
    generator.print_summary(summary)

    # Save reports
    if output_json:
        json_path = generator.save_summary(summary, output_json)
        print(f"\n✓ Saved JSON report: {json_path}")

    if output_html:
        html_path = generator.generate_html_report(summary, output_html)
        print(f"✓ Saved HTML report: {html_path}")

    return summary


if __name__ == "__main__":
    import sys

    test_dir = sys.argv[1] if len(sys.argv) > 1 else "./daf_output/evaluations/tests"
    output_json = sys.argv[2] if len(sys.argv) > 2 else None
    output_html = sys.argv[3] if len(sys.argv) > 3 else None

    generate_report_from_outputs(test_dir, output_json, output_html)







