#!/usr/bin/env python3
"""
Modern async test orchestrator for CoGames test suite.

Replaces bash script with pure Python async/await execution.
Provides concurrent test suite execution, comprehensive error handling,
and structured output management.
"""

import asyncio
import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
import subprocess
from enum import Enum

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn


class TestStatus(Enum):
    """Test suite execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TestSuiteConfig:
    """Configuration for a single test suite."""
    name: str
    file: str
    phase: int  # Phase 1 (CoGames) or Phase 2 (DAF)
    expected_count: int = 0


@dataclass
class TestResult:
    """Result of test suite execution."""
    name: str
    file: str
    status: TestStatus
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    total: int = 0
    duration: float = 0.0
    exit_code: int = 0
    error_message: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None


class TestOrchestrator:
    """Modern async test orchestrator."""

    def __init__(
        self,
        output_base: Path = Path("./daf_output"),
        max_workers: Optional[int] = None,
        verbose: bool = False,
        collect_only: bool = False,
    ):
        """Initialize orchestrator.
        
        Args:
            output_base: Base output directory
            max_workers: Max concurrent test suites (None = CPU count)
            verbose: Enable verbose output
            collect_only: Only collect test counts, don't execute
        """
        self.output_base = Path(output_base)
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        self.test_output_dir = self.output_base / "evaluations" / "tests"
        self.test_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = self.output_base / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_workers = max_workers or min(8, len(asyncio.all_tasks()) + 4)
        self.verbose = verbose
        self.collect_only = collect_only
        
        self.console = Console()
        self.results: Dict[str, TestResult] = {}
        self.test_plan: Dict[str, int] = {}
        
    async def run_test_suite(
        self,
        suite_config: TestSuiteConfig,
        progress_task: Optional[int] = None,
    ) -> TestResult:
        """Run a single test suite asynchronously.
        
        Args:
            suite_config: Test suite configuration
            progress_task: Progress task ID for updating progress bar
            
        Returns:
            TestResult with execution details
        """
        result = TestResult(
            name=suite_config.name,
            file=suite_config.file,
            status=TestStatus.RUNNING,
            start_time=datetime.now().isoformat(),
        )
        
        try:
            # Check if test file exists
            if not Path(suite_config.file).exists():
                result.status = TestStatus.SKIPPED
                result.error_message = f"Test file not found: {suite_config.file}"
                return result
            
            # Build pytest command
            cmd = [
                "python", "-m", "pytest",
                suite_config.file,
                "-v", "--tb=short",
                "--json-report",
                f"--json-report-file={self.test_output_dir}/{suite_config.name.lower().replace(' ', '_')}_output.json",
            ]
            
            # Add verbosity if requested
            if self.verbose:
                cmd.append("-vv")
            
            # Run test suite
            start = time.time()
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=600.0  # 10 minute timeout
            )
            
            duration = time.time() - start
            exit_code = process.returncode
            
            result.duration = duration
            result.exit_code = exit_code
            result.end_time = datetime.now().isoformat()
            
            # Parse results from stdout
            self._parse_pytest_output(result, stdout)
            
            # Determine status
            if exit_code == 0:
                result.status = TestStatus.PASSED
            else:
                result.status = TestStatus.FAILED
                result.error_message = stderr[:500] if stderr else None
            
        except asyncio.TimeoutError:
            result.status = TestStatus.FAILED
            result.error_message = "Test execution timed out (10 minutes)"
            result.end_time = datetime.now().isoformat()
        except Exception as e:
            result.status = TestStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now().isoformat()
        
        return result
    
    def _parse_pytest_output(self, result: TestResult, output: str) -> None:
        """Parse pytest output to extract test counts.
        
        Args:
            result: TestResult to update
            output: pytest output text
        """
        import re
        
        # Look for pytest summary line like "185 passed in 210.80s"
        for line in output.split("\n"):
            if "passed" in line or "failed" in line:
                # Extract counts
                passed_match = re.search(r"(\d+) passed", line)
                failed_match = re.search(r"(\d+) failed", line)
                skipped_match = re.search(r"(\d+) skipped", line)
                
                if passed_match:
                    result.passed = int(passed_match.group(1))
                if failed_match:
                    result.failed = int(failed_match.group(1))
                if skipped_match:
                    result.skipped = int(skipped_match.group(1))
                
                result.total = result.passed + result.failed + result.skipped
                break
    
    async def run_phase(
        self,
        phase: int,
        suites: List[TestSuiteConfig],
        progress: Optional[Progress] = None,
    ) -> Tuple[bool, List[TestResult]]:
        """Run all test suites in a phase concurrently.
        
        Args:
            phase: Phase number (1 or 2)
            suites: List of test suite configurations
            progress: Progress bar object
            
        Returns:
            Tuple of (all_passed, results)
        """
        self.console.print(f"\n[bold]PHASE {phase}: Running {len(suites)} test suites[/bold]")
        
        # Create semaphore to limit concurrent execution
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def limited_run(suite: TestSuiteConfig) -> TestResult:
            async with semaphore:
                task_id = None
                if progress:
                    task_id = progress.add_task(f"[cyan]{suite.name}...", total=None)
                
                result = await self.run_test_suite(suite, task_id)
                
                if progress and task_id is not None:
                    progress.update(task_id, completed=True)
                
                return result
        
        # Run all suites concurrently
        results = await asyncio.gather(*[limited_run(suite) for suite in suites])
        
        # Check if all passed
        all_passed = all(r.status == TestStatus.PASSED for r in results)
        
        # Print phase summary
        passed = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in results if r.status == TestStatus.FAILED)
        skipped = sum(1 for r in results if r.status == TestStatus.SKIPPED)
        
        status = "[green]✓ PASSED[/green]" if all_passed else "[red]✗ FAILED[/red]"
        self.console.print(
            f"Phase {phase}: {status} ({passed} passed, {failed} failed, {skipped} skipped)"
        )
        
        return all_passed, results
    
    async def run_all(
        self,
        cogames_suites: List[TestSuiteConfig],
        daf_suites: List[TestSuiteConfig],
    ) -> int:
        """Run all test phases.
        
        Args:
            cogames_suites: CoGames test suite configs
            daf_suites: DAF test suite configs
            
        Returns:
            Exit code (0 if all passed, 1 if any failed)
        """
        self.console.rule("[bold cyan]CoGames Test Suite Execution[/bold cyan]")
        
        start_time = time.time()
        phase1_passed = phase2_passed = True
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            # Phase 1: CoGames tests
            if cogames_suites:
                phase1_passed, phase1_results = await self.run_phase(
                    1, cogames_suites, progress
                )
                for result in phase1_results:
                    self.results[result.name] = result
            
            # Phase 2: DAF tests
            if daf_suites:
                phase2_passed, phase2_results = await self.run_phase(
                    2, daf_suites, progress
                )
                for result in phase2_results:
                    self.results[result.name] = result
        
        # Print overall summary
        total_time = time.time() - start_time
        self._print_summary(phase1_passed, phase2_passed, total_time)
        
        # Save results
        self._save_results()
        
        return 0 if (phase1_passed and phase2_passed) else 1
    
    def _print_summary(
        self,
        phase1_passed: bool,
        phase2_passed: bool,
        total_time: float,
    ) -> None:
        """Print execution summary.
        
        Args:
            phase1_passed: Whether phase 1 passed
            phase2_passed: Whether phase 2 passed
            total_time: Total execution time
        """
        self.console.rule("[bold cyan]Execution Summary[/bold cyan]")
        
        total_suites = len(self.results)
        passed_suites = sum(1 for r in self.results.values() if r.status == TestStatus.PASSED)
        failed_suites = sum(1 for r in self.results.values() if r.status == TestStatus.FAILED)
        
        total_tests = sum(r.total for r in self.results.values())
        passed_tests = sum(r.passed for r in self.results.values())
        failed_tests = sum(r.failed for r in self.results.values())
        skipped_tests = sum(r.skipped for r in self.results.values())
        
        self.console.print(f"\n[bold]Overall Status[/bold]: ", end="")
        if phase1_passed and phase2_passed:
            self.console.print("[green]✓ ALL TESTS PASSED[/green]")
        else:
            self.console.print("[red]✗ SOME TESTS FAILED[/red]")
        
        self.console.print(f"\n[bold]Test Suites[/bold]:")
        self.console.print(f"  Total:   {total_suites}")
        self.console.print(f"  Passed:  {passed_suites} [green]✓[/green]")
        self.console.print(f"  Failed:  {failed_suites} [red]✗[/red]")
        
        self.console.print(f"\n[bold]Tests[/bold]:")
        self.console.print(f"  Total:   {total_tests}")
        self.console.print(f"  Passed:  {passed_tests} [green]✓[/green]")
        self.console.print(f"  Failed:  {failed_tests} [red]✗[/red]")
        self.console.print(f"  Skipped: {skipped_tests} [yellow]⏭️[/yellow]")
        
        self.console.print(f"\n[bold]Performance[/bold]:")
        self.console.print(f"  Total time: {total_time:.2f}s")
        self.console.print(f"  Avg suite:  {total_time/total_suites:.2f}s" if total_suites else "  No suites")
    
    def _save_results(self) -> None:
        """Save execution results to JSON file."""
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "suites": {
                name: asdict(result)
                for name, result in self.results.items()
            },
        }
        
        # Convert status enum to string
        for suite_data in results_data["suites"].values():
            if isinstance(suite_data.get("status"), TestStatus):
                suite_data["status"] = suite_data["status"].value
        
        output_file = self.output_base / "test_results.json"
        with open(output_file, "w") as f:
            json.dump(results_data, f, indent=2, default=str)
        
        self.console.print(f"\n✓ Results saved to: {output_file}")


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Modern async test orchestrator for CoGames"
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Max concurrent test suites (default: CPU count)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--output-dir",
        default="./daf_output",
        help="Output directory (default: ./daf_output)"
    )
    parser.add_argument(
        "--only-cogames",
        action="store_true",
        help="Run only CoGames tests"
    )
    parser.add_argument(
        "--only-daf",
        action="store_true",
        help="Run only DAF tests"
    )
    
    args = parser.parse_args()
    
    # Define test suites
    cogames_suites = [
        TestSuiteConfig("CLI Tests", "tests/test_cli.py", 1, 6),
        TestSuiteConfig("Core Game Tests", "tests/test_cogs_vs_clips.py", 1, 4),
        TestSuiteConfig("CVC Assembler Hearts Tests", "tests/test_cvc_assembler_hearts.py", 1, 2),
        TestSuiteConfig("Procedural Maps Tests", "tests/test_procedural_maps.py", 1, 11),
        TestSuiteConfig("Scripted Policies Tests", "tests/test_scripted_policies.py", 1, 13),
        TestSuiteConfig("Train Integration Tests", "tests/test_train_integration.py", 1, 2),
        TestSuiteConfig("Train Vector Alignment Tests", "tests/test_train_vector_alignment.py", 1, 5),
        TestSuiteConfig("All Games Describe Tests", "tests/test_all_games_describe.py", 1, 47),
        TestSuiteConfig("All Games Eval Tests", "tests/test_all_games_eval.py", 1, 48),
        TestSuiteConfig("All Games Play Tests", "tests/test_all_games_play.py", 1, 47),
    ]
    
    daf_suites = [
        TestSuiteConfig("Configuration Tests", "daf/tests/test_config.py", 2, 15),
        TestSuiteConfig("Environment Check Tests", "daf/tests/test_environment_checks.py", 2, 13),
        TestSuiteConfig("Sweep Tests", "daf/tests/test_sweeps.py", 2, 16),
        TestSuiteConfig("Comparison Tests", "daf/tests/test_comparison.py", 2, 12),
        TestSuiteConfig("Deployment Tests", "daf/tests/test_deployment.py", 2, 11),
        TestSuiteConfig("Distributed Training Tests", "daf/tests/test_distributed_training.py", 2, 12),
        TestSuiteConfig("Visualization Tests", "daf/tests/test_visualization.py", 2, 9),
        TestSuiteConfig("Mission Analysis Tests", "daf/tests/test_mission_analysis.py", 2, 12),
    ]
    
    # Filter suites based on arguments
    if args.only_cogames:
        daf_suites = []
    elif args.only_daf:
        cogames_suites = []
    
    # Create orchestrator and run
    orchestrator = TestOrchestrator(
        output_base=Path(args.output_dir),
        max_workers=args.workers,
        verbose=args.verbose,
    )
    
    exit_code = await orchestrator.run_all(cogames_suites, daf_suites)
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())

