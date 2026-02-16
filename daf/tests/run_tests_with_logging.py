#!/usr/bin/env python3
"""
Main test orchestration script with structured logging.

Executes test suites with OutputManager and DAFLogger for comprehensive
logging, reporting, and metrics tracking.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from test_runner_logging import DAFRunnerLogger


def collect_test_counts(test_file: str) -> int:
    """Collect test count for a test file."""
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", test_file, "--collect-only", "-q"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout
        lines = output.strip().split("\n")
        if lines:
            last_line = lines[-1]
            # Extract number from output like "185 test collected"
            for word in last_line.split():
                if word.isdigit():
                    return int(word)
    except Exception as e:
        print(f"Warning: Could not collect tests from {test_file}: {e}")
    return 0


def run_test_suite(test_file: str, pytest_args: str = "") -> Tuple[bool, int, str]:
    """
    Run a single test suite.
    
    Returns:
        Tuple of (passed, exit_code, output)
    """
    try:
        cmd = f"python -m pytest {test_file} -v {pytest_args}"
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
        passed = result.returncode == 0
        return passed, result.returncode, result.stdout
    except subprocess.TimeoutExpired:
        return False, 124, "Test suite timed out"
    except Exception as e:
        return False, 1, str(e)


def main():
    """Main entry point for test orchestration."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run CoGames test suite with structured logging"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--only-daf",
        action="store_true",
        help="Run only DAF tests"
    )
    parser.add_argument(
        "--only-cogames",
        action="store_true",
        help="Run only CoGames tests"
    )
    parser.add_argument(
        "--output-dir",
        default="./daf_output",
        help="Output directory for logs and reports"
    )
    
    args = parser.parse_args()
    
    # Initialize logger
    runner_logger = DAFRunnerLogger(
        output_base=args.output_dir,
        verbose=args.verbose
    )
    
    # CoGames test suites
    cogames_tests = [
        ("tests/test_cli.py", "CLI Tests"),
        ("tests/test_cogs_vs_clips.py", "Core Game Tests"),
        ("tests/test_cvc_assembler_hearts.py", "CVC Assembler Hearts Tests"),
        ("tests/test_procedural_maps.py", "Procedural Maps Tests"),
        ("tests/test_scripted_policies.py", "Scripted Policies Tests"),
        ("tests/test_train_integration.py", "Train Integration Tests"),
        ("tests/test_train_vector_alignment.py", "Train Vector Alignment Tests"),
        ("tests/test_all_games_describe.py", "All Games Describe Tests"),
        ("tests/test_all_games_eval.py", "All Games Eval Tests"),
        ("tests/test_all_games_play.py", "All Games Play Tests"),
    ]
    
    # DAF test suites
    daf_tests = [
        ("daf/tests/test_config.py", "Configuration Tests"),
        ("daf/tests/test_environment_checks.py", "Environment Check Tests"),
        ("daf/tests/test_sweeps.py", "Sweep Tests"),
        ("daf/tests/test_comparison.py", "Comparison Tests"),
        ("daf/tests/test_deployment.py", "Deployment Tests"),
        ("daf/tests/test_distributed_training.py", "Distributed Training Tests"),
        ("daf/tests/test_visualization.py", "Visualization Tests"),
        ("daf/tests/test_mission_analysis.py", "Mission Analysis Tests"),
    ]
    
    # Phase 1: Collect test counts
    phase1_total = 0
    phase2_total = 0
    
    if not args.only_daf:
        for test_file, _ in cogames_tests:
            count = collect_test_counts(test_file)
            phase1_total += count
    
    if not args.only_cogames:
        for test_file, _ in daf_tests:
            count = collect_test_counts(test_file)
            phase2_total += count
    
    runner_logger.log_test_collection_phase(phase1_total, phase2_total, phase1_total + phase2_total)
    
    # Phase 1: Run CoGames tests
    phase1_passed = True
    if not args.only_daf:
        runner_logger.log_phase_start(1, "CoGames Core Tests")
        
        for test_file, suite_name in cogames_tests:
            if Path(test_file).exists():
                runner_logger.log_suite_start(suite_name, test_file)
                passed, exit_code, output = run_test_suite(test_file)
                
                # Extract duration from output if available
                duration = 0.0
                for line in output.split("\n"):
                    if " passed" in line and " in " in line:
                        try:
                            parts = line.split(" in ")
                            if len(parts) > 1:
                                duration_str = parts[1].split("s")[0]
                                duration = float(duration_str)
                        except:
                            pass
                
                runner_logger.log_suite_result(suite_name, passed, exit_code, duration)
                
                if not passed:
                    phase1_passed = False
        
        runner_logger.log_phase_complete(1, phase1_passed)
    
    # Phase 2: Run DAF tests
    phase2_passed = True
    if not args.only_cogames:
        runner_logger.log_phase_start(2, "DAF Module Tests")
        
        for test_file, suite_name in daf_tests:
            if Path(test_file).exists():
                runner_logger.log_suite_start(suite_name, test_file)
                passed, exit_code, output = run_test_suite(test_file)
                
                # Extract duration from output if available
                duration = 0.0
                for line in output.split("\n"):
                    if " passed" in line and " in " in line:
                        try:
                            parts = line.split(" in ")
                            if len(parts) > 1:
                                duration_str = parts[1].split("s")[0]
                                duration = float(duration_str)
                        except:
                            pass
                
                runner_logger.log_suite_result(suite_name, passed, exit_code, duration)
                
                if not passed:
                    phase2_passed = False
        
        runner_logger.log_phase_complete(2, phase2_passed)
    
    # Final summary
    runner_logger.log_full_test_complete(phase1_passed, phase2_passed)
    runner_logger.print_summary()
    runner_logger.save_metrics()
    
    # Exit with appropriate code
    if phase1_passed and phase2_passed:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())

