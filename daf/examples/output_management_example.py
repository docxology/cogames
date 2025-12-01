"""Example: Using DAF output management and logging infrastructure.

Demonstrates:
- OutputManager for organized output directories
- DAFLogger for structured logging
- TestRunner for unified test execution
- Report generation
"""

from pathlib import Path

from daf.logging_config import DAFLogger, create_daf_logger
from daf.output_manager import OutputManager, get_output_manager
from daf.test_runner import TestRunner


def example_output_manager():
    """Example: Using OutputManager for organized outputs."""
    print("\n" + "=" * 60)
    print("Example 1: OutputManager")
    print("=" * 60 + "\n")

    # Create or get global output manager
    output_mgr = get_output_manager(
        base_dir="./daf_example_output",
        verbose=True,
        log_to_file=True,
    )

    # Print output structure
    output_mgr.print_output_info()

    # Log operation start
    output_mgr.log_operation_start(
        "sweep",
        details={
            "policy": "lstm",
            "missions": ["training_facility_1"],
            "num_trials": 10,
        },
    )

    # Save JSON results
    results = {
        "trials": [
            {"config": {"lr": 0.001}, "score": 0.85},
            {"config": {"lr": 0.01}, "score": 0.92},
        ],
        "best_score": 0.92,
    }

    results_file = output_mgr.save_json_results(
        results,
        operation="sweep",
        filename="sweep_results",
    )

    # Save summary report
    summary = {
        "total_trials": 2,
        "best_score": 0.92,
        "duration_seconds": 45.3,
    }

    summary_file = output_mgr.save_summary_report(
        operation="sweep",
        summary=summary,
    )

    # Log operation complete
    output_mgr.log_operation_complete(
        "sweep",
        status="success",
        details={"best_score": 0.92},
    )

    # Save session metadata
    metadata_file = output_mgr.save_session_metadata()

    print(f"\n✓ Results saved to: {results_file}")
    print(f"✓ Summary saved to: {summary_file}")
    print(f"✓ Metadata saved to: {metadata_file}")


def example_daf_logger():
    """Example: Using DAFLogger for structured logging."""
    print("\n" + "=" * 60)
    print("Example 2: DAFLogger with structured output")
    print("=" * 60 + "\n")

    # Create logger with metrics tracking
    logger = create_daf_logger(
        name="example_experiment",
        log_dir=Path("./daf_example_output/logs"),
        verbose=True,
    )

    # Print formatted sections
    logger.print_section("Training Experiment", level=1)
    logger.print_section("Phase 1: Data Preparation", level=2)

    # Track operations with context manager
    with logger.track_operation("load_dataset", metadata={"size": 1000}):
        logger.info("Loading dataset...")
        # Simulate work
        import time
        time.sleep(0.1)
        logger.debug("Dataset loaded: 1000 samples")

    # Track another operation
    with logger.track_operation("train_model", metadata={"epochs": 10}):
        logger.info("Starting training...")
        for epoch in range(3):
            logger.debug(f"Epoch {epoch + 1}/10 complete")
        logger.info("Training finished")

    # Print summary
    logger.print_section("Summary", level=2)
    logger.print_metrics_summary()

    # Save metrics
    metrics_file = logger.save_metrics_json(
        Path("./daf_example_output/reports/example_metrics.json")
    )
    print(f"\n✓ Metrics saved to: {metrics_file}")


def example_test_runner():
    """Example: Using TestRunner for test execution and reporting."""
    print("\n" + "=" * 60)
    print("Example 3: TestRunner for organized test execution")
    print("=" * 60 + "\n")

    runner = TestRunner(output_base_dir="./daf_example_output", verbose=True)

    # Run test suites (using actual test files from the project)
    test_suites = [
        ("tests/test_cli.py", "cogames", "cli"),
        ("daf/tests/test_config.py", "daf", "config"),
        ("daf/tests/test_environment_checks.py", "daf", "environment"),
    ]

    print("Running test suites...\n")
    results = runner.run_test_batch(test_suites)

    # Save test outputs
    test_output_dir = runner.save_test_outputs()
    print(f"\n✓ Test outputs saved to: {test_output_dir}")

    # Print summary
    runner.print_test_summary()

    # Print failed tests if any
    runner.print_failed_tests()

    # Save test report
    report_file = runner.save_test_report()
    print(f"\n✓ Test report saved to: {report_file}")

    # Cleanup
    runner.cleanup()


def example_directory_structure():
    """Example: Show resulting directory structure."""
    print("\n" + "=" * 60)
    print("Example 4: Resulting Directory Structure")
    print("=" * 60 + "\n")

    import os

    output_dir = Path("./daf_example_output")

    if output_dir.exists():
        print(f"Directory structure of {output_dir}:\n")

        for root, dirs, files in os.walk(output_dir):
            level = root.replace(str(output_dir), "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")

            subindent = " " * 2 * (level + 1)
            for file in sorted(files)[:10]:  # Show first 10 files per dir
                print(f"{subindent}{file}")

            if len(files) > 10:
                print(f"{subindent}... and {len(files) - 10} more files")
    else:
        print(f"Directory {output_dir} not yet created")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("DAF Output Management and Logging - Usage Examples")
    print("=" * 70)

    try:
        example_output_manager()
        example_daf_logger()
        # Uncomment to run actual tests (requires pytest environment)
        # example_test_runner()
        example_directory_structure()

        print("\n" + "=" * 70)
        print("✓ All examples completed successfully!")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()






