#!/bin/bash
# Run full CoGames test suite: top-level cogames tests + DAF tests
# IMPORTANT: This script must be run from the project root directory
#
# This script generates:
# - Individual test output files (both text and JSON)
# - test_plan.json with expected test counts
# - Comprehensive Markdown/JSON reports
#
# Usage: ./daf/tests/run_daf_tests.sh [OPTIONS]
# 
# Options:
#   --verbose              Enable verbose pytest output
#   --no-skipped           Skip previously skipped tests (marked with pytest.skip)
#   --only-daf             Run only DAF tests (skip CoGames tests)
#   --only-cogames         Run only CoGames tests (skip DAF tests)
#   --parallel             Run tests in parallel using pytest-xdist (auto-detects CPU count)
#   --workers N            Run with N parallel workers (implies --parallel)
#   --marker MARKER        Run tests matching pytest marker (e.g., --marker "not slow")
#   --help                 Show this help message

set -o pipefail

# Parse command line arguments
VERBOSE=false
SKIP_SKIPPED=false
ONLY_DAF=false
ONLY_COGAMES=false
PYTEST_MARKER=""
PYTEST_EXTRA_ARGS=""
PARALLEL_WORKERS=""

while [ $# -gt 0 ]; do
    case $1 in
        --verbose)
            VERBOSE=true
            PYTEST_EXTRA_ARGS="$PYTEST_EXTRA_ARGS -vv -s"
            shift
            ;;
        --no-skipped)
            SKIP_SKIPPED=true
            PYTEST_EXTRA_ARGS="$PYTEST_EXTRA_ARGS -p no:cacheprovider"
            shift
            ;;
        --only-daf)
            ONLY_DAF=true
            shift
            ;;
        --only-cogames)
            ONLY_COGAMES=true
            shift
            ;;
        --parallel)
            # Auto-detect CPU count
            PARALLEL_WORKERS="auto"
            shift
            ;;
        --workers)
            PARALLEL_WORKERS="$2"
            shift 2
            ;;
        --marker)
            PYTEST_MARKER="$2"
            PYTEST_EXTRA_ARGS="$PYTEST_EXTRA_ARGS -m $PYTEST_MARKER"
            shift 2
            ;;
        --help)
            grep "^#" "$0" | grep -E "^\s*#\s*(Usage|Options|^#$)" | sed 's/^#\s*//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Add parallel execution flag if specified
if [ -n "$PARALLEL_WORKERS" ]; then
    PYTEST_EXTRA_ARGS="$PYTEST_EXTRA_ARGS -n $PARALLEL_WORKERS"
    echo "Running tests in parallel with workers: $PARALLEL_WORKERS"
fi

# Setup output directories
OUTPUT_BASE="./daf_output"
TEST_OUTPUT="$OUTPUT_BASE/evaluations/tests"
LOG_DIR="$OUTPUT_BASE/logs"
TEST_PLAN_FILE="$OUTPUT_BASE/test_plan.json"

# Ensure we're in the project root
# Script is at: cogames/daf/tests/run_daf_tests.sh
# Need to go up TWO levels to reach cogames/
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Create output directories BEFORE any operations that write to them
mkdir -p "$TEST_OUTPUT" "$LOG_DIR"

# ============================================================================
# Helper function: Collect test counts using pytest --collect-only
# ============================================================================
collect_test_counts() {
    local test_file=$1
    if [ -f "$test_file" ]; then
        python -m pytest "$test_file" --collect-only -q 2>/dev/null | tail -1 | grep -oE '[0-9]+' | head -1 || echo "0"
    else
        echo "0"
    fi
}

# ============================================================================
# PHASE 0: Collect Upfront Test Counts
# ============================================================================
echo "=========================================="
echo "TEST COLLECTION PHASE: Counting Tests Before Execution"
echo "=========================================="
echo ""

# Initialize JSON structure
TEST_PLAN_JSON="{\"timestamp\": \"$(date -u +'%Y-%m-%dT%H:%M:%SZ')\", \"phase1\": {\"suites\": [], \"total\": 0}, \"phase2\": {\"suites\": [], \"total\": 0}}"

# Phase 1: CoGames tests
echo "Collecting CoGames test counts..."
COGAMES_TESTS=(
    "tests/test_cli.py:CLI Tests"
    "tests/test_cogs_vs_clips.py:Core Game Tests"
    "tests/test_cvc_assembler_hearts.py:CVC Assembler Hearts Tests"
    "tests/test_procedural_maps.py:Procedural Maps Tests"
    "tests/test_scripted_policies.py:Scripted Policies Tests"
    "tests/test_train_integration.py:Train Integration Tests"
    "tests/test_train_vector_alignment.py:Train Vector Alignment Tests"
    "tests/test_all_games_describe.py:All Games Describe Tests"
    "tests/test_all_games_eval.py:All Games Eval Tests"
    "tests/test_all_games_play.py:All Games Play Tests"
)

PHASE1_TOTAL=0
PHASE1_SUITES=""
for test_spec in "${COGAMES_TESTS[@]}"; do
    IFS=':' read -r test_file test_name <<< "$test_spec"
    if [ ! "$ONLY_DAF" = true ]; then
        count=$(collect_test_counts "$test_file")
        PHASE1_TOTAL=$((PHASE1_TOTAL + count))
        echo "  $test_name: $count tests"
        PHASE1_SUITES="$PHASE1_SUITES{\"name\": \"$test_name\", \"file\": \"$test_file\", \"expected\": $count},"
    fi
done
PHASE1_SUITES="${PHASE1_SUITES%,}"

# Phase 2: DAF tests
echo "Collecting DAF test counts..."
DAF_TESTS=(
    "daf/tests/test_config.py:Configuration Tests"
    "daf/tests/test_environment_checks.py:Environment Check Tests"
    "daf/tests/test_sweeps.py:Sweep Tests"
    "daf/tests/test_comparison.py:Comparison Tests"
    "daf/tests/test_deployment.py:Deployment Tests"
    "daf/tests/test_distributed_training.py:Distributed Training Tests"
    "daf/tests/test_visualization.py:Visualization Tests"
    "daf/tests/test_mission_analysis.py:Mission Analysis Tests"
)

PHASE2_TOTAL=0
PHASE2_SUITES=""
for test_spec in "${DAF_TESTS[@]}"; do
    IFS=':' read -r test_file test_name <<< "$test_spec"
    if [ ! "$ONLY_COGAMES" = true ]; then
        count=$(collect_test_counts "$test_file")
        PHASE2_TOTAL=$((PHASE2_TOTAL + count))
        echo "  $test_name: $count tests"
        PHASE2_SUITES="$PHASE2_SUITES{\"name\": \"$test_name\", \"file\": \"$test_file\", \"expected\": $count},"
    fi
done
PHASE2_SUITES="${PHASE2_SUITES%,}"

# Create test plan JSON
cat > "$TEST_PLAN_FILE" << EOF
{
  "timestamp": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')",
  "phase1": {
    "total": $PHASE1_TOTAL,
    "suites": [$PHASE1_SUITES]
  },
  "phase2": {
    "total": $PHASE2_TOTAL,
    "suites": [$PHASE2_SUITES]
  },
  "grand_total": $((PHASE1_TOTAL + PHASE2_TOTAL))
}
EOF

echo ""
echo "=========================================="
echo "TEST PLAN SUMMARY"
echo "=========================================="
echo "Phase 1 (CoGames): $PHASE1_TOTAL tests across 10 suites"
echo "Phase 2 (DAF):     $PHASE2_TOTAL tests across 8 suites"
echo "Total to execute:  $((PHASE1_TOTAL + PHASE2_TOTAL)) tests"
echo "Test plan saved to: $TEST_PLAN_FILE"
echo "=========================================="
echo ""

echo "=========================================="
echo "CoGames Full Test Suite - Comprehensive Verification"
echo "=========================================="
echo ""
echo "Running tests from: $(pwd)"
echo "Output directory: $OUTPUT_BASE"
echo ""

# ============================================================================
# PHASE 1: Run Top-Level CoGames Test Suite
# ============================================================================
echo "=========================================="
echo "PHASE 1: Top-Level CoGames Test Suite"
echo "=========================================="

# Skip phase 1 if only DAF tests requested
if [ "$ONLY_DAF" = true ]; then
    echo "Skipping CoGames tests (--only-daf specified)"
    echo ""
    COGAMES_TESTS_PASSED=true
else
    echo ""
    COGAMES_TESTS_PASSED=true

    # List of test files to run
    COGAMES_TESTS=(
        "tests/test_cli.py:CLI Tests"
        "tests/test_cogs_vs_clips.py:Core Game Tests"
        "tests/test_cvc_assembler_hearts.py:CVC Assembler Hearts Tests"
        "tests/test_procedural_maps.py:Procedural Maps Tests"
        "tests/test_scripted_policies.py:Scripted Policies Tests"
        "tests/test_train_integration.py:Train Integration Tests"
        "tests/test_train_vector_alignment.py:Train Vector Alignment Tests"
        "tests/test_all_games_describe.py:All Games Describe Tests"
        "tests/test_all_games_eval.py:All Games Eval Tests"
        "tests/test_all_games_play.py:All Games Play Tests"
    )

    # Run each CoGames test suite
    if [ "$ONLY_DAF" = false ]; then
        for test_spec in "${COGAMES_TESTS[@]}"; do
            IFS=':' read -r test_file test_name <<< "$test_spec"
            
            if [ -f "$test_file" ]; then
                echo "Running CoGames $test_name..."
                output_file="$TEST_OUTPUT/cogames_$(basename "$test_file" .py)_output.txt"
                
                python -m pytest "$test_file" -v $PYTEST_EXTRA_ARGS 2>&1 | tee "$output_file"
                test_exit_code=$?
                
                if [ $test_exit_code -eq 0 ]; then
                    echo "✓ $test_name passed"
                else
                    echo "✗ $test_name FAILED"
                    COGAMES_TESTS_PASSED=false
                fi
                echo ""
            else
                echo "⚠ $test_file not found (skipping)"
                echo ""
            fi
        done
    fi
fi

echo "=========================================="
if [ "$COGAMES_TESTS_PASSED" = true ]; then
    echo "PHASE 1 Complete: All CoGames Tests Passed"
else
    echo "PHASE 1 Complete: Some CoGames Tests Failed"
fi
echo "=========================================="
echo ""

# ============================================================================
# PHASE 2: Run DAF Test Suite
# ============================================================================
echo "=========================================="
echo "PHASE 2: DAF Test Suite - Comprehensive Verification"
echo "=========================================="

# Skip phase 2 if only CoGames tests requested
if [ "$ONLY_COGAMES" = true ]; then
    echo "Skipping DAF tests (--only-cogames specified)"
    echo ""
    DAF_TESTS_PASSED=true
else
    echo ""
    DAF_TESTS_PASSED=true

    # List of DAF test files to run
    DAF_TESTS=(
        "daf/tests/test_config.py:Configuration Tests"
        "daf/tests/test_environment_checks.py:Environment Check Tests"
        "daf/tests/test_sweeps.py:Sweep Tests"
        "daf/tests/test_comparison.py:Comparison Tests"
        "daf/tests/test_deployment.py:Deployment Tests"
        "daf/tests/test_distributed_training.py:Distributed Training Tests"
        "daf/tests/test_visualization.py:Visualization Tests"
        "daf/tests/test_mission_analysis.py:Mission Analysis Tests"
    )

    # Run each DAF test suite
    for test_spec in "${DAF_TESTS[@]}"; do
        IFS=':' read -r test_file test_name <<< "$test_spec"
        
        if [ -f "$test_file" ]; then
            echo "Running DAF $test_name..."
            output_file="$TEST_OUTPUT/daf_$(basename "$test_file" .py)_output.txt"
            
            python -m pytest "$test_file" -v $PYTEST_EXTRA_ARGS 2>&1 | tee "$output_file"
            test_exit_code=$?
            
            if [ $test_exit_code -eq 0 ]; then
                echo "✓ DAF $test_name passed"
            else
                echo "✗ DAF $test_name FAILED"
                DAF_TESTS_PASSED=false
            fi
            echo ""
        else
            echo "⚠ $test_file not found (skipping)"
            echo ""
        fi
    done
fi

# Note: Orchestrators tests require additional setup
echo "NOTE: Skipping DAF Orchestrators tests - requires additional infrastructure setup"
echo ""

echo "=========================================="
if [ "$DAF_TESTS_PASSED" = true ]; then
    echo "PHASE 2 Complete: All DAF Tests Passed"
else
    echo "PHASE 2 Complete: Some DAF Tests Failed"
fi
echo "=========================================="
echo ""

echo "=========================================="
echo "FULL TEST SUITE COMPLETE"
echo "=========================================="

if [ "$COGAMES_TESTS_PASSED" = true ] && [ "$DAF_TESTS_PASSED" = true ]; then
    echo "✓ Phase 1: All CoGames Tests Passed"
    echo "✓ Phase 2: All DAF Tests Passed"
    OVERALL_STATUS="SUCCESS"
else
    if [ "$COGAMES_TESTS_PASSED" = false ]; then
        echo "✗ Phase 1: Some CoGames Tests Failed"
    else
        echo "✓ Phase 1: All CoGames Tests Passed"
    fi
    
    if [ "$DAF_TESTS_PASSED" = false ]; then
        echo "✗ Phase 2: Some DAF Tests Failed"
    else
        echo "✓ Phase 2: All DAF Tests Passed"
    fi
    OVERALL_STATUS="FAILURE"
fi

echo "=========================================="
echo ""
echo "Test outputs saved to: $TEST_OUTPUT"
echo ""

# Generate comprehensive Markdown report
echo "Generating Markdown test report..."
python3 "$SCRIPT_DIR/generate_test_report.py" "$TEST_OUTPUT"
echo ""

# Also generate plain text summary for quick reference
cat > "$OUTPUT_BASE/TEST_RUN_SUMMARY.txt" << EOF
========================================
CoGames Full Test Suite Summary
========================================
Generated: $(date)
Status: $OVERALL_STATUS

OUTPUT FORMATS:
  - Text: Individual .txt files (verbose pytest output)
  - JSON: Individual .json files (structured test data)
  - JUnit: Individual .xml files (CI/CD compatible)
  - Markdown: TEST_RUN_SUMMARY.md (human-readable report)
  - JSON Summary: TEST_RUN_SUMMARY.json (aggregated results)

DIRECTORY STRUCTURE:
  Base:        $OUTPUT_BASE
  Tests:       $TEST_OUTPUT
  Logs:        $LOG_DIR

PHASE 1: CoGames Core Tests
  Status: $([ "$COGAMES_TESTS_PASSED" = true ] && echo "PASSED" || echo "FAILED")
  Tests run: 10 suites
  - CLI Tests
  - Core Game Tests
  - CVC Assembler Hearts Tests
  - Procedural Maps Tests
  - Scripted Policies Tests
  - Train Integration Tests
  - Train Vector Alignment Tests
  - All Games Describe Tests
  - All Games Eval Tests
  - All Games Play Tests

PHASE 2: DAF Module Tests
  Status: $([ "$DAF_TESTS_PASSED" = true ] && echo "PASSED" || echo "FAILED")
  Tests run: 8 suites
  - Configuration Tests
  - Environment Check Tests
  - Sweep Tests
  - Comparison Tests
  - Deployment Tests
  - Distributed Training Tests
  - Visualization Tests
  - Mission Analysis Tests

OUTPUT FILES:
  Individual test outputs: $TEST_OUTPUT/
  - Markdown Report: $OUTPUT_BASE/TEST_RUN_SUMMARY.md
  - JSON Report: $OUTPUT_BASE/TEST_RUN_SUMMARY.json
  - Text Summary: $OUTPUT_BASE/TEST_RUN_SUMMARY.txt

Each test file produces a separate .txt output file with full pytest output.

RECOMMENDED NEXT STEPS:
  1. Review Markdown report: $OUTPUT_BASE/TEST_RUN_SUMMARY.md
  2. For detailed test logs: Check files in $TEST_OUTPUT/
  3. For CoGames tests: See tests/*.py
  4. For DAF tests: See daf/tests/*.py

EOF

echo "Summary reports saved to:"
echo "  - Text: $OUTPUT_BASE/TEST_RUN_SUMMARY.txt"
echo "  - Markdown: $OUTPUT_BASE/TEST_RUN_SUMMARY.md"
echo "  - JSON: $OUTPUT_BASE/TEST_RUN_SUMMARY.json"
echo ""

# Optional: Clean up old test outputs (keep last 10 runs)
echo "To clean up old test outputs, run:"
echo "  python3 $SCRIPT_DIR/test_output_cleanup.py --keep 10 --output-dir $OUTPUT_BASE"
echo ""

# Exit with appropriate code
if [ "$COGAMES_TESTS_PASSED" = true ] && [ "$DAF_TESTS_PASSED" = true ]; then
    exit 0
else
    exit 1
fi
