#!/bin/bash
# Run comprehensive test suite: core CoGames tests first, then DAF sidecar tests
# IMPORTANT: This script must be run from the project root directory

set -e

echo "=================================================================="
echo "Comprehensive Test Suite - CoGames + DAF Sidecar"
echo "=================================================================="
echo ""

# Ensure we're in the project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Running from: $(pwd)"
echo ""

# ============================================================================
# PART 1: Core CoGames Tests
# ============================================================================
echo "=========================================="
echo "PART 1: Core CoGames Tests"
echo "=========================================="
echo ""

echo "Running core CoGames game tests..."
uv run python -m pytest tests/test_all_games_describe.py tests/test_all_games_eval.py tests/test_all_games_play.py -v

echo ""
echo "Running CoGames CLI tests..."
uv run python -m pytest tests/test_cli.py -v

echo ""
echo "Running CoGames Cogs vs Clips tests..."
uv run python -m pytest tests/test_cogs_vs_clips.py tests/test_cvc_assembler_hearts.py -v

echo ""
echo "Running CoGames procedural and scripted tests..."
uv run python -m pytest tests/test_procedural_maps.py tests/test_scripted_policies.py -v

echo ""
echo "Running CoGames training integration tests..."
uv run python -m pytest tests/test_train_integration.py tests/test_train_vector_alignment.py -v

echo ""
echo "=========================================="
echo "Core CoGames Tests Complete"
echo "=========================================="
echo ""

# ============================================================================
# PART 2: DAF Sidecar Tests
# ============================================================================
echo "=========================================="
echo "PART 2: DAF Sidecar Tests"
echo "=========================================="
echo ""

echo "Running DAF Configuration Tests..."
uv run python -m pytest daf/tests/test_config.py -v

echo ""
echo "Running DAF Environment Check Tests..."
uv run python -m pytest daf/tests/test_environment_checks.py -v

echo ""
echo "Running DAF Sweep Tests..."
uv run python -m pytest daf/tests/test_sweeps.py -v

echo ""
echo "Running DAF Orchestrator Tests (VERIFY ORDERING)..."
uv run python -m pytest daf/tests/test_orchestrators.py -v

echo ""
echo "Running DAF Comparison Tests..."
uv run python -m pytest daf/tests/test_comparison.py -v

echo ""
echo "Running DAF Deployment Tests..."
uv run python -m pytest daf/tests/test_deployment.py -v

echo ""
echo "Running DAF Distributed Training Tests..."
uv run python -m pytest daf/tests/test_distributed_training.py -v

echo ""
echo "Running DAF Visualization Tests..."
uv run python -m pytest daf/tests/test_visualization.py -v

echo ""
echo "Running DAF Mission Analysis Tests..."
uv run python -m pytest daf/tests/test_mission_analysis.py -v

echo ""
echo "=========================================="
echo "DAF Sidecar Tests Complete"
echo "=========================================="
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo "=================================================================="
echo "All Tests Complete: Core CoGames + DAF Sidecar"
echo "=================================================================="

