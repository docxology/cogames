#!/bin/bash
# Run all DAF tests with uv
# IMPORTANT: This script must be run from the project root directory

set -e

echo "=========================================="
echo "DAF Test Suite - Comprehensive Verification"
echo "=========================================="
echo ""

# Ensure we're in the project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Running tests from: $(pwd)"
echo ""

echo "Running DAF Configuration Tests..."
python -m pytest tests/test_config.py -v

echo ""
echo "Running DAF Environment Check Tests..."
python -m pytest tests/test_environment_checks.py -v

echo ""
echo "Running DAF Sweep Tests..."
python -m pytest tests/test_sweeps.py -v

echo ""
echo "Running DAF Orchestrator Tests (VERIFY ORDERING)..."
python -m pytest tests/test_orchestrators.py -v

echo ""
echo "Running DAF Comparison Tests..."
python -m pytest tests/test_comparison.py -v

echo ""
echo "Running DAF Deployment Tests..."
python -m pytest tests/test_deployment.py -v

echo ""
echo "Running DAF Distributed Training Tests..."
python -m pytest tests/test_distributed_training.py -v

echo ""
echo "Running DAF Visualization Tests..."
python -m pytest tests/test_visualization.py -v

echo ""
echo "Running DAF Mission Analysis Tests..."
python -m pytest tests/test_mission_analysis.py -v

echo ""
echo "=========================================="
echo "All DAF Tests Complete"
echo "=========================================="

