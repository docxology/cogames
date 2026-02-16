#!/bin/bash
# Run full CoGames evaluation suite with comprehensive visualizations
#
# This script executes real cogames evaluations (not unit tests) and generates
# a complete visualization dashboard.
#
# Usage:
#   ./daf/scripts/run_full_suite.sh [OPTIONS]
#
# Options:
#   --quick              Quick mode (3 episodes, minimal sweep)
#   --policies P1 P2...  Policies to evaluate (default: cogames.policy.starter_agent.StarterPolicy)
#   --missions M1 M2...  Missions to run on (default: cogsguard_machina_1.basic)
#   --no-sweep           Skip hyperparameter sweep phase
#   --help               Show this help message
#
# Examples:
#   # Quick evaluation
#   ./daf/scripts/run_full_suite.sh --quick
#
#   # Full evaluation with LSTM
#   ./daf/scripts/run_full_suite.sh --policies lstm cogames.policy.starter_agent.StarterPolicy
#
#   # Multiple missions
#   ./daf/scripts/run_full_suite.sh --missions cogsguard_machina_1.basic assembler_2

set -e

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Check if running inside a virtual environment
if [ -z "$VIRTUAL_ENV" ] && [ -f ".venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Display header
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║          CoGames Full Evaluation Suite                           ║"
echo "║          Real Policy Evaluation + Visualizations                 ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Run the Python script with torch preloaded to avoid macOS ARM + Python 3.13 abort
python3 "$SCRIPT_DIR/_preload_torch.py" "$SCRIPT_DIR/run_full_suite.py" "$@"
exit_code=$?

# Show output location reminder
if [ $exit_code -eq 0 ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Outputs are in: daf_output/full_suite/suite_*/"
    echo ""
    echo "Key files:"
    echo "  • dashboard/dashboard.html     - Interactive dashboard"
    echo "  • comparisons/report.html      - Policy comparison report"
    echo "  • sweeps/*/sweep_progress.png  - Hyperparameter analysis"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi

exit $exit_code

