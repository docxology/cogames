#!/bin/bash
# Run comprehensive test suite: core CoGames tests + DAF sidecar tests
# Delegates to the comprehensive test runner in daf/tests/
#
# Usage: ./daf/run_tests.sh [OPTIONS]
#
# For all options, see: ./daf/tests/run_daf_tests.sh --help

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Run full test suite
exec ./daf/tests/run_daf_tests.sh "$@"
