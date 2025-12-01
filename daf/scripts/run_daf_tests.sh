#!/bin/bash
# Run DAF tests
# Delegates to the comprehensive test runner in daf/tests/
#
# Usage: ./daf/scripts/run_daf_tests.sh [OPTIONS]
# 
# For all options, see: ./daf/tests/run_daf_tests.sh --help

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Run only DAF tests by default when using this script
exec ./daf/tests/run_daf_tests.sh --only-daf "$@"
