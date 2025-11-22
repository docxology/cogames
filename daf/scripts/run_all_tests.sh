#!/bin/bash
# Run full test suite: top-level cogames tests + DAF tests
# Can be run from anywhere in the project tree

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Delegate to the comprehensive DAF test script
exec ./daf/tests/run_daf_tests.sh

