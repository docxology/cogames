# DAF Comparison Module

**DAF sidecar utility: Policy comparison by invoking `cogames.evaluate.evaluate()`.**

## Overview

Head-to-head policy evaluation with statistical significance testing and benchmarking.

## CoGames Integration

**Primary Method**: `cogames.evaluate.evaluate()`

Each policy comparison:
1. Invokes evaluation once with all policies
2. Extracts per-policy, per-mission rewards
3. Performs statistical analysis (t-tests, effect sizes)
4. Generates reports

## Key Functions

- `daf_compare_policies()` - Multi-policy comparison
- `daf_benchmark_suite()` - Standardized benchmarks
- `daf_policy_ablation()` - Ablation studies

## Testing

See `daf/tests/test_comparison.py` for coverage.

## See Also

- `daf/docs/INTEGRATION_MAP.md` - Full integration details
- `daf/docs/modules/` - All modules
