# DAF Distributed Training Module

**DAF sidecar utility: Training orchestration by wrapping `cogames.train.train()`.**

## Overview

Single-node and multi-node training coordination.

## CoGames Integration

**Primary Method**: `cogames.train.train()`

- Single-node: Directly invokes `cogames.train.train()`
- Multi-node: Wrapper layer handles coordination

## Key Functions

- `daf_launch_distributed_training(env_cfg, policy_class_path, device, num_steps)` - Training launcher
- `daf_create_training_cluster()` - Cluster setup
- `daf_get_training_status()` - Status tracking

## Testing

See `daf/tests/test_distributed_training.py` for coverage.

## See Also

- `daf/docs/INTEGRATION_MAP.md` - Full integration details
