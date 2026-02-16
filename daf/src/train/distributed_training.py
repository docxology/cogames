"""Distributed training orchestration for DAF.

DAF sidecar utility: Wraps `cogames.train.train()` with distributed coordination.

Extends CoGames training with multi-node, multi-GPU support. For single-node training,
directly invokes `cogames.train.train()`. Multi-node coordination is added as a wrapper layer.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from rich.console import Console

# cogames.train imported lazily inside functions to avoid mettagrid dependency at import time

if TYPE_CHECKING:
    import torch

logger = logging.getLogger("daf.distributed_training")


class DistributedTrainingResult:
    """Results from a distributed training run."""

    def __init__(
        self,
        final_checkpoint: Optional[Path] = None,
        training_steps: int = 0,
        final_loss: Optional[float] = None,
        final_reward: Optional[float] = None,
        wall_time_seconds: float = 0.0,
        num_workers: int = 1,
        aggregate_stats: Optional[dict[str, Any]] = None,
    ):
        """Initialize training result.

        Args:
            final_checkpoint: Path to final checkpoint
            training_steps: Total training steps completed
            final_loss: Final loss value
            final_reward: Final average reward
            wall_time_seconds: Total wall time in seconds
            num_workers: Number of workers used
            aggregate_stats: Aggregated statistics from all workers
        """
        self.final_checkpoint = final_checkpoint
        self.training_steps = training_steps
        self.final_loss = final_loss
        self.final_reward = final_reward
        self.wall_time_seconds = wall_time_seconds
        self.num_workers = num_workers
        self.aggregate_stats = aggregate_stats or {}
        self.timestamp = datetime.now()

    def training_rate(self) -> float:
        """Get training steps per second.

        Returns:
            Steps per second
        """
        if self.wall_time_seconds <= 0:
            return 0.0
        return self.training_steps / self.wall_time_seconds


def daf_launch_distributed_training(
    env_cfg: Any,
    policy_class_path: str,
    device: "torch.device",
    initial_weights_path: Optional[str] = None,
    num_steps: int = 1_000_000,
    checkpoints_path: Path = Path("./daf_output/checkpoints"),
    seed: int = 42,
    batch_size: int = 4096,
    minibatch_size: int = 4096,
    num_nodes: int = 1,
    workers_per_node: int = 1,
    backend: str = "torch",
    env_cfg_supplier: Optional[Callable[[], Any]] = None,
    log_outputs: bool = False,
    console: Optional[Console] = None,
) -> DistributedTrainingResult:
    """Launch distributed training across multiple nodes/GPUs.

    Orchestrates multi-node training by wrapping cogames.train.train()
    with distributed coordination logic.

    Args:
        env_cfg: Environment configuration (or None if using supplier)
        policy_class_path: Full class path to policy
        device: Torch device for training
        initial_weights_path: Optional path to initial weights
        num_steps: Total training steps
        checkpoints_path: Where to save checkpoints
        seed: Random seed
        batch_size: Batch size for training
        minibatch_size: Minibatch size for updates
        num_nodes: Number of training nodes
        workers_per_node: Workers per node
        backend: Distributed backend ("torch", "ray", "dask")
        env_cfg_supplier: Optional callable to generate configs
        log_outputs: Whether to log training outputs
        console: Optional Rich console for output

    Returns:
        DistributedTrainingResult with checkpoint and statistics

    Raises:
        ValueError: If configuration is invalid
        RuntimeError: If distributed setup fails
    """
    if console is None:
        console = Console()

    # Validate inputs
    if not policy_class_path:
        raise ValueError("policy_class_path cannot be empty")
    if not env_cfg and not env_cfg_supplier:
        raise ValueError("Either env_cfg or env_cfg_supplier must be provided")
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if num_nodes < 1:
        raise ValueError(f"num_nodes must be >= 1, got {num_nodes}")
    if workers_per_node < 1:
        raise ValueError(f"workers_per_node must be >= 1, got {workers_per_node}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if minibatch_size < 1:
        raise ValueError(f"minibatch_size must be >= 1, got {minibatch_size}")
    
    supported_backends = ["torch", "ray", "dask"]
    if backend not in supported_backends:
        raise ValueError(f"backend must be one of {supported_backends}, got {backend}")
    
    logger.info(f"Distributed training configuration: {num_nodes} nodes, {workers_per_node} workers/node, {backend} backend")

    console.print("[cyan]DAF Distributed Training[/cyan]")
    console.print(f"  Nodes: {num_nodes}")
    console.print(f"  Workers per node: {workers_per_node}")
    console.print(f"  Backend: {backend}")
    console.print(f"  Total training steps: {num_steps:,}")
    console.print()

    if num_nodes == 1 and workers_per_node == 1:
        # Fall back to single-process training
        console.print("[yellow]Single-node training requested, using standard cogames.train.train()[/yellow]\n")

        import time

        start_time = time.time()

        from cogames.train import train
        train(
            env_cfg=env_cfg,
            policy_class_path=policy_class_path,
            device=device,
            initial_weights_path=initial_weights_path,
            num_steps=num_steps,
            checkpoints_path=checkpoints_path,
            seed=seed,
            batch_size=batch_size,
            minibatch_size=minibatch_size,
            env_cfg_supplier=env_cfg_supplier,
            log_outputs=log_outputs,
        )

        elapsed = time.time() - start_time

        elapsed = time.time() - start_time

        # Find final checkpoint
        final_ckpt = None
        try:
            from mettagrid.policy.loader import find_policy_checkpoints
            checkpoints = find_policy_checkpoints(checkpoints_path, "cogames.cogs_vs_clips")
            final_ckpt = checkpoints[-1] if checkpoints else None
        except (ImportError, AttributeError):
            logger.warning("Could not load policy checkpoints (mettagrid not available or incompatible)")

        return DistributedTrainingResult(
            final_checkpoint=final_ckpt,
            training_steps=num_steps,
            wall_time_seconds=elapsed,
            num_workers=1,
        )

    # Multi-node distributed training would go here
    # For now, we provide the architecture but defer implementation
    # to specific backends (Ray, Dask, torch.distributed)

    console.print("[yellow]Multi-node training not yet fully implemented[/yellow]")
    console.print("[yellow]Falling back to single-node training[/yellow]\n")

    import time

    start_time = time.time()

    from cogames.train import train
    train(
        env_cfg=env_cfg,
        policy_class_path=policy_class_path,
        device=device,
        initial_weights_path=initial_weights_path,
        num_steps=num_steps,
        checkpoints_path=checkpoints_path,
        seed=seed,
        batch_size=batch_size,
        minibatch_size=minibatch_size,
        env_cfg_supplier=env_cfg_supplier,
        log_outputs=log_outputs,
    )

    elapsed = time.time() - start_time

    final_ckpt = None
    try:
        from mettagrid.policy.loader import find_policy_checkpoints
        checkpoints = find_policy_checkpoints(checkpoints_path, "cogames.cogs_vs_clips")
        final_ckpt = checkpoints[-1] if checkpoints else None
    except (ImportError, AttributeError):
        logger.warning("Could not load policy checkpoints (mettagrid not available or incompatible)")

    return DistributedTrainingResult(
        final_checkpoint=final_ckpt,
        training_steps=num_steps,
        wall_time_seconds=elapsed,
        num_workers=num_nodes * workers_per_node,
    )


def daf_create_training_cluster(
    num_nodes: int = 1,
    workers_per_node: int = 1,
    backend: str = "torch",
    console: Optional[Console] = None,
) -> dict[str, Any]:
    """Set up distributed compute resources.

    Initializes distributed backend and allocates workers across nodes.

    Args:
        num_nodes: Number of training nodes
        workers_per_node: Workers per node
        backend: Distributed backend
        console: Optional Rich console for output

    Returns:
        Cluster configuration dictionary

    Raises:
        RuntimeError: If cluster setup fails
    """
    if console is None:
        console = Console()

    if num_nodes == 1 and workers_per_node == 1:
        return {"backend": "single", "num_nodes": 1, "workers_per_node": 1}

    console.print(f"[cyan]Setting up {backend} cluster[/cyan]")
    console.print(f"  Nodes: {num_nodes}")
    console.print(f"  Workers per node: {workers_per_node}")

    if backend == "ray":
        try:
            import ray

            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            console.print("[green]✓ Ray cluster initialized[/green]")
        except ImportError:
            raise RuntimeError("Ray not installed. Install with: pip install ray") from None
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Ray: {e}") from e

    elif backend == "dask":
        try:
            import dask.distributed

            client = dask.distributed.Client(processes=False, n_workers=workers_per_node * num_nodes)
            console.print(f"[green]✓ Dask cluster initialized[/green]")
        except ImportError:
            raise RuntimeError("Dask not installed. Install with: pip install dask[distributed]") from None
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Dask: {e}") from e

    elif backend == "torch":
        console.print("[yellow]torch.distributed backend requires external torch launch utilities[/yellow]")

    return {
        "backend": backend,
        "num_nodes": num_nodes,
        "workers_per_node": workers_per_node,
    }


def daf_aggregate_training_stats(
    worker_stats: list[dict[str, Any]],
    aggregation_method: str = "mean",
) -> dict[str, Any]:
    """Collect and aggregate metrics across workers.

    Args:
        worker_stats: List of statistics dictionaries from each worker
        aggregation_method: How to aggregate ("mean", "sum", "max", "min")

    Returns:
        Aggregated statistics dictionary
    """
    if not worker_stats:
        return {}

    aggregated = {}

    # Get all metric names from first worker
    first_worker = worker_stats[0]
    for metric_name in first_worker:
        values = [w.get(metric_name, 0.0) for w in worker_stats if metric_name in w]

        if not values:
            continue

        try:
            values_float = [float(v) for v in values]
        except (ValueError, TypeError):
            # Skip non-numeric metrics
            continue

        if aggregation_method == "mean":
            aggregated[metric_name] = sum(values_float) / len(values_float)
        elif aggregation_method == "sum":
            aggregated[metric_name] = sum(values_float)
        elif aggregation_method == "max":
            aggregated[metric_name] = max(values_float)
        elif aggregation_method == "min":
            aggregated[metric_name] = min(values_float)

    return aggregated


def daf_get_training_status(
    checkpoints_path: Path,
    policy_class_path: str = "cogames.cogs_vs_clips",
) -> dict[str, Any]:
    """Get current training status and latest checkpoint info.

    Args:
        checkpoints_path: Directory containing checkpoints
        policy_class_path: Policy class for checkpoint discovery

    Returns:
        Status dictionary with checkpoint info and stats
    """
    try:
        from mettagrid.policy.loader import find_policy_checkpoints
    except ImportError:
        logging.getLogger("daf.distributed_training").warning(
            "mettagrid not available — cannot discover checkpoints"
        )
        return {
            "num_checkpoints": 0,
            "latest_checkpoint": None,
            "checkpoint_times": [],
        }

    checkpoints = find_policy_checkpoints(checkpoints_path, policy_class_path)

    status = {
        "num_checkpoints": len(checkpoints),
        "latest_checkpoint": None,
        "checkpoint_times": [],
    }

    if checkpoints:
        latest = checkpoints[-1]
        status["latest_checkpoint"] = str(latest)

        # Get checkpoint file info if available
        try:
            stat_info = latest.stat()
            status["latest_checkpoint_size_bytes"] = stat_info.st_size
            status["latest_checkpoint_mtime"] = stat_info.st_mtime
        except Exception:
            pass

    return status

