"""DAF training and deployment: distributed training, deployment, auth.

Submodules:
    distributed_training  Multi-node training orchestration
    deployment            Policy packaging and deployment
    auth_integration      CoGames authentication
"""

from __future__ import annotations

__all__ = [
    "daf_launch_distributed_training",
    "daf_create_training_cluster",
    "daf_aggregate_training_stats",
    "daf_get_training_status",
    "DistributedTrainingResult",
    "daf_package_policy",
    "daf_validate_deployment",
    "daf_deploy_policy",
    "daf_monitor_deployment",
    "daf_rollback_deployment",
    "daf_submit_policy",
    "DeploymentResult",
    "daf_login",
    "daf_check_auth_status",
    "daf_get_auth_token",
]

_MODULE_MAP = {
    "daf_launch_distributed_training": "daf.src.train.distributed_training",
    "daf_create_training_cluster": "daf.src.train.distributed_training",
    "daf_aggregate_training_stats": "daf.src.train.distributed_training",
    "daf_get_training_status": "daf.src.train.distributed_training",
    "DistributedTrainingResult": "daf.src.train.distributed_training",
    "daf_package_policy": "daf.src.train.deployment",
    "daf_validate_deployment": "daf.src.train.deployment",
    "daf_deploy_policy": "daf.src.train.deployment",
    "daf_monitor_deployment": "daf.src.train.deployment",
    "daf_rollback_deployment": "daf.src.train.deployment",
    "daf_submit_policy": "daf.src.train.deployment",
    "DeploymentResult": "daf.src.train.deployment",
    "daf_login": "daf.src.train.auth_integration",
    "daf_check_auth_status": "daf.src.train.auth_integration",
    "daf_get_auth_token": "daf.src.train.auth_integration",
}


def __getattr__(name: str):
    """Lazy import of train submodule attributes."""
    if name in _MODULE_MAP:
        import importlib
        mod = importlib.import_module(_MODULE_MAP[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'daf.src.train' has no attribute '{name}'")
