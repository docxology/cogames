# DAF API Reference

Complete reference for all `daf_*` functions and classes in the Distributed Agent Framework.

**Note**: DAF is a sidecar utility that extends CoGames. All DAF functions invoke underlying CoGames methods. See integration points in ARCHITECTURE.md.

## Configuration Module (`daf.config`)

### Classes

#### `DAFConfig`

Global DAF configuration settings.

```python
from daf.config import DAFConfig

config = DAFConfig(
    output_dir="./daf_output",
    max_parallel_jobs=8,
    enable_gpu=True,
    verbose=False
)
```

**Attributes:**
- `output_dir`: Output directory for results (default: `./daf_output`)
- `checkpoint_dir`: Directory for checkpoints (default: `./daf_output/checkpoints`)
- `max_checkpoints_to_keep`: Max checkpoints per run (default: 5)
- `max_parallel_jobs`: Parallel jobs limit (default: 4)
- `enable_gpu`: Use GPU if available (default: True)
- `gpu_memory_fraction`: GPU memory usage fraction (default: 0.8)
- `verbose`: Enable verbose logging (default: False)
- `log_to_file`: Log to file (default: True)
- `monitor_system_stats`: Monitor CPU/GPU/memory (default: True)
- `track_experiments`: Enable experiment tracking (default: True)

#### `DAFSweepConfig`

Hyperparameter sweep configuration.

```python
from daf.config import DAFSweepConfig

config = DAFSweepConfig.from_yaml("sweep.yaml")
```

**Methods:**
- `from_yaml(path)`: Load from YAML file
- `from_json(path)`: Load from JSON file
- `to_yaml(path)`: Save to YAML file

**Attributes:**
- `name`: Sweep name
- `missions`: List of mission names/paths
- `policy_class_path`: Policy class path
- `search_space`: Parameter search space dict
- `strategy`: "grid", "random", or "bayesian"
- `num_trials`: Number of trials (for random/Bayesian)
- `episodes_per_trial`: Episodes per configuration
- `objective_metric`: Metric to optimize
- `optimize_direction`: "maximize" or "minimize"

#### `DAFDeploymentConfig`

Policy deployment configuration.

**Attributes:**
- `policy_name`: Human-readable name
- `policy_class_path`: Policy class path
- `weights_path`: Path to weights/checkpoint
- `deployment_endpoint`: Target deployment URL
- `authentication_server`: Auth server URL
- `validation_missions`: Missions for validation
- `version`: Semantic version string
- `enable_rollback`: Allow rollback (default: True)

#### `DAFComparisonConfig`

Policy comparison configuration.

**Attributes:**
- `name`: Comparison name
- `policies`: List of policy class paths
- `missions`: List of missions
- `episodes_per_mission`: Episodes per mission
- `generate_html_report`: Generate HTML (default: True)
- `save_raw_results`: Save JSON results (default: True)

#### `DAFPipelineConfig`

Workflow orchestration configuration.

**Attributes:**
- `name`: Pipeline name
- `stages`: Ordered list of stages
- `stop_on_failure`: Stop on error (default: True)
- `parallel_stages`: Run stages in parallel (default: False)

### Functions

#### `daf_load_config(config_path) → DAFConfig`

Load global DAF configuration from YAML/JSON file.

```python
from daf.config import daf_load_config

config = daf_load_config("daf_config.yaml")
```

## Environment Checks Module (`daf.environment_checks`)

### Classes

#### `EnvironmentCheckResult`

Container for environment check results.

**Methods:**
- `add_check(name, passed, warning=None, info=None)`: Record check result
- `add_error(name, error_msg)`: Record check error
- `is_healthy()`: Return True if all checks passed
- `summary()`: Get human-readable summary

### Functions

#### `daf_check_gpu_availability(console=None) → EnvironmentCheckResult`

Platform-aware GPU verification. Checks MPS on macOS (Apple Silicon), CUDA on Linux/Windows.

```python
from daf.environment_checks import daf_check_gpu_availability

result = daf_check_gpu_availability()
if result.is_healthy():
    print("GPU available:", result.info.get("device"))
    # On macOS: "MPS (Apple Silicon)"
    # On Linux/Windows: "CUDA" with device count
```

**Platform Detection:**
- **macOS**: Checks MPS (Metal Performance Shaders) for Apple Silicon GPUs
- **Linux/Windows**: Checks CUDA for NVIDIA GPUs

**Returns:**
- `EnvironmentCheckResult` with GPU backend info in `result.info["gpu_backend"]`

#### `daf_check_cuda_availability(console=None) → EnvironmentCheckResult`

Legacy function for CUDA-only check. **Deprecated**: Use `daf_check_gpu_availability()` instead.

Now internally calls `daf_check_gpu_availability()` for platform-aware detection.

```python
from daf.environment_checks import daf_check_cuda_availability

result = daf_check_cuda_availability()
if not result.is_healthy():
    print("GPU issues detected")
```

#### `daf_check_disk_space(checkpoint_dir, min_available_gb=10.0) → EnvironmentCheckResult`

Check sufficient disk space for checkpoints.

#### `daf_check_dependencies() → EnvironmentCheckResult`

Validate all required packages are installed.

#### `daf_check_mission_configs(missions) → EnvironmentCheckResult`

Validate mission configurations can be loaded.

#### `daf_check_environment(...) → EnvironmentCheckResult`

Comprehensive environment check before training/sweeps.

```python
from daf.environment_checks import daf_check_environment

result = daf_check_environment(
    missions=["training_facility_1"],
    check_cuda=True,  # Actually checks GPU (MPS or CUDA based on platform)
    check_disk=True
)

# Check results
print(result.summary())  # "Environment Check: HEALTHY (16/16 checks passed)"
```

**Checks Performed:**
- GPU availability (MPS on macOS, CUDA on Linux/Windows)
- Disk space for checkpoints
- Required package dependencies
- Mission configuration loading

#### `daf_get_recommended_device() → torch.device`

Get recommended device for training. Returns MPS on Apple Silicon, CUDA on NVIDIA systems, CPU otherwise.

```python
from daf.environment_checks import daf_get_recommended_device

device = daf_get_recommended_device()
# Returns: torch.device("mps") on Apple Silicon
#          torch.device("cuda") on NVIDIA
#          torch.device("cpu") otherwise
```

**Integration**: Wraps `cogames.device.resolve_training_device()` with DAF-specific environment checks and informative console output.

## Distributed Training Module (`daf.distributed_training`)

### Classes

#### `DistributedTrainingResult`

Results from a distributed training run.

**Attributes:**
- `final_checkpoint`: Path to final checkpoint
- `training_steps`: Total steps completed
- `wall_time_seconds`: Total wall time
- `num_workers`: Number of workers used

**Methods:**
- `training_rate()`: Get steps per second

### Functions

#### `daf_launch_distributed_training(...) → DistributedTrainingResult`

Launch distributed training across nodes/GPUs. **Wraps `cogames.train.train()`** with distributed coordination.

```python
from daf.distributed_training import daf_launch_distributed_training

result = daf_launch_distributed_training(
    env_cfg=config,
    policy_class_path="cogames.policy.lstm.LSTMPolicy",
    device=device,
    num_steps=1_000_000
)
```

**Integration**: Invokes `cogames.train.train()` internally for single-node training. Multi-node coordination is added as a wrapper layer.

#### `daf_create_training_cluster(num_nodes=1, workers_per_node=1, backend="torch")`

Set up distributed compute resources.

#### `daf_aggregate_training_stats(worker_stats, aggregation_method="mean")`

Collect and aggregate metrics across workers.

#### `daf_get_training_status(checkpoints_path)`

Get current training status and latest checkpoint info.

## Sweeps Module (`daf.sweeps`)

### Classes

#### `SweepTrialResult`

Result from a single sweep trial.

**Attributes:**
- `trial_id`: Trial identifier
- `hyperparameters`: Parameter dict
- `primary_metric`: Objective metric value
- `all_metrics`: All recorded metrics
- `success`: Whether trial succeeded

#### `SweepResult`

Results from complete hyperparameter sweep.

**Methods:**
- `add_trial(trial)`: Add trial result
- `get_best_trial()`: Get best performing trial
- `get_worst_trial()`: Get worst performing trial
- `top_trials(n=5)`: Get top N trials
- `save_json(path)`: Save results to JSON
- `print_summary(console)`: Print summary to console
- `finalize()`: Mark sweep complete

### Functions

#### `daf_grid_search(search_space) → list[dict]`

Generate grid search configurations.

```python
from daf.sweeps import daf_grid_search

search_space = {
    "lr": [0.001, 0.01],
    "batch_size": [32, 64]
}
configs = daf_grid_search(search_space)
```

#### `daf_random_search(search_space, num_samples, seed=42) → list[dict]`

Generate random search configurations.

#### `daf_launch_sweep(sweep_config) → SweepResult`

Launch a hyperparameter sweep.

```python
from daf.sweeps import daf_launch_sweep
from daf.config import DAFSweepConfig

config = DAFSweepConfig.from_yaml("sweep.yaml")
results = daf_launch_sweep(config)
```

#### `daf_sweep_best_config(sweep_result) → dict`

Extract best configuration from sweep results.

#### `daf_sweep_status(results_path) → dict`

Get sweep status from results file.

## Comparison Module (`daf.comparison`)

### Classes

#### `PolicyComparisonResult`

Results from comparing two policies.

**Attributes:**
- `policy_a_name`: First policy name
- `policy_b_name`: Second policy name
- `avg_reward_a`: Average reward/score for policy A
- `avg_reward_b`: Average reward/score for policy B
- `p_value`: Statistical significance p-value
- `is_significant`: Whether difference is significant
- `effect_size`: Cohen's d effect size

**Methods:**
- `summary_string()`: Get human-readable summary

#### `ComparisonReport`

Complete report from multi-policy comparison.

**Methods:**
- `add_policy_results(policy_name, mission_rewards)`: Add policy results
- `compute_pairwise_comparisons(significance_level=0.05)`: Run statistical tests
- `print_summary(console)`: Print to console
- `save_json(path)`: Save to JSON file (includes `detailed_metrics` if available)

**Attributes:**
- `policy_detailed_metrics`: Dict of policy → mission → agent metrics (resources, actions, etc.)

**Properties:**
- `summary_statistics`: Dict of policy statistics

### Functions

#### `daf_compare_policies(policies, missions, episodes_per_mission=5, use_performance_score=True) → ComparisonReport`

Compare multiple policies on given missions. **Uses `cogames.evaluate.evaluate()`** internally for evaluation.

```python
from daf.comparison import daf_compare_policies
from mettagrid.policy.policy import PolicySpec

policies = [
    PolicySpec(class_path="lstm"),
    PolicySpec(class_path="baseline")
]
missions = [("training_facility_1", env_cfg)]

report = daf_compare_policies(policies, missions)
print(report.summary_statistics)
# {'baseline': {'avg_reward': 6168.60, 'std_dev': 0.0}, 'random': {'avg_reward': 1206.42, 'std_dev': 0.0}}

# Access detailed metrics
print(report.policy_detailed_metrics['baseline']['training_facility_1'])
# {'carbon.gained': 50.0, 'energy.gained': 5525.0, 'action.move.success': 2607.5, ...}
```

**Parameters:**
- `policies`: List of PolicySpec objects to compare
- `missions`: List of (mission_name, env_config) tuples
- `episodes_per_mission`: Episodes to run per mission (default: 5)
- `use_performance_score`: If True, compute composite performance score from agent metrics when raw rewards are 0 (default: True)

**Performance Score Computation:**
When environments return zero rewards, DAF computes a meaningful performance score from agent metrics:
- Resources gained (carbon, silicon, oxygen, germanium, energy)
- Inventory diversity (weighted higher)
- Successful actions (movement)
- Penalties for failures and being stuck

**Integration**: Invokes `cogames.evaluate.evaluate()` for policy evaluation, then adds statistical analysis and reporting.

#### `daf_policy_ablation(...) → ComparisonReport`

Run ablation study on policy components.

#### `daf_benchmark_suite(policy_specs, benchmark_name="standard") → ComparisonReport`

Run standardized benchmark suite.

## Visualization Module (`daf.visualization`)

### Functions

#### `daf_plot_training_curves(checkpoint_dir, output_dir="training_plots")`

Visualize training progress from checkpoint directory.

#### `daf_plot_policy_comparison(comparison_report, output_dir="comparison_plots")`

Generate comparison plots from ComparisonReport.

```python
from daf.visualization import daf_plot_policy_comparison

daf_plot_policy_comparison(comparison_report)
```

**Generated Files:**
- `policy_rewards_comparison.png` - Bar chart of overall performance
- `performance_by_mission.png` - Grouped bars per mission
- `reward_distributions.png` - Violin plots of score distributions

#### `daf_plot_detailed_metrics_comparison(comparison_report, output_dir="comparison_plots")`

Generate detailed agent metric comparison plots from ComparisonReport with `policy_detailed_metrics`.

```python
from daf.visualization import daf_plot_detailed_metrics_comparison

# Requires comparison_report to have policy_detailed_metrics populated
daf_plot_detailed_metrics_comparison(comparison_report, output_dir="comparisons/")
```

**Generated Files:**
- `metrics_resources_gained.png` - Bar chart: carbon, silicon, oxygen, germanium gained
- `metrics_resources_held.png` - Bar chart: current resource amounts
- `metrics_energy.png` - Bar chart: energy amount, gained, lost
- `metrics_actions.png` - Bar chart: move success/fail, noop, change_vibe
- `metrics_inventory.png` - Bar chart: inventory diversity metrics
- `metrics_radar.png` - Radar chart: key metrics comparison
- `action_distribution.png` - Pie charts: action type distribution per policy

**Metric Categories Plotted:**
- **Resources Gained**: carbon.gained, silicon.gained, oxygen.gained, germanium.gained
- **Resources Held**: carbon.amount, silicon.amount, oxygen.amount, germanium.amount
- **Energy**: energy.amount, energy.gained, energy.lost
- **Actions**: action.move.success, action.move.failed, action.noop.success, action.change_vibe.success
- **Inventory**: inventory.diversity, inventory.diversity.ge.2, etc.

#### `daf_plot_sweep_results(sweep_result, output_dir="sweep_plots")`

Generate plots from sweep results.

**Generated Files:**
- `sweep_progress.png` - Trial performance over time
- `best_configuration.png` - Best hyperparameter configuration
- `hyperparameter_importance.png` - Correlation with performance

#### `daf_plot_sweep_heatmap(sweep_result, hp_x, hp_y, output_path)`

Generate performance heatmap across two hyperparameters.

#### `daf_plot_sweep_parallel_coordinates(sweep_result, output_path)`

Generate parallel coordinates plot for hyperparameter exploration.

#### `daf_export_comparison_html(comparison_report, output_path="report.html")`

Export comparison report as interactive HTML. Includes detailed metrics tables when `policy_detailed_metrics` is available.

**HTML Sections:**
- Summary statistics with rankings
- Pairwise statistical comparisons
- Detailed agent metrics tables (Resources, Energy, Actions, Inventory, Status)

#### `daf_generate_leaderboard(comparison_report, output_path=None) → str`

Generate policy leaderboard table in Markdown format.

#### `daf_generate_summary_dashboard(sweep_result=None, comparison_report=None, output_dir="dashboard")`

Generate comprehensive dashboard with multiple visualizations.

**Generated Files:**
- `dashboard.html` - Interactive summary dashboard
- All sweep/comparison plots in subdirectories

## Deployment Module (`daf.deployment`)

### Classes

#### `DeploymentResult`

Result from a deployment operation.

**Attributes:**
- `policy_name`: Policy name
- `version`: Version string
- `status`: "success", "validation_failed", or "deployment_failed"
- `message`: Status message
- `deployment_id`: Deployment identifier

### Functions

#### `daf_package_policy(policy_class_path, weights_path=None, additional_files=None) → DeploymentResult`

Package policy with dependencies for deployment.

```python
from daf.deployment import daf_package_policy

result = daf_package_policy(
    policy_class_path="cogames.policy.lstm.LSTMPolicy",
    weights_path="checkpoints/policy.pt"
)
```

#### `daf_validate_deployment(policy_class_path, weights_path=None, validation_missions=None) → DeploymentResult`

Validate policy in isolated environment.

#### `daf_deploy_policy(policy_name, package_path, deployment_endpoint, auth_token=None) → DeploymentResult`

Deploy policy to production endpoint.

#### `daf_monitor_deployment(deployment_id, endpoint, auth_token=None) → dict`

Monitor deployed policy performance.

#### `daf_rollback_deployment(deployment_id, endpoint, previous_version) → DeploymentResult`

Rollback deployment to previous version.

## Orchestrators Module (`daf.orchestrators`)

### Classes

#### `PipelineResult`

Result from orchestrated pipeline execution.

**Attributes:**
- `pipeline_name`: Pipeline name
- `status`: "success", "failed", or "partial"
- `stages_completed`: List of completed stages
- `stages_failed`: List of failed stages
- `errors`: List of error messages
- `outputs`: Dict of stage outputs
- `total_time_seconds`: Total execution time

**Methods:**
- `is_success()`: Return True if all stages completed

### Functions

#### `daf_run_training_pipeline(...) → PipelineResult`

Complete training workflow orchestrator.

```python
from daf.orchestrators import daf_run_training_pipeline

result = daf_run_training_pipeline(
    policy_class_path="cogames.policy.lstm.LSTMPolicy",
    mission_names=["training_facility_1"],
    num_training_steps=1_000_000
)
```

#### `daf_run_sweep_pipeline(sweep_config) → PipelineResult`

Complete sweep workflow orchestrator.

#### `daf_run_comparison_pipeline(policy_specs, missions, episodes_per_mission=5) → PipelineResult`

Complete policy comparison workflow orchestrator.

#### `daf_run_benchmark_pipeline(policy_specs, benchmark_name="standard") → PipelineResult`

Standardized benchmark orchestrator.

## Error Handling

All DAF functions raise standard Python exceptions:

- `FileNotFoundError`: Configuration or file not found
- `ValueError`: Invalid configuration or input
- `RuntimeError`: Execution errors (e.g., training failed)
- `ImportError`: Optional dependencies not available

## Logging

Enable verbose logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("daf")
```

DAF uses loggers in these modules:
- `daf.environment_checks`
- `daf.distributed_training`
- `daf.sweeps`
- `daf.comparison`
- `daf.visualization`
- `daf.deployment`
- `daf.orchestrators`

