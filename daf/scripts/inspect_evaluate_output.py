
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rich.console import Console
from cogames import evaluate
from mettagrid.policy.policy import PolicySpec
from cogames.cli.mission import get_mission

console = Console()

def inspect_evaluate():
    mission_name = "cogsguard_machina_1.basic"
    name, env_cfg, _ = get_mission(mission_name)
    
    policy = PolicySpec(class_path="cogames.policy.starter_agent.StarterPolicy")
    
    console.print(f"Running evaluation for {mission_name}...")
    summaries = evaluate.evaluate(
        console=Console(quiet=True),
        missions=[(name, env_cfg)],
        policy_specs=[policy],
        proportions=[1.0],
        episodes=2,  # Run 2 episodes to check for list vs scalar
        action_timeout_ms=250,
        seed=42
    )
    
    summary = summaries[0]
    console.print(f"\nSummary type: {type(summary)}")
    console.print(f"Attributes: {dir(summary)}")
    
    if hasattr(summary, "policy_summaries"):
        ps = summary.policy_summaries[0]
        console.print(f"\nPolicy Summary Attributes: {dir(ps)}")
        if hasattr(ps, "avg_agent_metrics"):
             console.print(f"avg_agent_metrics type: {type(ps.avg_agent_metrics)}")
        
        # Check for per-episode metrics
        # Common patterns: 'episode_metrics', 'metrics_history', etc.
        for attr in dir(ps):
            if "metric" in attr:
                val = getattr(ps, attr)
                console.print(f"Attribute '{attr}': {type(val)} - {val if not isinstance(val, (list, dict)) or len(val) < 5 else '... (len ' + str(len(val)) + ')'}")

if __name__ == "__main__":
    inspect_evaluate()
