
import logging
import sys
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from daf.src.eval.sweeps import daf_launch_sweep, daf_sweep_best_config
from daf.src.viz.gif_generator import generate_gif_from_simulation
from mettagrid.policy.policy import PolicySpec

# Mock config class since I don't want to rely on Pydantic loading from file
class MockSweepConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("demo_sweep_viz")

def main():
    print("----------------------------------------------------------------")
    print("   DAF Sweep & Visualization Demo")
    print("----------------------------------------------------------------")

    # 1. Define Sweep Config
    # We will sweep over "test_param" just to simulate a sweep
    # In reality, this param might not affect StarterPolicy, but it proves the machinery works
    sweep_config = MockSweepConfig(
        name="demo_sweep",
        policy_class_path="cogames.policy.starter_agent.StarterPolicy",
        missions=["cogsguard_machina_1.basic"],
        episodes_per_trial=1,
        num_trials=2,
        search_space={"test_param": [1, 2]},  # Simple grid
        strategy="grid",
        objective_metric="avg_reward",
        optimize_direction="maximize",
        seed=42
    )

    print("[1/3] Launching Sweep...")
    try:
        sweep_result = daf_launch_sweep(sweep_config)
    except Exception as e:
        print(f"[FAIL] Sweep failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 2. Get Best Config
    best_config = daf_sweep_best_config(sweep_result)
    print(f"[2/3] Best Config Found: {best_config}")

    if not best_config:
        print("[FAIL] No best config found")
        sys.exit(1)

    # 3. Generate GIF for Best Config
    print("[3/3] Generating GIF for Best Policy...")
    
    output_path = Path("daf_output/demo_sweep/best_policy.gif")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Create PolicySpec (in real usage, we'd apply best_config params to policy)
        policy_spec = PolicySpec(class_path=sweep_config.policy_class_path)
        
        # We need the env_cfg for the mission
        from cogames.cli.mission import get_mission
        _, env_cfg, _ = get_mission("cogsguard_machina_1.basic")

        gif_path = generate_gif_from_simulation(
            env_cfg=env_cfg,
            policy_spec=policy_spec,
            output_path=output_path,
            fps=10,
            max_steps=200
        )
        
        print(f"[SUCCESS] GIF generated at: {gif_path}")
        if gif_path.exists() and gif_path.stat().st_size > 0:
             print(f"          Size: {gif_path.stat().st_size} bytes")
        else:
             print("[FAIL] GIF is empty/missing")
             sys.exit(1)

    except Exception as e:
        print(f"[FAIL] Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n[VERIFIED] Ability to sweep and generate GIFs confirmed.")

if __name__ == "__main__":
    main()
