
import json
import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from daf.src.viz.gif_generator import (
    generate_gif_from_replay,
    get_mettascope_url,
    load_replay
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_viz")

def main():
    print("----------------------------------------------------------------")
    print("   DAF Visualization Verification: Triple Check")
    print("----------------------------------------------------------------")

    # 1. Create dummy replay
    replay_data = {
        "grid_width": 10,
        "grid_height": 10,
        "num_steps": 5,
        "map_size": [10, 10],
        "objects": [
            {
                "id": 1,
                "type_name": "agent_policy_1",
                "location": [[0, [2, 2]], [1, [3, 3]], [2, [4, 4]], [3, [5, 5]], [4, [6, 6]]],
            },
            {
                "id": 2, 
                "type_name": "wall",
                "location": [5, 5]
            },
             {
                "id": 3, 
                "type_name": "resource_energy",
                "location": [8, 8]
            }
        ],
    }
    
    tmp_dir = Path("daf_output/verification")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    replay_path = tmp_dir / "test_replay.json"
    with open(replay_path, "w") as f:
        json.dump(replay_data, f)
        
    print(f"[OK] Created dummy replay: {replay_path}")
    
    # 2. Verify Load
    try:
        data = load_replay(replay_path)
        print(f"[OK] Loaded replay successfully. Steps: {data.get('num_steps')}")
    except Exception as e:
        print(f"[FAIL] Failed to load replay: {e}")
        sys.exit(1)

    # 3. Generate GIF
    gif_path = tmp_dir / "test_viz.gif"
    try:
        output = generate_gif_from_replay(
            replay_path, 
            gif_path, 
            fps=2, 
            cell_size=32
        )
        print(f"[OK] Generated GIF: {output}")
        if output.exists() and output.stat().st_size > 0:
            print(f"     Size: {output.stat().st_size} bytes")
        else:
            print("[FAIL] GIF file is empty or missing")
            sys.exit(1)
    except Exception as e:
        print(f"[FAIL] Animation generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 4. Verify MettaScope URL
    url = get_mettascope_url("https://example.com/replays/test_replay.json")
    print(f"[OK] MettaScope URL check: {url}")
    
    print("\n[SUCCESS] Triple check complete. Visualization pipeline is operational.")

if __name__ == "__main__":
    main()
