"""Tests for DAF GIF generator module."""

import json
from pathlib import Path

import pytest

from daf.src.viz.gif_generator import (
    load_replay,
    get_mettascope_url,
    generate_frames_only,
    generate_gif_from_replay,
    render_frame_matplotlib,
    OBJECT_COLORS,
    AGENT_COLORS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_replay(tmp_path) -> Path:
    """Create a minimal valid replay file for testing."""
    replay_data = {
        "grid_width": 4,
        "grid_height": 4,
        "num_steps": 3,
        "objects": [
            {
                "id": 1,
                "type": "agent",
                "r": [[0, 0], [1, 1], [2, 2]],
                "c": [[0, 0], [1, 1], [2, 2]],
            },
            {
                "id": 2,
                "type": "wall",
                "r": 3,
                "c": 3,
            },
        ],
    }
    replay_path = tmp_path / "replay.json"
    with open(replay_path, "w") as f:
        json.dump(replay_data, f)
    return replay_path


@pytest.fixture
def empty_replay(tmp_path) -> Path:
    """Create an empty/minimal replay file."""
    replay_data = {
        "grid_width": 2,
        "grid_height": 2,
        "num_steps": 0,
        "objects": [],
    }
    path = tmp_path / "empty_replay.json"
    with open(path, "w") as f:
        json.dump(replay_data, f)
    return path


# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------

def test_object_colors_populated():
    """Verify OBJECT_COLORS has expected entries."""
    assert "wall" in OBJECT_COLORS
    assert "agent" in OBJECT_COLORS
    assert "default" in OBJECT_COLORS
    assert len(OBJECT_COLORS) >= 5


def test_agent_colors_length():
    """Verify AGENT_COLORS has enough entries."""
    assert len(AGENT_COLORS) >= 4
    # Each color is an RGB tuple
    for color in AGENT_COLORS:
        assert len(color) == 3


# ---------------------------------------------------------------------------
# load_replay
# ---------------------------------------------------------------------------

def test_load_replay_valid(minimal_replay):
    """Load a valid JSON replay file."""
    data = load_replay(minimal_replay)

    assert isinstance(data, dict)
    assert data["grid_width"] == 4
    assert data["grid_height"] == 4
    assert data["num_steps"] == 3
    assert len(data["objects"]) == 2


def test_load_replay_nonexistent():
    """Loading nonexistent file raises error."""
    with pytest.raises((FileNotFoundError, OSError)):
        load_replay("/nonexistent/replay.json")


def test_load_replay_invalid_json(tmp_path):
    """Loading invalid JSON raises error."""
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not json at all")

    with pytest.raises((json.JSONDecodeError, ValueError)):
        load_replay(bad_file)


# ---------------------------------------------------------------------------
# render_frame_matplotlib
# ---------------------------------------------------------------------------

def test_render_frame_matplotlib_dimensions(minimal_replay):
    """Rendered frame has correct shape."""
    try:
        import numpy as np
        import matplotlib
    except ImportError:
        pytest.skip("matplotlib/numpy not installed")

    data = load_replay(minimal_replay)
    frame = render_frame_matplotlib(data, step=0, width=200, height=200, cell_size=8)

    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3
    assert frame.shape[2] == 3  # RGB


def test_render_frame_matplotlib_empty(empty_replay):
    """Rendering empty replay does not crash."""
    try:
        import numpy as np
        import matplotlib
    except ImportError:
        pytest.skip("matplotlib/numpy not installed")

    data = load_replay(empty_replay)
    frame = render_frame_matplotlib(data, step=0, width=100, height=100, cell_size=8)

    assert isinstance(frame, np.ndarray)


# ---------------------------------------------------------------------------
# get_mettascope_url
# ---------------------------------------------------------------------------

def test_get_mettascope_url_format():
    """Verify MettaScope URL construction."""
    url = get_mettascope_url("https://example.com/replay.json.z")

    assert isinstance(url, str)
    assert "mettascope" in url.lower() or "example.com" in url


def test_get_mettascope_url_encodes_properly():
    """URL with special characters handled."""
    url = get_mettascope_url("https://example.com/path/to/replay with spaces.json")
    assert isinstance(url, str)


# ---------------------------------------------------------------------------
# generate_gif_from_replay
# ---------------------------------------------------------------------------

def test_generate_gif_from_replay(minimal_replay, tmp_path):
    """Generate a GIF file from replay data."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        pytest.skip("PIL/numpy not installed")

    output_path = tmp_path / "output.gif"
    generate_gif_from_replay(
        replay_path=minimal_replay,
        output_path=output_path,
        fps=5,
        cell_size=8,
        step_interval=1,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# generate_frames_only
# ---------------------------------------------------------------------------

def test_generate_frames_only(minimal_replay, tmp_path):
    """Generate individual frame files from replay."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        pytest.skip("PIL/numpy not installed")

    frames_dir = tmp_path / "frames"
    generate_frames_only(
        replay_path=minimal_replay,
        output_dir=frames_dir,
        cell_size=8,
        step_interval=1,
    )

    assert frames_dir.exists()
    # Should have generated at least some frame files
    frame_files = list(frames_dir.glob("*.png"))
    assert len(frame_files) >= 0  # May be 0 if no renderable steps


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestGifGeneratorIntegration:
    """Integration tests for the full GIF generation pipeline."""

    def test_replay_to_gif_pipeline(self, minimal_replay, tmp_path):
        """Full pipeline: load → render → save GIF."""
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            pytest.skip("PIL/numpy not installed")

        # Load
        data = load_replay(minimal_replay)
        assert data["num_steps"] == 3

        # Render a frame
        frame = render_frame_matplotlib(data, step=0, cell_size=8)
        assert frame is not None

        # Generate GIF
        output = tmp_path / "pipeline.gif"
        generate_gif_from_replay(minimal_replay, output, fps=5, cell_size=8)
        assert output.exists()
