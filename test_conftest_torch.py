import torch  # noqa: F401

def pytest_configure(config):
    """Pre-import torch during pytest configuration."""
    pass

def test_torch_available():
    assert torch.__version__
    print(f"torch {torch.__version__} OK")
