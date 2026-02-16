"""Pre-load torch before pytest starts to avoid SIGABRT on macOS ARM + Python 3.13.

Usage: python _torch_preload.py -m pytest [args...]
"""
import sys
try:
    import torch  # noqa: F401
except ImportError:
    pass

# Re-invoke the original command
if __name__ == "__main__":
    # Remove this script from argv and run the rest
    import runpy
    sys.argv = sys.argv[1:]
    if sys.argv[0] == "-m":
        # python -m pytest -> run pytest module
        sys.argv = sys.argv[1:]
        runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)
