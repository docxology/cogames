"""Pre-load torch before pytest to prevent SIGABRT on macOS ARM + Python 3.13.

Torch 2.8.0's C extension aborts when first imported inside a pytest process
(due to signal handler conflicts with pytest's faulthandler). Pre-loading
torch before pytest starts avoids the issue since subsequent imports use the
cached sys.modules entry.

Usage:
    python daf/scripts/_preload_torch.py -m pytest [pytest args...]
    python daf/scripts/_preload_torch.py daf/scripts/run_full_suite.py [args...]
"""

import sys

# Pre-load torch while the process is still clean
try:
    import torch  # noqa: F401
except ImportError:
    pass  # torch not installed — GPU tests will be skipped

# Forward execution to the target module/script
if __name__ == "__main__":
    import runpy

    args = sys.argv[1:]
    if not args:
        print("Usage: python _preload_torch.py -m <module> [args...] | <script.py> [args...]")
        sys.exit(1)

    if args[0] == "-m":
        # python _preload_torch.py -m pytest daf/tests/ ...
        sys.argv = args[1:]
        runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)
    else:
        # python _preload_torch.py script.py ...
        sys.argv = args
        runpy.run_path(sys.argv[0], run_name="__main__")
