"""
tests/conftest.py
-------------------
Ensures physics/ and plotting/ are importable from test files without
needing the package installed or sys.path hacks inside every test module.

Repo layout (per actual project structure, no src/ wrapper):

    fourier-optics-simulator/
    ├── physics/
    │   ├── fft_engine.py
    │   ├── grid.py
    │   └── masks.py
    ├── plotting/
    │   └── core.py
    └── tests/
        └── conftest.py   <- this file

Test files import as `from masks import make_grid`, `from grid import
Grid1D`, etc. (flat module names, matching how Week 1/2 source files
import from each other internally) -- so both physics/ and plotting/ need
to be on sys.path directly, not just the repo root.

tests/test_api.py additionally imports `backend.main` as a real dotted
package (backend/ has an __init__.py, unlike physics/plotting/), which
needs the REPO ROOT itself on sys.path -- `python -m pytest` (the command
CLAUDE.md documents) adds this automatically via Python's own `-m` flag,
but plain `pytest` does not, and failed here with `ModuleNotFoundError: No
module named 'backend'` until this line was added -- confirmed directly,
not assumed.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "physics"))
sys.path.insert(0, str(REPO_ROOT / "plotting"))