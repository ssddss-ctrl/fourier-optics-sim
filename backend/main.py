"""
backend/main.py
------------------
FastAPI app for the Fourier Optics Lithography Simulator, replacing the
retired Streamlit app (app/main_streamlit_archived.py) as the project's
live UI backend. The React frontend (frontend/) is the new consumer of
this API; physics/ itself is untouched by the migration.

Run with (from the repo root):
    uvicorn backend.main:app --reload

IMPORT MECHANISM -- CHECKED, NOT ASSUMED
------------------------------------------------------------------------
physics/ modules import each other FLATLY (e.g. physics/grid.py does
`from masks import make_grid`, not `from physics.masks import ...` --
confirmed directly by reading physics/grid.py's own import lines). That
means `from physics.grid import Grid1D` from here would actually import
physics/grid.py successfully at the module-object level, but executing
its own top-level `from masks import make_grid` statement would then
immediately raise `ModuleNotFoundError: No module named 'masks'`, because
physics/ itself was never added to sys.path -- only the repo root would
be, via the `physics.` dotted path. This is the exact failure
CLAUDE.md documents app/main.py hitting originally, and tests/conftest.py
works around it the same way this file does: add physics/ ITSELF to
sys.path (not just the repo root), then import flatly (`from grid import
Grid1D`), matching the convention every physics/ module already uses
internally. backend/ sits at the same repo-root nesting level app/ and
tests/ do (a sibling of physics/, not inside it), so the identical fix
applies unchanged.

Verified empirically: started uvicorn from the repo root and hit
GET /health -- see README for the exact command.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "physics"))

from fastapi import FastAPI

from grid import Grid1D  # flat import -- proves the sys.path setup above works

app = FastAPI(title="Fourier Optics Lithography Simulator API")


@app.get("/health")
def health() -> dict:
    """
    Liveness/readiness check. Also exercises a real physics/ import
    (Grid1D) on every call, not just at startup, so a broken sys.path
    setup would fail loudly here rather than only at import time.
    """
    grid = Grid1D(L=10.0, N=256)
    return {
        "status": "ok",
        "physics_import_check": {
            "module": "grid.Grid1D",
            "N": grid.N,
            "dx": grid.dx,
        },
    }
