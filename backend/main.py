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

SIMULATOR ENDPOINTS (backend/schemas.py + backend/simulator.py)
------------------------------------------------------------------------
The four POST endpoints below wrap physics/ for the simulator page,
mirroring exactly what app/main_streamlit_archived.py computes (same
default parameter values, same pattern_type/coherence branching, same
EPE/linewidth-error edge cases) -- physics/ itself is not modified, only
called through. All request/response validation is Pydantic
(backend/schemas.py); all the actual physics/-calling logic is in
backend/simulator.py, kept separate from this file so it can be
unit-tested directly without going through HTTP (see tests/test_api.py).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "physics"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from grid import Grid1D  # flat import -- proves the sys.path setup above works

from .schemas import (
    MaskRequest,
    MaskResponse,
    AtfOtfRequest,
    AtfOtfResponse,
    AerialImageRequest,
    AerialImageResponse,
    PrintedFeatureRequest,
    PrintedFeatureResponse,
)
from . import simulator

app = FastAPI(title="Fourier Optics Lithography Simulator API")

# Vite's dev server (frontend/) runs on localhost:5173 by default -- allow
# it explicitly rather than a wildcard, since this is a real (if small)
# CORS policy, not just a local convenience toggle.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.post("/api/mask", response_model=MaskResponse)
def api_mask(req: MaskRequest) -> dict:
    return simulator.compute_mask(req)


@app.post("/api/aerial-image", response_model=AerialImageResponse)
def api_aerial_image(req: AerialImageRequest) -> dict:
    return simulator.compute_aerial_image(req)


@app.post("/api/atf-otf", response_model=AtfOtfResponse)
def api_atf_otf(req: AtfOtfRequest) -> dict:
    return simulator.compute_atf_otf(req)


@app.post("/api/printed-feature", response_model=PrintedFeatureResponse)
def api_printed_feature(req: PrintedFeatureRequest) -> dict:
    return simulator.compute_printed_feature(req)
