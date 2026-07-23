# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A from-scratch coherent optical lithography simulator, built week-by-week alongside Goodman's
*Introduction to Fourier Optics* (4th ed.). Each `physics/` module corresponds to a specific
chapter/week of that build and implements one stage of the forward imaging chain:

```
Mask pattern → Angular spectrum propagation → Lens (NA cutoff) → Aerial image → Threshold → Printed feature → OPC correction
```

See `README.md` for the week-by-week module map and `docs/physics_assumptions.md` for the
modeling assumptions in force at each stage (units, coherence, pattern definitions, etc.) —
consult it before changing sampling/grid behavior.

## Commands

```bash
# Install (physics/ + tests/scripts — matplotlib, no web framework)
pip install -r requirements.txt

# Run the app (React frontend + FastAPI backend — two processes)
pip install -r requirements-backend.txt
uvicorn backend.main:app --reload --reload-dir backend --reload-dir physics   # http://localhost:8000
cd frontend && npm install && npm run dev                                     # http://localhost:5173

# Run all tests
python -m pytest

# Run a single test file / test
python -m pytest tests/test_lens.py
python -m pytest tests/test_lens.py::test_pupil_function_freq_hard_edge

# Regenerate the reference plots in assets/
python scripts/generate_diffraction_patterns.py
python scripts/generate_imaging_comparison.py
python scripts/generate_lens_imaging.py
python scripts/generate_aberration_comparison.py
```

There is no lint/format/typecheck command configured for `physics/`/`backend/` (no ruff/black/mypy
config present). `pyrightconfig.json` sets `extraPaths` for editor import resolution only.
`frontend/` has its own `npm run lint` (ESLint, configured by the Vite scaffold).

The retired Streamlit app (`app/main_streamlit_archived.py`) still runs via
`streamlit run app/main_streamlit_archived.py` if needed for comparison, but it is not the
maintained UI — see "Frontend/backend split" below.

## Architecture

### Pipeline order (each module builds on the previous, not reimplements it)

1. **`physics/masks.py`** — grid construction (`make_grid`) and binary mask/target patterns
   (`single_line`, `line_space_grating`, `two_lines`). No cross-module dependencies.
2. **`physics/fft_engine.py`** — the only place raw `np.fft` calls are turned into physically
   labeled spectra (`fft1d`/`ifft1d`, cycles/µm) and where grid sufficiency is checked
   (`check_sampling`, Nyquist + space-bandwidth product per Goodman Eq. 2-57/2-58).
3. **`physics/grid.py`** — `Grid1D` bundles `L, N, x, dx, f, df, f_max` and wraps
   `check_sampling` as `.verify_sampling(min_feature)`. Every physics module downstream of this
   point takes a `Grid1D` rather than raw `L`/`N`/`dx` arguments.
4. **`physics/propagation.py`** — angular spectrum propagator (`propagate`), exact at any
   distance `z`, splitting propagating vs. evanescent spectral components.
5. **`physics/diffraction.py`** — Fraunhofer (far-field) limit of the same physics, plus
   closed-form analytic patterns (sinc², grating orders) used to validate the numerical FFT
   pipeline against textbook results.
6. **`physics/lens.py`** — lens-as-Fourier-transformer: exact FT at a finite focal length, plus
   the hard-edged pupil cutoff (`pupil_function_freq`) that band-limits the mask spectrum by NA.
   Produces `coherent_aerial_image`. Assumes unit-magnification, telecentric, infinite-conjugate
   imaging only (stated explicitly in the module docstring).
7. **`physics/imaging.py`** — adds the incoherent/OTF path (`incoherent_aerial_image`, built
   from `lens.py`'s ATF/pupil, not reimplemented), intensity thresholding (`apply_threshold`),
   and print-fidelity metrics (`edge_placement_error`, `linewidth_error`). Thresholding/EPE are
   lithography-engineering conventions, not Goodman equations — flagged as such in-module.
8. **`physics/constants.py`** — project-wide `WAVELENGTH`/`NA_DEFAULT` defaults only; every
   consumer still takes wavelength/NA as explicit optional parameters (no hidden physical
   constants inside function bodies).
9. **`plotting/core.py`** — the shared 3-panel (target/mask/spectrum) figure scaffold
   (`three_panel_plot`) plus shared style constants (`MASK_COLOR`, `TARGET_COLOR`,
   `SPECTRUM_COLOR`, `IMAGE_COLOR`, `_style_ax`), used only by `scripts/generate_*.py` for the
   static build-log PNGs now (see "Frontend/backend split" below for the live UI).
10. **`backend/main.py`** — FastAPI app; the current live consumer of the full pipeline
    end-to-end (replaces `app/main_streamlit_archived.py` in that role).

Not yet implemented: the OPC correction loop (Week 12) — referenced as locked/upcoming in
`backend/`/`frontend/` and `README.md`. Aberrations (Week 11) are implemented in
`physics/aberrations.py`, not `physics/pupil.py` as originally planned.

### Frontend/backend split (replaces the Streamlit app)

The Streamlit app (`app/main.py`) has been retired to `app/main_streamlit_archived.py` (kept for
reference only, not maintained) and replaced by:

- **`backend/`** — a FastAPI app (`backend/main.py`) that imports `physics/` directly (no
  `plotting/`, no matplotlib/plotly — those were UI-layer concerns of the old Streamlit app).
  `requirements-backend.txt` covers its dependencies (`fastapi`, `uvicorn`, `numpy`) separately
  from `requirements.txt`, which now only serves `physics/`/`tests/`/`scripts/`.
- **`frontend/`** — a Vite + React + TypeScript app (`react-router-dom`, `three` /
  `@react-three/fiber` / `@react-three/drei`, `framer-motion`, `tailwindcss`) that will call the
  FastAPI backend once its computation endpoints exist. As of this migration it's a barebones
  scaffold (a placeholder route) — no aerial-image/ATF-OTF panels have been ported from the
  Streamlit app yet.

`physics/` itself did not change for this migration — the split is purely about how the UI talks
to it (HTTP/JSON via FastAPI instead of direct Python calls inside a Streamlit script rerun).

### Import convention (flat, not a package)

There is no `src/` layout and internal modules import each other **flatly**, matching how the
project actually loads at runtime:

```python
# inside physics/lens.py, physics/grid.py, etc.
from fft_engine import fft1d, ifft1d
from constants import WAVELENGTH, NA_DEFAULT
from masks import make_grid
```

This only works because `physics/` and `plotting/` are added to `sys.path` *directly* (not just
the repo root) before anything imports from them:
- `tests/conftest.py` does this for pytest.
- `app/main_streamlit_archived.py` did this itself at the top of the file (repo root + `physics/`
  + `plotting/`) when it was the live app.
- `backend/main.py` does this now (repo root + `physics/` — no `plotting/`, the backend has no
  matplotlib/plotly dependency at all).

If you add a new entry point (a script, a notebook, another app), it needs the same `sys.path`
setup — importing via `physics.modulename` will fail with `ModuleNotFoundError` for any module
that itself imports a sibling flatly (`grid.py`, `lens.py`, `imaging.py`, `propagation.py`,
`diffraction.py`, `aberrations.py` all do this). `physics/__init__.py`/`plotting/__init__.py` use
relative imports and only re-export a handful of Week-1 names — don't rely on them for anything
added since. `backend/` is a sibling of `physics/` at the repo-root nesting level (not inside it),
exactly like `app/`/`tests/` were — the identical fix applies unchanged; verified directly by
starting `uvicorn backend.main:app` and hitting `/health`, which imports and calls `grid.Grid1D`
on every request.

### Units convention (every module states this in its header — keep consistent)

- Spatial coordinates: **µm**
- Spatial frequencies: **cycles/µm (µm⁻¹)**
- Wavelength: **µm** (`app/main.py` converts its UI input from nm to µm before calling into
  `physics/`)
- NA: dimensionless

### Documentation convention

Every `physics/` module carries a "WHY THIS MODULE EXISTS IN THE PIPELINE" docstring section
tying it to the specific Goodman equation numbers it implements and to what sits upstream/
downstream of it, plus explicit callouts when something is a simplifying assumption or an
engineering convention rather than textbook physics (e.g. thresholding in `imaging.py`, the
unit-magnification assumption in `lens.py`). New physics code should follow this pattern —
cite the Goodman section/equation, state what's reused vs. new, and flag any deviation from the
general theory explicitly rather than leaving it implicit.

### Tests

`tests/` mirrors `physics/` one-to-one (`test_grid.py`, `test_fft_engine.py`,
`test_propagation.py`, `test_diffraction.py`, `test_lens.py`, `test_imaging.py`). Tests validate
numerics against hand calculations and closed-form Goodman results, not just shape/type checks —
follow that standard for new physics functions.

## Weekly workflow

This project follows a fixed weekly process. When asked to build a week's deliverables:

1. **Files first.** Confirm which existing files are needed before writing anything. Do not
   create or modify files until the necessary context/files are available.
2. **One file at a time.** Deliver one file, explain (a) what changed vs. the previous version,
   (b) which Goodman equation/concept it implements, (c) any textbook equations used that
   weren't in the handwritten notes, (d) the physical role this step plays in the pipeline and
   how it connects to steps before/after. Wait for confirmation before moving to the next file.
3. **Verify before delivering.** Trace through at least one representative input by hand or by
   running it — do not deliver unverified code.
4. **Test.** After all files are delivered, provide the exact `pytest` command to run.
5. **Git.** If tests pass, provide exact commands to commit and push. If tests fail, fix before
   proceeding.

Physics/equation research (searching Goodman for relevant sections, deciding what a chapter
requires) happens in a separate Claude.ai project with the textbook in context — that reasoning
isn't done here. This repo's job is implementation, verification, and testing.

## Build log

Build logs are written in Claude.ai, not here — this repo doesn't need to produce them. If asked
to summarize a week's changes for a build log, keep it factual (files changed, what was
validated, bugs hit) rather than drafting prose.
