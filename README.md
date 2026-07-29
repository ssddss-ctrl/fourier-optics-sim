# Fourier Optics Lithography Simulator

A ground-up implementation of a coherent optical lithography simulator, built
alongside Goodman's *Introduction to Fourier Optics*.

## What it does

Simulates the full optical lithography chain:

```
Mask pattern → Angular spectrum → Lens (NA cutoff) → Aerial image → Threshold → Printed feature → OPC correction
```

Given a mask pattern and optical parameters (λ, NA, coherence, aberrations),
the simulator predicts what actually gets printed on the wafer — and can
automatically correct the mask to compensate for optical distortion (OPC).

## Running the app

The app is a React frontend + FastAPI backend (two processes, run in separate terminals):

```bash
# Backend (FastAPI) — http://localhost:8000
pip install -r requirements-backend.txt
uvicorn backend.main:app --reload --reload-dir backend --reload-dir physics

# Frontend (Vite + React + TypeScript) — http://localhost:5173
cd frontend
npm install
npm run dev
```

The previous Streamlit app is retired but kept for reference at
`app/main_streamlit_archived.py` (`streamlit run app/main_streamlit_archived.py`, using
`pip install -r requirements.txt`) — it is not the maintained UI.

## Project structure

```
fourier_optics_sim/
├── backend/
│   ├── main.py                # FastAPI app (live UI backend): 7 POST endpoints + /health
│   ├── schemas.py              # Pydantic request/response models (1D + 2D)
│   └── simulator.py            # physics/-calling logic, unit-tested independent of HTTP
├── frontend/                  # Vite + React + TypeScript app (live UI)
│   └── src/
│       ├── pages/Landing.tsx        # Animated hologram landing page
│       ├── pages/Simulator.tsx      # Fixed-viewport 4-page pager (1D)
│       ├── pages/Simulator2D.tsx    # 2D mask/heatmap showcase page
│       ├── components/simulator/    # 1D mask/tune/optics/results sections + OPC panel
│       └── lib/heatmapTheme.ts      # Validated colorscales for the 2D heatmap panels
├── app/
│   └── main_streamlit_archived.py   # Retired Streamlit app, kept for reference
├── physics/
│   ├── masks.py               # Grid + binary mask patterns          ✅ Week 1
│   ├── fft_engine.py          # FFT helpers, physical freq axis      ✅ Week 6 (+ 2D: fft2d/ifft2d)
│   ├── propagation.py         # Angular spectrum propagator          ✅ Week 7
│   ├── diffraction.py         # Fraunhofer diffraction               ✅ Week 8
│   ├── lens.py                # Lens as Fourier transformer          ✅ Week 9
│   ├── imaging.py             # ATF/OTF imaging models               ✅ Week 10
│   ├── aberrations.py         # Defocus wavefront + generalized pupil ✅ Week 11
│   ├── opc.py                 # Edge-bias OPC correction loop        ✅ Week 12
│   ├── grid2d.py               # 2D extension: Grid2D
│   ├── masks2d.py              # 2D extension: contact-hole array, chip-block layout
│   ├── lens2d.py                # 2D extension: circular pupil, 2D coherent aerial image
│   └── imaging2d.py            # 2D extension: IoU fidelity score (no 2D OPC/EPE -- see below)
├── plotting/
│   ├── core.py                 # Matplotlib scaffold for scripts/generate_*.py PNGs
│   └── interactive.py          # Plotly dark theme (ported to frontend/src/lib/plotlyTheme.ts)
├── notebooks/
│   └── opc_demo.ipynb          # Naive mask -> distorted print -> OPC -> corrected print
├── docs/
│   └── physics_assumptions.md
├── requirements.txt           # physics/ + tests/ + scripts/ (matplotlib, scipy, no web framework)
├── requirements-backend.txt   # backend/ (fastapi, uvicorn, numpy)
└── README.md
```

## Week-by-week build plan

| Week | Physics topic | Module added | App feature unlocked |
|------|--------------|--------------|----------------------|
| 1 | 1D Fourier fundamentals, mask representation | `masks.py` | Pattern designer + spectrum viewer |
| 6 | FFT normalization, sampling, space-bandwidth | `fft_engine.py` | Sampling diagnostics |
| 7 | Angular spectrum, evanescent cutoff | `propagation.py` | Propagation distance slider |
| 8 | Fraunhofer diffraction, pattern library | `diffraction.py` | Far-field diffraction viewer |
| 9 | Lens as FT, coherent imaging, NA pupil | `lens.py` | **Aerial image panel** |
| 10 | ATF vs OTF, thresholding, print error | `imaging.py` | **Printed feature panel** |
| 11 | Aberrations, focus error | `aberrations.py` | Focus error sweep |
| 12 | OPC correction loop | `opc.py` | **OPC correction panel** |

Every module above is implemented and live in `frontend/`/`backend/` (the React/FastAPI app) —
there is no remaining pipeline stage still only in the archived Streamlit app.

## 2D Extension

A separate addendum on top of the Week 1–12 build above (not a continuation of the weekly
sequence — that build plan is complete): a second, smaller pipeline generalizing masks, the
lens pupil, and coherent imaging to two spatial dimensions, at `/simulator-2d`.

```
2D mask (contact-hole array / chip-block layout) → circular-pupil lens → 2D aerial image
    → threshold → printed feature → IoU fidelity score
```

- **Real 2D patterns**: a periodic contact/via array and a simple interconnect-style block
  layout, both dark-field 2D masks (`physics/masks2d.py`), not a 1D cross-section.
- **A genuine circular lens pupil** (`physics/lens2d.py`) — physically more correct than the 1D
  pipeline's brick-wall pupil approximation, since a real lens aperture is a disk.
- **Closed-form validation**: the circular pupil's coherent point-spread function matches
  Goodman's classic Airy-disk profile (`scipy.special.j1`), finally delivering the "Week 8: Airy
  pattern" this project's docs originally planned but never built in 1D.
- **What's deliberately NOT included**: 2D OPC and a formal 2D edge-placement-error metric. A 2D
  "edge" is a contour, not a point along one axis — correcting it needs gauge points along a
  polygon boundary biased along the local normal, genuinely different machinery than
  `physics/opc.py`'s 1D loop, not a mechanical extension of it. `physics/imaging2d.py`'s
  `iou_score` (intersection-over-union between printed and target) is this extension's simpler
  fidelity stand-in instead. See `docs/physics_assumptions.md`'s "2D Extension Assumptions"
  section for the full list of scoped-out boundaries.

## Physics reference

See `docs/physics_assumptions.md` for all modeling assumptions.

Primary reference: Goodman, *Introduction to Fourier Optics*, 4th ed.
