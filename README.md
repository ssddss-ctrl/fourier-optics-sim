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
│   ├── main.py                # FastAPI app (live UI backend): 6 POST endpoints + /health
│   ├── schemas.py              # Pydantic request/response models
│   └── simulator.py            # physics/-calling logic, unit-tested independent of HTTP
├── frontend/                  # Vite + React + TypeScript app (live UI)
│   └── src/
│       ├── pages/Landing.tsx        # Animated hologram landing page
│       ├── pages/Simulator.tsx      # Fixed-viewport 4-page pager
│       └── components/simulator/    # Mask/tune/optics/results sections + OPC panel
├── app/
│   └── main_streamlit_archived.py   # Retired Streamlit app, kept for reference
├── physics/
│   ├── masks.py               # Grid + binary mask patterns          ✅ Week 1
│   ├── fft_engine.py          # FFT helpers, physical freq axis      ✅ Week 6
│   ├── propagation.py         # Angular spectrum propagator          ✅ Week 7
│   ├── diffraction.py         # Fraunhofer diffraction               ✅ Week 8
│   ├── lens.py                # Lens as Fourier transformer          ✅ Week 9
│   ├── imaging.py             # ATF/OTF imaging models               ✅ Week 10
│   ├── aberrations.py         # Defocus wavefront + generalized pupil ✅ Week 11
│   └── opc.py                 # Edge-bias OPC correction loop        ✅ Week 12
├── plotting/
│   ├── core.py                 # Matplotlib scaffold for scripts/generate_*.py PNGs
│   └── interactive.py          # Plotly dark theme (ported to frontend/src/lib/plotlyTheme.ts)
├── notebooks/
│   └── opc_demo.ipynb          # Naive mask -> distorted print -> OPC -> corrected print
├── docs/
│   └── physics_assumptions.md
├── requirements.txt           # physics/ + tests/ + scripts/ (matplotlib, no web framework)
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

## Physics reference

See `docs/physics_assumptions.md` for all modeling assumptions.

Primary reference: Goodman, *Introduction to Fourier Optics*, 4th ed.
