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

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

## Project structure

```
fourier_optics_sim/
├── app/
│   └── main.py               # Streamlit app (the final product)
├── physics/
│   ├── masks.py              # Grid + binary mask patterns          ✅ Week 1
│   ├── fft_engine.py         # FFT helpers, physical freq axis      🔜 Week 6
│   ├── propagation.py        # Angular spectrum propagator          🔜 Week 7
│   ├── diffraction.py        # Fraunhofer diffraction               🔜 Week 8
│   ├── lens.py               # Lens as Fourier transformer          🔜 Week 9
│   ├── imaging.py            # ATF/OTF imaging models               🔜 Week 10
│   └── pupil.py              # Generalized pupil + aberrations      🔜 Week 11
├── plotting/
│   └── core.py               # Reusable 4-panel plotting scaffold   ✅ Week 1
├── docs/
│   └── physics_assumptions.md
├── requirements.txt
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
| 11 | Aberrations, focus error | `pupil.py` | Focus error sweep |
| 12 | OPC correction loop | (in app) | **OPC correction panel** |

## Physics reference

See `docs/physics_assumptions.md` for all modeling assumptions.

Primary reference: Goodman, *Introduction to Fourier Optics*, 4th ed.
