# Physics Assumptions — Fourier Optics Lithography Simulator

## Overview
This simulator models optical lithography using scalar diffraction theory.
We track how a mask pattern propagates through an optical system to form an
aerial image on the wafer plane.

---

## Week 1 Assumptions (Representation Only)

### 1. 1D Scalar Fields
- All fields this week are 1D (one spatial dimension, `x`)
- Extension to 2D follows naturally via separability (covered Week 6+)
- Fields are real-valued binary arrays at this stage

### 2. Binary Mask
- Mask is a **binary amplitude mask**: values are either 0 (opaque) or 1 (transparent)
- No phase shift, no partial transmission, no phase-shifting mask (PSM) yet
- Physically: chrome-on-glass mask, fully blocking or fully passing light

### 3. Spatial Units
- Spatial coordinates are in **micrometers (µm)**
- Frequency coordinates are in **cycles/µm** (i.e., µm⁻¹)
- Wavelength will be introduced in Week 8+ (λ = 193 nm for ArF, λ = 13.5 nm for EUV)

### 4. Sampling / Grid
- Grid is uniformly spaced with spacing `dx` (µm)
- Total field size `L = N * dx` (µm)
- Nyquist limit: minimum feature resolved = `2 * dx`
- Frequency resolution: `df = 1/L` (cycles/µm)
- Max frequency representable: `f_max = 1/(2*dx)` (cycles/µm)
- **Rule of thumb**: use at least 8–10 samples per minimum feature width

### 5. Target Patterns (Week 1)
- **Single line**: a single opaque feature of width `w` centered in the field
  - Physically represents an isolated line on the mask
- **Line-space grating**: periodic array of lines with pitch `p` and duty cycle 0.5
  - Physically represents a dense array (equal lines and spaces)

### 6. What "Spectrum" Means Here
- The spectrum is the **1D Fourier transform** of the mask transmission function
- Magnitude tells you how much of each spatial frequency is present
- A single line → sinc-shaped spectrum (broad, many frequencies)
- A grating → discrete spikes at harmonics of 1/pitch

---

## Upcoming Assumptions (Weeks 2–12)
- Week 6: FFT normalization, physical frequency axis, sampling constraints
- Week 7: Angular spectrum propagation, evanescent cutoff
- Week 8: Fraunhofer (far-field) diffraction, Airy pattern
- Week 9: Lens as Fourier transformer, coherent imaging, NA pupil cutoff
- Week 10: Coherent (ATF) vs incoherent (OTF) imaging modes
- Week 11: Aberrations, focus error, generalized pupil
- Week 12: OPC correction loop
