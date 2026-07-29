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

Note (added with the 2D extension, below): the "Airy pattern" listed for Week 8 was never built
in 1D -- a circular aperture has no 1D cross-section that reduces to it. It is finally delivered
by the 2D extension's `physics/lens2d.py` (a genuine circular pupil) instead, validated directly
against this closed form in `tests/test_lens2d.py`.

---

## 2D Extension Assumptions

Everything above (Weeks 1–12) is a 1D theory: one spatial axis `x`, one frequency axis `fx`.
This section documents the assumptions in force for the separate 2D extension
(`physics/grid2d.py`, `masks2d.py`, `lens2d.py`, `imaging2d.py`, `/simulator-2d`), which
generalizes masks, the lens pupil, and coherent imaging to two spatial dimensions -- it is an
addendum on top of the Week 1-12 build, not a continuation of the weekly sequence (that build
plan is complete; see `README.md`).

### 1. Square field, square grid only
A single `L` (field width, µm) and `N` (points per axis) apply to BOTH the `x` and `y` axes --
no independent per-axis field widths or sample counts. Every 2D pattern this extension defines
(contact-hole arrays, a simple rectangular chip-block layout) reads naturally on a square field.

### 2. A genuine circular pupil, not the 1D brick-wall
`lens.py`'s 1D `pupil_function_freq` is `|fx| <= f_cutoff` -- already a simplification of a real
lens's actual aperture geometry, which is a circular disk. `lens2d.py`'s
`pupil_function_freq_2d` is `fx^2 + fy^2 <= f_cutoff^2`, a literal circle -- MORE physically
correct than the 1D module, not a mere dimensional extension of its approximation. Both use the
identical NA/wavelength -> frequency relation (`lens.cutoff_frequency`, reused unchanged, since
it has no axis-count dependence).

### 3. Coherent imaging only
`lens2d.coherent_aerial_image_2d` has no defocus/aberration parameter and no 2D incoherent/OTF
path -- both are straightforward, well-understood generalizations of existing 1D code
(`imaging.py`, `aberrations.py`) that simply were not the priority for this extension's stated
goal (a mask -> aerial-image -> printed-feature heatmap visualization). Extension points for a
future pass, not oversights.

### 4. No 2D OPC, no formal 2D edge-placement-error metric
This is the one deliberate, load-bearing scope boundary of the entire 2D extension, not a
missing feature. A 1D "edge" (`imaging.find_edges`) is a scan for a 0<->1 transition along one
axis -- a small, well-defined set of points. A 2D binary pattern's "edges" are CONTOURS: the
boundaries of 2D regions. Measuring how far a printed contour deviates from a target contour
requires choosing gauge points along the boundary and biasing each perpendicular to the local
edge direction -- genuinely different, more open-ended machinery than `physics/opc.py`'s 1D
loop, not a mechanical dimensional extension of it (this is, not coincidentally, also why real
commercial OPC/resolution-enhancement tools are complex: model-based correction with hundreds of
gauge points per polygon edge, run iteratively). `physics/imaging2d.py`'s `iou_score`
(intersection-over-union between the printed and target binary patterns) is this extension's
deliberately simpler fidelity stand-in -- a standard image-segmentation metric, not a Goodman
result, chosen specifically because it needs no edge/contour detection at all. Degenerate case
(both patterns entirely empty) returns NaN plus a warning string, mirroring `imaging.py`'s own
EPE/linewidth-error convention rather than silently guessing a value.

### 5. Sampling/performance defaults
`L=10.0` µm (matching the 1D default), `N=128` per axis. Chosen for the live backend's JSON
payload size over HTTP (a 128x128 array is ~16,384 floats; bundling four such arrays in one
`/api/2d/simulate` response lands around 0.6-0.8 MB, acceptable for one debounced request) rather
than for any physics reason -- `N=256` would roughly quadruple that payload; `N=64` renders
visibly blocky for a contact-hole array.
