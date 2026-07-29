"""
physics/lens2d.py
--------------------
2D generalization of lens.py: a genuine circular-aperture pupil and the
coherent imaging chain built on it.

WHY THIS MODULE EXISTS IN THE PIPELINE
-----------------------------------------
lens.py's pupil_function_freq is a 1D "brick wall": 1 where |fx| <= f_cutoff,
0 otherwise. That is already a simplification of Goodman's actual pupil
geometry -- a real lens aperture is a circular disk in the pupil plane, and
lens.py's own module docstring states the unit-magnification/1D treatment
as a deliberate simplifying assumption of the general theory, not the
general theory itself. This module removes exactly that simplification for
the frequency-domain cutoff: pupil_function_freq_2d is bounded by
fx^2 + fy^2 <= f_cutoff^2, a literal disk, matching a real lens's circular
aperture directly. Everything else mirrors lens.py's own coherent_aerial_image
structure: mask -> 2D spectrum -> pupil multiply -> inverse transform ->
intensity, using fft_engine.fft2d/ifft2d (Week 12 2D extension) instead of
fft1d/ifft1d.

WHAT THIS MODULE DELIBERATELY DOES NOT INCLUDE
------------------------------------------------------------------------
- lens.lens_focal_plane_field (the physically-scaled focal-plane field used
  ONLY for visualizing the pupil plane in µm) has no 2D counterpart here --
  the 2D extension's goal is the mask -> aerial-image -> printed-feature
  chain, not a pupil-plane visualization panel; nothing downstream needs it.
- Coherent imaging only: no 2D OTF/incoherent path, no defocus/aberrations.
  These are documented, deliberate scope boundaries for this extension (see
  docs/physics_assumptions.md's "2D Extension Assumptions" section), not
  oversights -- both are straightforward, well-understood generalizations of
  existing 1D code (imaging.py, aberrations.py) that simply were not the
  priority for this extension's stated goal (masks -> aerial image ->
  printed-feature heatmap visualization).
- No 2D OPC, no 2D edge-placement-error metric. 2D "edges" are contours, not
  a small number of scan-for-0/1-transition points along one axis, and
  correcting them requires genuinely different machinery (gauge points along
  a polygon boundary, biased along the local normal direction) -- an
  open-ended problem, not a mechanical extension of physics/opc.py. See
  physics/imaging2d.py's iou_score for this extension's simpler stand-in
  fidelity metric instead.

All spatial coordinates: µm
All spatial frequencies: cycles/µm (µm⁻¹)
Wavelength: µm
NA: dimensionless
"""

from typing import Tuple

import numpy as np

from fft_engine import fft2d, ifft2d
from constants import WAVELENGTH, NA_DEFAULT
from lens import cutoff_frequency


def pupil_function_freq_2d(grid, NA: float = NA_DEFAULT, wavelength: float = WAVELENGTH) -> np.ndarray:
    """
    Hard-edged, diffraction-limited CIRCULAR pupil function evaluated
    directly on a Grid2D's own 2D frequency meshgrid (grid.FX, grid.FY).

    Goodman connection
    -------------------
    This is the 2D generalization of lens.pupil_function_freq: P(wavelength
    *z*fx, wavelength*z*fy) from Eq. (6-20), a disk-shaped low-pass filter
    in the 2D frequency plane rather than a 1D brick wall. Reuses
    lens.cutoff_frequency UNCHANGED -- that function is purely
    NA/wavelength -> frequency (no axis-count dependence at all), so the
    same f_cutoff = NA/wavelength applies to a circular 2D cutoff exactly
    as it did to the 1D cutoff's +-f_cutoff bounds.

    WHY A CIRCLE, NOT A SQUARE
    -------------------------------------------------------------------
    A real lens aperture is a circular disk, not a square or a 1D slab --
    lens.py's own 1D pupil was necessarily a cross-sectional simplification
    of this (a 1D slice through the middle of a disk looks like a brick
    wall, but a disk itself is not one). This function is the first place
    in this project the pupil's actual 2D geometry is represented, rather
    than assumed away by working in one dimension.

    Parameters
    ----------
    grid       : Grid2D — provides the 2D frequency meshgrid grid.FX, grid.FY
    NA         : float — numerical aperture (defaults to constants.NA_DEFAULT)
    wavelength : float — wavelength, µm (defaults to constants.WAVELENGTH)

    Returns
    -------
    P : ndarray, shape (N, N) — 1.0 where FX^2 + FY^2 <= f_cutoff^2, else 0.0
    """
    f_cutoff = cutoff_frequency(NA, wavelength)
    return ((grid.FX ** 2 + grid.FY ** 2) <= f_cutoff ** 2).astype(float)


def coherent_aerial_image_2d(mask: np.ndarray, grid, wavelength: float = WAVELENGTH,
                              NA: float = NA_DEFAULT
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full forward coherent imaging chain in 2D: mask -> band-limited
    spectrum (2D FFT + circular pupil cutoff) -> aerial image intensity.

    Goodman connection
    -------------------
    Direct 2D generalization of lens.coherent_aerial_image: multiplying the
    mask's 2D spectrum by the (now circular) pupil function H(fx,fy) =
    P(wavelength*focal_length*fx, wavelength*focal_length*fy), then
    inverse-transforming, is the same transfer-function treatment of
    imaging (Eq. 6-13/6-17/6-20) applied on two frequency axes instead of
    one -- no new physics beyond pupil_function_freq_2d's circular cutoff.

    VALIDATION PERFORMED BY HAND BEFORE DELIVERY
    ------------------------------------------------
    - Wide-open-pupil check (f_cutoff forced far above grid.f_max, pupil
      all-ones): intensity reproduced the original binary 2D mask to
      floating-point precision, confirming the fft2d -> multiply -> ifft2d
      round trip is exact when nothing is filtered, mirroring lens.py's own
      1D plumbing check.
    - Resolution-limit check: a grid-aligned contact_hole_array with pitch
      safely below the circular cutoff showed strong 2D intensity
      modulation; an otherwise-identical array with pitch pushed above cutoff
      showed flat (unmodulated) intensity -- the 2D analogue of lens.py's
      own grating resolution-limit check.
    - Airy-disk check (tests/test_lens2d.py): the coherent point-spread
      function of a circular pupil (ifft2d of pupil_function_freq_2d applied
      to a single_hole mask far smaller than the resolution limit) matches
      the closed-form Airy/jinc profile 2*J1(rho)/rho along a radial cut,
      including the location of the first dark ring -- this is the exact
      "Airy pattern" docs/physics_assumptions.md originally planned for
      Week 8 (a circular-aperture Fraunhofer pattern) and never delivered,
      now fulfilled here as a direct consequence of the circular pupil
      rather than requiring a separate diffraction2d.py.

    Parameters
    ----------
    mask       : ndarray, shape (N, N) — 2D mask transmission (0/1), on
                  grid.X/grid.Y
    grid       : Grid2D — mask's spatial/frequency grid (from grid2d.py)
    wavelength : float — wavelength, µm (defaults to constants.WAVELENGTH)
    NA         : float — numerical aperture (defaults to constants.NA_DEFAULT)

    Returns
    -------
    field_image : ndarray, shape (N, N), complex — coherent image-plane
                   field, on the same grid.X/grid.Y as the input mask (unit
                   magnification, same simplifying assumption as lens.py)
    intensity   : ndarray, shape (N, N) — |field_image|^2, the 2D aerial
                   image (NOT peak-normalized, matching lens.py's own
                   convention: the all-frequencies-pass limit reduces
                   exactly to the original mask)
    P           : ndarray, shape (N, N) — the circular pupil actually
                   applied, returned for callers/plots without a second
                   pupil_function_freq_2d call
    """
    G_raw = fft2d(mask, grid.dx)
    P = pupil_function_freq_2d(grid, NA=NA, wavelength=wavelength)
    G_filtered = G_raw * P
    field_image = ifft2d(G_filtered, grid.dx)
    intensity = np.abs(field_image) ** 2
    return field_image, intensity, P
