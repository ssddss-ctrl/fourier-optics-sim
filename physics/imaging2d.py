"""
physics/imaging2d.py
------------------------
2D counterpart to imaging.py: the incoherent/OTF imaging path, plus
print-fidelity quantification, for the 2D mask -> aerial image ->
printed-feature chain.

WHY THIS MODULE EXISTS IN THE PIPELINE
-----------------------------------------
imaging.py adds two things on top of lens.py's coherent path: the
incoherent/OTF imaging path (built from lens.py's own ATF/pupil, not
reimplemented), and the thresholding/print-fidelity quantification that
closes the loop. This module does both analogous jobs for 2D:

    2D mask --[lens2d.py pupil]--> 2D ATF --[this module]--> 2D OTF
        --[this module]--> incoherent 2D aerial image
    2D mask --[lens2d.coherent_aerial_image_2d]--> coherent 2D aerial image
    { either aerial image } --[threshold]--> printed feature
    { 2D target, 2D printed feature } --[this module]--> fidelity score

Mirrors imaging.py's own split exactly: lens2d.py owns the pupil and the
coherent path; this module owns the OTF/incoherent path and the
print-fidelity metric, built on top of lens2d.py rather than duplicating
its pupil logic.

`imaging.apply_threshold` IS REUSED HERE UNCHANGED -- NOT REIMPLEMENTED
------------------------------------------------------------------------
apply_threshold(intensity, threshold) is already a pure elementwise
comparison (`intensity >= threshold`) with no assumption about array
dimensionality baked in. It works correctly, with no modification at all,
on a 2D intensity array exactly as it does on a 1D one. Callers of this
2D pipeline should import apply_threshold directly from imaging.py; it is
not re-exported or wrapped here, to keep it unambiguous that this is the
exact same function, not a look-alike 2D copy.

WHY THIS MODULE DOES NOT PROVIDE A 2D EPE/LINEWIDTH-ERROR EQUIVALENT
------------------------------------------------------------------------
imaging.find_edges/edge_placement_error/linewidth_error are fundamentally
1D algorithms: they scan a single array for 0<->1 transitions along one
axis. A 2D binary pattern's "edges" are CONTOURS -- the boundaries of 2D
regions -- and measuring how far a printed contour deviates from a target
contour requires choosing gauge points along the boundary and biasing each
perpendicular to the local edge direction (this is, not coincidentally,
also exactly the missing piece for a hypothetical 2D OPC loop). That is a
genuinely different, more open-ended algorithm than a 1D array scan, not a
mechanical dimensional extension -- explicitly out of scope for this
project's 2D extension (see docs/physics_assumptions.md's "2D Extension
Assumptions" section). iou_score below is a deliberately SIMPLER stand-in
metric that answers the same practical question ("how well did it print?")
without needing any contour/edge detection at all.

All spatial coordinates: µm
"""

from typing import Optional, Tuple

import numpy as np

from fft_engine import fft2d, ifft2d
from constants import WAVELENGTH, NA_DEFAULT
from lens2d import pupil_function_freq_2d


# ── ATF / OTF (2D) ───────────────────────────────────────────────────────────

def amplitude_point_spread_function_2d(grid, wavelength: float = WAVELENGTH,
                                        NA: float = NA_DEFAULT) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coherent amplitude point-spread function h(x,y): the 2D inverse Fourier
    transform of the circular pupil -- the direct 2D generalization of
    imaging.amplitude_point_spread_function, using lens2d.pupil_function_freq_2d
    (reused unchanged) in place of lens.pupil_function_freq.

    No defocus_waves parameter -- this 2D extension is coherent/incoherent
    only, no aberrations (see docs/physics_assumptions.md's "2D Extension
    Assumptions"). H is always the bare circular pupil, unlike the 1D
    function's optional generalized-pupil branch.

    Parameters
    ----------
    grid       : Grid2D — provides the 2D frequency meshgrid grid.FX, grid.FY
    wavelength : float — wavelength, µm (defaults to constants.WAVELENGTH)
    NA         : float — numerical aperture (defaults to constants.NA_DEFAULT)

    Returns
    -------
    h : ndarray, shape (N, N), complex — amplitude PSF (index-0-centered,
         same DFT-seam convention ifft2d/ifft1d always produce -- see
         lens2d.py's own Airy-disk test for how to re-center it if a
         physically-aligned radial cut is ever needed)
    H : ndarray, shape (N, N) — the circular pupil actually used, returned
         alongside h so callers don't need a second call to reconstruct it
    """
    H = pupil_function_freq_2d(grid, NA=NA, wavelength=wavelength)
    h = ifft2d(H, grid.dx)
    return h, H


def optical_transfer_function_2d(grid, wavelength: float = WAVELENGTH,
                                  NA: float = NA_DEFAULT) -> Tuple[np.ndarray, np.ndarray]:
    """
    2D optical transfer function (OTF): the frequency response for
    INCOHERENT illumination -- the direct 2D generalization of
    imaging.optical_transfer_function (route (1): FT of the intensity PSF
    |h|^2, normalized by its DC value).

    VALIDATION PERFORMED BY HAND BEFORE DELIVERY
    ------------------------------------------------
    - OTF(0,0) == 1.0 exactly (by construction, the same DC-normalization
      argument as the 1D function).
    - max(|OTF|) == 1.0, attained only at DC (Property 3, Schwarz's
      inequality -- the same proof the 1D docstring cites applies
      unchanged in 2D, since it doesn't depend on axis count).
    - max(|Im(OTF)|) ~ 1e-9 (real to numerical precision) -- expected,
      since this 2D extension has no aberration path, so the pupil is
      always real (unlike the 1D module's optional complex generalized
      pupil).

    Parameters
    ----------
    grid       : Grid2D — provides the 2D frequency meshgrid grid.FX, grid.FY
    wavelength : float — wavelength, µm (defaults to constants.WAVELENGTH)
    NA         : float — numerical aperture (defaults to constants.NA_DEFAULT)

    Returns
    -------
    OTF : ndarray, shape (N, N), complex — normalized 2D OTF (OTF(0,0) = 1
           exactly; real-valued to numerical precision, since no
           aberration path exists in this 2D extension)
    H   : ndarray, shape (N, N) — the circular pupil (ATF) actually applied
    """
    h, H = amplitude_point_spread_function_2d(grid, wavelength=wavelength, NA=NA)
    intensity_psf = np.abs(h) ** 2
    OTF_raw = fft2d(intensity_psf, grid.dx)
    # DC index: same 1D index works on both axes since the grid is square
    # (grid.fx == grid.fy), exactly like grid2d.py's own verify_sampling
    # note that a square grid's per-axis checks are identical.
    dc_index = int(np.argmin(np.abs(grid.fx)))
    OTF = OTF_raw / OTF_raw[dc_index, dc_index]
    return OTF, H


# ── Aerial images (2D) ───────────────────────────────────────────────────────

def incoherent_aerial_image_2d(mask: np.ndarray, grid, wavelength: float = WAVELENGTH,
                                NA: float = NA_DEFAULT
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Incoherent 2D aerial image: the wafer-plane intensity produced by
    imaging `mask` through the circular-pupil system under INCOHERENT
    illumination -- the direct 2D generalization of
    imaging.incoherent_aerial_image, using fft2d/ifft2d in place of
    fft1d/ifft1d and this module's own optical_transfer_function_2d in
    place of imaging.optical_transfer_function.

    Frequency-domain multiplication (fft2d(mask) * OTF -> ifft2d), not a
    real-space convolution with the intensity PSF, for the identical
    reason imaging.incoherent_aerial_image gives for the 1D case: h's
    DFT-seam indexing (see amplitude_point_spread_function_2d's docstring)
    doesn't line up with grid.X/grid.Y's own indexing, so a naive
    real-space convolution would silently produce a shifted image. Working
    entirely in frequency domain and inverse-transforming once at the end
    sidesteps that regardless of how the intermediate PSF happens to be
    centered -- both fft2d and ifft2d always operate consistently on the
    same grid.X/grid.Y <-> grid.FX/grid.FY pair.

    VALIDATION PERFORMED BY HAND BEFORE DELIVERY
    ------------------------------------------------
    - Wide-open-pupil check (NA/wavelength combo whose cutoff safely
      exceeds grid.f_max): incoherent image reproduced the original 2D
      binary mask to floating-point precision, the same limiting-case
      check lens2d.coherent_aerial_image_2d already performs.
    - DC/energy check: mean(intensity) equals mean(mask) to floating-point
      precision, confirming OTF(0,0)=1's normalization preserves the
      object's average intensity regardless of NA.
    - Realistic-NA check: intensity stayed non-negative everywhere.
    - Resolution-limit check: differs measurably from the coherent path
      for a contact-hole array near the resolution limit, the same
      contrast the 1D test_incoherent_differs_from_coherent check makes.

    Parameters
    ----------
    mask       : ndarray, shape (N, N) — mask transmission (0/1), on
                  grid.X/grid.Y
    grid       : Grid2D — mask's spatial/frequency grid (from grid2d.py)
    wavelength : float — wavelength, µm (defaults to constants.WAVELENGTH)
    NA         : float — numerical aperture (defaults to constants.NA_DEFAULT)

    Returns
    -------
    intensity : ndarray, shape (N, N) — incoherent 2D aerial image, on
                 grid.X/grid.Y (NOT peak-normalized, matching
                 lens2d.coherent_aerial_image_2d's own convention)
    OTF       : ndarray, shape (N, N), complex — the OTF actually applied
    H         : ndarray, shape (N, N) — the circular pupil (ATF) actually
                 applied, returned for side-by-side plotting against the
                 coherent path's own P
    """
    OTF, H = optical_transfer_function_2d(grid, wavelength=wavelength, NA=NA)
    G_obj = fft2d(mask, grid.dx)
    intensity = np.real(ifft2d(G_obj * OTF, grid.dx))
    return intensity, OTF, H


# ── Thresholding and print-fidelity metric (engineering, not Goodman) ───────

def iou_score(target: np.ndarray, printed: np.ndarray) -> Tuple[float, Optional[str]]:
    """
    Intersection-over-Union (Jaccard index) between a 2D printed pattern
    and its target -- this project's 2D stand-in for the 1D pipeline's
    edge-placement-error/linewidth-error metrics.

    NOT A GOODMAN EQUATION -- STANDARD IMAGE-SEGMENTATION METRIC
    ------------------------------------------------------------------
    IoU is a standard overlap metric from image segmentation/object
    detection, not a result from Fourier optics -- used here purely as an
    engineering convenience, exactly the same spirit in which imaging.py
    flags apply_threshold/edge_placement_error/linewidth_error as
    lithography-engineering conventions rather than textbook physics.

    WHY IoU, NOT A PLAIN OVERLAP FRACTION
    ------------------------------------------------------------------
    A simpler "intersection / target_area" fraction is gameable: an
    all-white, badly over-exposed `printed` pattern (printed=1 everywhere)
    would score a false 1.0 against ANY non-empty target, since every
    target pixel is trivially "covered." IoU's union term in the
    denominator correctly penalizes exactly that failure mode -- an
    over-printed pattern also enlarges the union, pulling the score down,
    not just the intersection up.

    Degenerate case (both target and printed are entirely empty, so
    union == 0): returns (NaN, warning string), mirroring imaging.py's own
    edge_placement_error/linewidth_error convention of returning NaN plus
    an explanatory string on an ill-defined case rather than silently
    guessing a value (e.g. returning 1.0 for "nothing was supposed to
    print and nothing printed" would hide a threshold/NA combination that
    is probably wrong, not a genuinely perfect match).

    Parameters
    ----------
    target  : ndarray — intended binary pattern (0/1), any shape
    printed : ndarray — thresholded printed-feature estimate (0/1), same
               shape as target

    Returns
    -------
    score   : float — intersection/union in [0.0, 1.0], or NaN if both
               patterns are entirely empty
    warning : str or None — explanatory message when score is NaN, else None
    """
    target_on = target > 0.5
    printed_on = printed > 0.5

    intersection = np.logical_and(target_on, printed_on).sum()
    union = np.logical_or(target_on, printed_on).sum()

    if union == 0:
        return float("nan"), "Target and printed pattern are both empty -- IoU is undefined."

    return float(intersection) / float(union), None
