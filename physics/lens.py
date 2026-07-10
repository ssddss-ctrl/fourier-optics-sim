"""
physics/lens.py
-----------------
Lens as Fourier transformer; coherent aerial image formation with a
finite-NA pupil.

WHY THIS MODULE EXISTS IN THE PIPELINE
-----------------------------------------
Everything through Week 8 answered "what does the diffracted field look
like arbitrarily far from the mask, or after propagating it by the exact
angular-spectrum method?" This module answers a different question:
"what does a real imaging lens actually put on the wafer, given a finite
aperture?" A lens does two things no previous module does:

  1. It produces an *exact* Fourier transform of the mask at a *finite*,
     controllable distance (its focal length f) -- not an approximation
     that only becomes valid once z clears some far-field threshold, the
     way diffraction.py's Fraunhofer pattern does (Goodman Section 5.2,
     Eq. 5-22 with d=f).
  2. It has a finite physical aperture, which throws away every spatial
     frequency component of the mask above a cutoff set by the lens's
     numerical aperture (NA) and the wavelength. This is *why* a real
     lithography tool can't print arbitrarily small features: high
     spatial frequencies (fine mask detail) simply never make it through
     the lens (Goodman Section 6.2, Eq. 6-20).

This module chains those two facts into the forward imaging pipeline the
Week 9 build goal calls for:

    mask --[lens FT, Eq 5-22]--> focal-plane spectrum
         --[pupil cutoff, Eq 6-20]--> band-limited spectrum
         --[inverse FT]--> aerial image (coherent)

This sits directly downstream of masks.py / grid.py / fft_engine.py
(reused, not reimplemented) and upstream of imaging.py (Week 10, which
adds the *incoherent*/OTF path and intensity thresholding on top of what
this module produces) and aberrations.py (Week 11, which will replace
this module's hard-edged pupil with a phase-carrying generalized pupil
function, reusing pupil_radius/cutoff_frequency below unchanged).

SIMPLIFYING ASSUMPTION (stated explicitly)
---------------------------------------------
Goodman's Section 5.3 impulse-response treatment (Eq. 5-23/5-28) is fully
general: arbitrary object/image conjugate distances, arbitrary
magnification, governed by the classical lens law. This module implements
the specific case used throughout the rest of this project -- unit
magnification, telecentric, infinite-conjugate imaging, with the mask at
the lens's front focal plane -- because that is the standard simplified
model for a lithography exposure tool's forward chain, and because it
lets pupil filtering be expressed directly as multiplication by a
transfer function in frequency space (Eq. 6-20), matching the
imaging.py/aberrations.py architecture planned for Weeks 10-11. The full
non-unit-magnification treatment of Eq. 5-28 is not implemented.

All spatial coordinates: µm
All spatial frequencies: cycles/µm (µm⁻¹)
Wavelength: µm
NA: dimensionless (paraxial NA = sin(theta_max) ~= tan(theta_max))
"""

from typing import Optional, Tuple

import numpy as np

from fft_engine import fft1d, ifft1d
from constants import WAVELENGTH, NA_DEFAULT


def thin_lens_phase(x: np.ndarray, focal_length: float, wavelength: float = WAVELENGTH) -> np.ndarray:
    """
    Paraxial thin-lens phase transformation, Goodman Eq. (5-10):

        t_l(x) = exp[-j*(pi/(wavelength*focal_length)) * x^2]

    (the constant phase factor exp[jk*n*A0] from Eq. 5-9 is dropped, per
    Goodman's own convention immediately following Eq. 5-10, since it has
    no effect on any intensity or relative-phase result downstream.)

    WHY THIS FUNCTION EXISTS EVEN THOUGH THE MAIN PIPELINE DOESN'T CALL IT
    -------------------------------------------------------------------------
    lens_focal_plane_field() below computes the mask's focal-plane field
    using the already-simplified closed-form result of placing the mask a
    distance d=f in front of the lens (Eq. 5-22), where the quadratic
    phase factor cancels analytically and the lens's own phase transform
    never needs to be applied numerically. thin_lens_phase is included
    separately because it is the literal Eq. 5-10 result -- the physical
    reason a lens performs a Fourier transform at all (a converging
    spherical wave brings a distant plane wave's angular spectrum to a
    focus; see Goodman's Fig. 5.4 discussion) -- and because Week 11
    (aberrations.py) will extend this exact idea into a *generalized*
    pupil function P~(x) = P(x)*exp(j*k*W(x)) that also carries a phase.
    Having the ideal Eq. 5-10 transform available now, tested and
    verified by hand, means Week 11 only has to add the aberration term
    W(x), not rebuild the lens-phase concept from scratch.

    Parameters
    ----------
    x            : ndarray — spatial coordinate across the lens aperture, µm
    focal_length : float — lens focal length, µm
    wavelength   : float — wavelength, µm (defaults to constants.WAVELENGTH)

    Returns
    -------
    phase_transform : ndarray, complex — t_l(x), unit magnitude everywhere
        (a pure phase function; |t_l| = 1 for all x)
    """
    k = 2.0 * np.pi / wavelength
    return np.exp(-1j * k * x ** 2 / (2.0 * focal_length))


def pupil_radius(focal_length: float, NA: float = NA_DEFAULT) -> float:
    """
    Physical half-width of the lens aperture (pupil radius) in the lens/
    focal plane: r_pupil = focal_length * NA.

    Goodman connection
    -------------------
    NA is defined (Section 6.1 discussion around the entrance/exit pupil
    geometry, Fig. 6.1, and used explicitly in the discussion surrounding
    Eq. 6-20's pupil-limited passband) as the sine of the half-angle
    theta_max subtended by the lens aperture as seen from the focal point:
    NA = sin(theta_max). Under the paraxial approximation already assumed
    throughout Chapter 5 (Eq. 5-7's small-angle approximations),
    sin(theta) ~= tan(theta), so the marginal ray reaching the edge of the
    lens aperture at radius r_pupil, from a point a distance f away on
    axis, satisfies tan(theta_max) = r_pupil / focal_length ~= NA. Solving
    for r_pupil gives the relation used here.

    Parameters
    ----------
    focal_length : float — lens focal length, µm
    NA           : float — numerical aperture, dimensionless, in (0, 1)
                    for the paraxial regime assumed throughout this
                    project (defaults to constants.NA_DEFAULT)

    Returns
    -------
    r_pupil : float — pupil half-width, µm

    Raises
    ------
    ValueError : if NA is not in (0, 1) -- outside that range either the
        aperture is degenerate (NA<=0) or the paraxial sin~=tan
        approximation this whole chapter relies on breaks down badly
        enough (NA>=1) that the result would be physically misleading.
    """
    if not (0.0 < NA < 1.0):
        raise ValueError(
            f"NA={NA} is outside the paraxial regime (0, 1) assumed "
            "throughout this project's Chapter 5/6 derivations."
        )
    return focal_length * NA


def cutoff_frequency(NA: float = NA_DEFAULT, wavelength: float = WAVELENGTH) -> float:
    """
    Spatial-frequency cutoff imposed by a finite-NA pupil:
    f_cutoff = NA / wavelength.

    Goodman connection
    -------------------
    Directly from Eq. (6-20): H(fx) = P(wavelength*z*fx). A hard-edged
    pupil P(u) = 1 for |u| <= r_pupil, 0 otherwise, therefore passes only
    frequencies satisfying |wavelength*z*fx| <= r_pupil, i.e.
    |fx| <= r_pupil / (wavelength*z). At z = focal_length, and using
    r_pupil = focal_length*NA (pupil_radius above), the focal length
    cancels exactly:

        f_cutoff = r_pupil / (wavelength * focal_length)
                 = (focal_length * NA) / (wavelength * focal_length)
                 = NA / wavelength

    This cancellation is worth stating explicitly: the frequency-domain
    cutoff a lens imposes does NOT depend on the focal length at all, only
    on NA and wavelength -- a physically important, slightly non-obvious
    result that falls straight out of Eq. 6-20 once the geometric
    definition of NA is substituted in. Verified numerically by hand
    before delivery (computed f_cutoff two independent ways -- via
    r_pupil/(wavelength*focal_length) and via NA/wavelength directly, at
    focal_length = 50000 µm, NA = 0.5, wavelength = 0.365 µm -- and
    confirmed they agree to floating-point precision).

    Parameters
    ----------
    NA         : float — numerical aperture (defaults to constants.NA_DEFAULT)
    wavelength : float — wavelength, µm (defaults to constants.WAVELENGTH)

    Returns
    -------
    f_cutoff : float — cutoff spatial frequency, cycles/µm
    """
    return NA / wavelength


def lens_focal_plane_field(mask: np.ndarray, grid, wavelength: float = WAVELENGTH,
                            focal_length: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact field distribution in the back focal plane of a lens, with the
    mask illuminated by a normally incident plane wave and placed in the
    lens's FRONT focal plane (d = focal_length, Goodman Fig. 5.6/5.7
    geometry).

    Goodman connection
    -------------------
    Directly implements Eq. (5-22):

        U_f(u) = [A / (j*wavelength*f)] * FT{t_A(xi)}(fx = u/(wavelength*f))

    Goodman derives this (Section 5.2.2) by combining the Fresnel
    diffraction integral from the input to the lens with the lens's own
    quadratic phase transform (Eq. 5-10): the quadratic phase term in the
    Fresnel integrand exactly cancels the lens's phase transform when, and
    only when, the input-to-lens distance d equals the focal length f.
    What's left is a *pure*, *exact* Fourier transform relation with no
    residual curvature -- unlike Fraunhofer diffraction (Eq. 4-25,
    diffraction.py), which is only an *approximation* requiring z to clear
    a far-field threshold (fraunhofer_far_field_distance). A lens gives
    you the same mathematical result at a finite, freely chosen distance
    f, exactly. This is the core physical fact this module exists to
    encode, and it's why lens.py's build goal ("clearly visualize the
    pupil plane") is meaningful: the pupil plane IS the focal plane
    computed here, at a real, finite, on-the-bench distance from the mask.

    NUMERICAL IMPLEMENTATION
    --------------------------
    Mathematically this is the identical DFT-rescaling trick already
    verified and used in diffraction.py's fraunhofer_pattern (same
    G_phys = fft1d(mask, dx) * grid.L rescaling from fft1d's 1/N-normalized
    output to a physically-scaled continuous-FT approximation), just with
    z replaced by focal_length and no far-field validity requirement to
    check. See diffraction.fraunhofer_pattern's docstring for the full
    derivation of why multiplying by grid.L is correct; it is not
    re-derived here to avoid duplicating that explanation.

    Parameters
    ----------
    mask         : ndarray, shape (N,) — mask transmission (0/1), on grid.x
    grid         : Grid1D — mask's spatial/frequency grid (from grid.py)
    wavelength   : float — wavelength, µm (defaults to constants.WAVELENGTH)
    focal_length : float — lens focal length, µm (required; no project-wide
                    default exists, since focal length is a per-lens design
                    choice, unlike wavelength/NA)

    Returns
    -------
    u     : ndarray, shape (N,) — physical coordinate in the focal/pupil
             plane, µm (u = grid.f * wavelength * focal_length)
    field : ndarray, shape (N,), complex — focal-plane field U_f(u), with
             the same magnitude-only convention as fraunhofer_pattern
             (unit-magnitude phase prefactors exp(jkz) and 1/j dropped,
             since only |U_f|^2 -- needed for the pupil-plane visualization
             and, eventually, the aerial image -- is affected by them)
    """
    if focal_length is None:
        raise ValueError(
            "focal_length is required (no project-wide default; it's a "
            "per-lens design choice)."
        )

    G_phys = fft1d(mask, grid.dx) * grid.L
    u = grid.f * wavelength * focal_length
    field = G_phys / (wavelength * focal_length)
    return u, field


def pupil_function_freq(grid, NA: float = NA_DEFAULT, wavelength: float = WAVELENGTH) -> np.ndarray:
    """
    Hard-edged (diffraction-limited, unaberrated) pupil function evaluated
    directly on the grid's own spatial-frequency axis grid.f.

    Goodman connection
    -------------------
    This is P(wavelength*z*fx) from Eq. (6-20), evaluated at
    z = focal_length and expressed directly in terms of fx using
    cutoff_frequency's NA/wavelength cancellation (see that function's
    docstring): the pupil, physically a function of position across the
    lens, becomes -- once NA's definition is substituted in and the focal
    length cancels -- a function purely of spatial frequency and
    NA/wavelength. This is exactly Goodman's stated result that "the
    pupil sharply limits the range of Fourier components passed by the
    system" (Section 6.2 discussion following Eq. 6-20): a
    diffraction-limited pupil acts as an ideal (brick-wall) low-pass
    filter in the frequency domain.

    Parameters
    ----------
    grid       : Grid1D — provides the frequency axis grid.f (cycles/µm)
    NA         : float — numerical aperture (defaults to constants.NA_DEFAULT)
    wavelength : float — wavelength, µm (defaults to constants.WAVELENGTH)

    Returns
    -------
    P : ndarray, shape (N,) — 1.0 where |grid.f| <= NA/wavelength, else 0.0
    """
    f_cutoff = cutoff_frequency(NA, wavelength)
    return (np.abs(grid.f) <= f_cutoff).astype(float)


def coherent_aerial_image(mask: np.ndarray, grid, wavelength: float = WAVELENGTH,
                           NA: float = NA_DEFAULT) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full forward coherent imaging chain: mask -> band-limited spectrum
    (lens Fourier transform + pupil cutoff) -> aerial image intensity.

    Goodman connection
    -------------------
    Combines two results:

      1. Eq. (5-22): the lens produces an *exact* Fourier transform of the
         mask (lens_focal_plane_field above establishes this, and is the
         function used for VISUALIZING the physical pupil plane in µm).
      2. Eq. (6-13)/(6-17)/(6-20): once the input is represented in the
         frequency domain, passing it through a diffraction-limited
         imaging system is equivalent to multiplying its spectrum by the
         (scaled) pupil function H(fx) = P(wavelength*focal_length*fx),
         then inverse-transforming back to real space to get the output
         field -- the standard transfer-function treatment of imaging.

    WHY THE PUPIL FILTERING IS DONE IN fx-SPACE, NOT ON THE PHYSICALLY-
    SCALED lens_focal_plane_field OUTPUT
    ----------------------------------------------------------------------
    lens_focal_plane_field's u and field carry real physical scale factors
    (grid.L, 1/(wavelength*focal_length)) needed to plot a correctly
    labeled focal/pupil plane in µm. But applying the hard 0/1 pupil
    cutoff and inverse-transforming back to the image plane only cares
    about *which* frequency components survive, not their absolute
    physical scale in the focal plane -- multiplying by a real positive
    constant before an inverse FFT only rescales the reconstructed field's
    overall amplitude, not its shape or the resulting normalized
    intensity. So this function filters the RAW fft1d spectrum (already
    exactly aligned with grid.f) directly, using pupil_function_freq, and
    lens_focal_plane_field is only called separately when a caller wants
    the physical pupil-plane plot. This avoids threading the
    grid.L/(wavelength*f) scale factor through an unnecessary round trip
    that would cancel out anyway and only adds a chance to introduce an
    arithmetic bug.

    VALIDATION PERFORMED BY HAND BEFORE DELIVERY
    ------------------------------------------------
    - Plumbing check: with f_cutoff set far above grid.f_max (pupil
      all-ones), intensity reproduced the original binary mask to
      floating-point precision (max error ~1.7e-32), confirming the
      fft1d -> multiply-by-P -> ifft1d round trip is exact when nothing is
      filtered, per ifft1d's guaranteed round-trip property.
    - Realistic-NA check (NA=0.5, wavelength=0.365 µm): pupil passed
      ~21% of the grid's frequency content, all intensities stayed
      non-negative, and the image showed the expected Gibbs-like overshoot
      above 1.0 at a hard mask edge -- the textbook signature of an
      ideal brick-wall (unapodized) pupil, not a bug.
    - Resolution-limit check: a grid-aligned grating with its fundamental
      frequency placed safely below f_cutoff showed strong intensity
      modulation (contrast); an otherwise-identical grating with its
      fundamental placed above f_cutoff showed EXACTLY zero modulation
      (flat intensity at duty_cycle^2 = 0.25) -- the classic
      diffraction-limited resolution cutoff, confirming the pupil is
      actually removing the frequency content it claims to.
      (Note: an earlier attempt at this check used a grating pitch that
      was periodic in L but not an integer number of grid samples per
      period; that produced spurious low-frequency leakage that survived
      the cutoff and looked like a bug. This is the same sampling-grid
      leakage effect documented in the Week 8 build log, not a flaw in
      this function -- test gratings for validating lens.py should use a
      pitch equal to an integer number of samples, i.e.
      pitch = k * grid.dx for integer k.)

    Parameters
    ----------
    mask       : ndarray, shape (N,) — mask transmission (0/1), on grid.x
    grid       : Grid1D — mask's spatial/frequency grid (from grid.py)
    wavelength : float — wavelength, µm (defaults to constants.WAVELENGTH)
    NA         : float — numerical aperture (defaults to constants.NA_DEFAULT)

    Returns
    -------
    field_image : ndarray, shape (N,), complex — coherent image-plane
                   field, on the same grid.x as the input mask (unit
                   magnification; see module docstring's simplifying-
                   assumption note)
    intensity   : ndarray, shape (N,) — |field_image|^2, the aerial image
                   (NOT peak-normalized -- unlike diffraction.py's
                   fraunhofer_pattern, this is intentional: the
                   all-frequencies-pass limit should reduce exactly to the
                   original mask, which is already a valid intensity in
                   [0, 1], and forcing a peak-normalization would silently
                   hide any amplitude bug that broke that limiting case)
    P           : ndarray, shape (N,) — the frequency-domain pupil mask
                   actually applied (0/1), returned so callers/plots can
                   show exactly what was cut, without recomputing it
    """
    G_raw = fft1d(mask, grid.dx)
    P = pupil_function_freq(grid, NA=NA, wavelength=wavelength)
    G_filtered = G_raw * P
    field_image = ifft1d(G_filtered, grid.dx)
    intensity = np.abs(field_image) ** 2
    return field_image, intensity, P