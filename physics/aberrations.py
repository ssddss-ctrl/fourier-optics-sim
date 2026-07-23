"""
physics/aberrations.py
-------------------------
Defocus wavefront error and the generalized (phase-carrying) pupil function
that lens.py's hard-edged pupil is a special case of.

WHY THIS MODULE EXISTS IN THE PIPELINE
-----------------------------------------
Every pupil used through Week 10 (lens.pupil_function_freq) is real-valued:
1 inside the NA cutoff, 0 outside, with no phase term at all. That's the
"perfect, unaberrated lens" idealization Goodman uses to derive the basic
ATF/OTF machinery. Real lenses -- and, just as commonly in lithography,
mask patterns imaged at the wrong focal plane -- pick up a phase error
across the pupil that this idealization can't represent.

This module adds exactly that phase term:

    1. focus_error_wavefront computes W(fx), the wavefront aberration
       function for pure defocus, expressed directly on the frequency
       axis (rather than the pupil-plane coordinate Goodman's general
       aberration theory uses), by reusing lens.cutoff_frequency's
       existing NA/wavelength -> frequency mapping.
    2. generalized_pupil_function multiplies any real pupil (in practice,
       lens.pupil_function_freq's output) by exp(j*k*W) to produce a
       complex pupil that carries both amplitude support (still 0/1,
       unchanged) and phase error.

Nothing here replaces lens.pupil_function_freq or imaging.py's ATF/OTF
construction -- both are reused unchanged. Instead, imaging.py's
amplitude_point_spread_function is extended (Week 11 change, not a new
module) to optionally pass its real pupil through
generalized_pupil_function before the inverse transform, so the existing
OTF-from-ATF machinery (Eq. 6-25/6-28, valid "for systems both with and
without aberrations" per Goodman) picks up the aberration for free.

    mask --[pupil, Eq 6-20]--> real ATF --[this module, defocus phase]-->
        generalized (complex) ATF --[already in imaging.py]--> OTF, images

GOODMAN CONNECTION
---------------------
- generalized_pupil_function implements the generalized pupil function
  relation (build notes Eq. 14): P_G(fx) = P(fx) * exp(j*k*W(fx)), with
  k = 2*pi/wavelength -- the standard way Goodman's Ch. 6 aberration
  treatment folds a wavefront error into the pupil once the aberration-free
  pupil P(fx) is already in hand.
- focus_error_wavefront implements the defocus wavefront error (build notes
  Eq. 16), the classical quadratic aberration term
  W(rho) = Wm * (rho/rho_max)^2, reparameterized directly in terms of
  spatial frequency: since the pupil-plane radius rho maps linearly onto
  frequency (rho/rho_max = fx/f_cutoff, the same substitution
  lens.cutoff_frequency already uses to collapse Eq. 6-20 onto the
  frequency axis), the quadratic form carries over unchanged with
  rho_max -> f_cutoff. Wm = defocus_waves * wavelength is the peak
  wavefront error, expressed in "waves" (multiples of wavelength) --
  the conventional unit for specifying defocus in optical design.

SIMPLIFYING ASSUMPTION (stated explicitly)
---------------------------------------------
Only pure defocus (the quadratic term) is implemented here. Goodman's
general aberration theory (Ch. 6.4, e.g. the Seidel/Zernike expansions)
includes coma, astigmatism, spherical aberration, etc. -- none of those are
built here; generalized_pupil_function itself is aberration-agnostic (it
just exponentiates whatever W it's given), so adding another aberration
type later means writing a new W(fx) function, not touching this one.

All spatial frequencies: cycles/µm (µm⁻¹)
Wavelength: µm
defocus_waves: dimensionless (peak wavefront error in units of wavelength)
W: µm (wavefront error has units of optical path length)
"""

import numpy as np

from constants import WAVELENGTH, NA_DEFAULT
from lens import cutoff_frequency


def focus_error_wavefront(grid, defocus_waves: float = 0.0, NA: float = NA_DEFAULT,
                           wavelength: float = WAVELENGTH) -> np.ndarray:
    """
    Quadratic (pure defocus) wavefront error W(fx), evaluated on the grid's
    own frequency axis grid.f.

    Goodman connection
    -------------------
    Defocus wavefront error (build notes Eq. 16): W(rho) = Wm * (rho /
    rho_max)^2, where rho is radial position across the pupil and rho_max
    is the pupil edge. Substituting rho/rho_max = fx/f_cutoff -- the same
    linear pupil-coordinate/frequency correspondence lens.cutoff_frequency
    already relies on to express Eq. 6-20's cutoff purely in terms of
    NA/wavelength -- gives W directly as a function of spatial frequency:

        W(fx) = Wm * (fx / f_cutoff)^2,   Wm = defocus_waves * wavelength

    Wm is the peak wavefront error at the pupil edge (fx = +/- f_cutoff),
    expressed in "waves" via defocus_waves (defocus_waves=1.0 means a full
    wave, Wm = wavelength, of peak error).

    Parameters
    ----------
    grid           : Grid1D — provides the frequency axis grid.f (cycles/µm)
    defocus_waves  : float — peak defocus error, in units of wavelength
                     (0.0 = no aberration)
    NA             : float — numerical aperture (defaults to constants.NA_DEFAULT)
    wavelength     : float — wavelength, µm (defaults to constants.WAVELENGTH)

    Returns
    -------
    W : ndarray, shape (N,) — wavefront error on grid.f, µm
    """
    f_cutoff = cutoff_frequency(NA, wavelength)
    Wm = defocus_waves * wavelength
    return Wm * (grid.f / f_cutoff) ** 2


def generalized_pupil_function(pupil: np.ndarray, W: np.ndarray,
                                wavelength: float = WAVELENGTH) -> np.ndarray:
    """
    Generalized (phase-carrying) pupil function: an aberration-free pupil
    multiplied by the phase error exp(j*k*W) that a wavefront error W
    imposes.

    Goodman connection
    -------------------
    Generalized pupil function (build notes Eq. 14):

        P_G(fx) = P(fx) * exp(j*k*W(fx)),   k = 2*pi/wavelength

    where P(fx) is the aberration-free pupil (in practice,
    lens.pupil_function_freq's 0/1 output) and W(fx) is any wavefront
    error function (in practice, focus_error_wavefront's output, but this
    function does not assume defocus specifically -- any W of the right
    shape works). When W is identically zero, exp(j*k*0) = 1 and P_G
    reduces exactly to P, so this is a strict generalization of the
    existing hard-edged pupil, not a replacement for it.

    Parameters
    ----------
    pupil      : ndarray, shape (N,) — aberration-free pupil (real, 0/1)
    W          : ndarray, shape (N,) — wavefront error, µm
    wavelength : float — wavelength, µm (defaults to constants.WAVELENGTH)

    Returns
    -------
    P_G : ndarray, shape (N,), complex — generalized pupil. |P_G| == pupil
        exactly (phase-only perturbation) since |exp(j*k*W)| == 1
        everywhere.
    """
    k = 2 * np.pi / wavelength
    return pupil * np.exp(1j * k * W)
