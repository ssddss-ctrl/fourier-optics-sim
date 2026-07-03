"""
physics/diffraction.py
-----------------------
Fraunhofer (far-field) diffraction simulator for arbitrary 1D apertures,
plus closed-form analytic predictions used to validate the numerical
simulator.

WHY THIS MODULE EXISTS IN THE PIPELINE
----------------------------------------
Up to this point the pipeline has: built an aperture (masks.py), given it
a physical grid (grid.py), and propagated it a finite distance using the
angular spectrum method (propagation.py, exact at any z). This module adds
the *far-field limit* of that same propagation problem — the regime where
z is large enough that the quadratic phase term in the Fresnel integral
becomes negligible (Goodman Eq. 4-24) and the diffracted field collapses
to a single scaled Fourier transform of the aperture (Eq. 4-25), rather
than the fuller Fresnel integral. This is *not* redundant with
propagation.py: propagation.py is the general angular-spectrum engine
valid at any z (near or far field); this module is the closed-form,
much cheaper Fraunhofer limit, valid only once z clears the far-field
distance, and it is what lets single-slit sinc² and grating-order
predictions (Eq. 4-26/4-27, 4-34/4-36) be checked directly against
textbook closed forms — something the general angular-spectrum machinery
doesn't give you for free at intermediate z.

Physically: this is "what pattern lands on a screen far downstream of the
mask." It sits conceptually parallel to propagation.py (a specialization
of the same physics), and both are inputs the eventual lens module will
draw on, since a lens placed after the aperture produces its Fraunhofer
pattern at a *finite* distance (the focal length) rather than requiring
z to physically satisfy Eq. 4-24 (Goodman §5.2) — Week 9 territory, not
implemented here.

All spatial coordinates: µm
All spatial frequencies: cycles/µm (µm⁻¹)
Wavelength: µm
Intensity: normalized so the pattern peak equals 1 (project convention)
"""

import numpy as np

from fft_engine import fft1d


def fraunhofer_far_field_distance(aperture_width: float, wavelength: float) -> float:
    """
    Minimum propagation distance z for the Fraunhofer approximation to be
    valid, using the "antenna designer's formula" (Goodman §4.3, discussed
    immediately following Eq. 4-24): z > 2*D^2/lambda, where D is the full
    linear extent of the aperture. This is the *looser* of the two
    conditions Goodman gives (the stricter one uses z >> rather than z >,
    and a smaller numerical prefactor) — it's the standard practical
    threshold used in the text's own worked example (D=2.5cm aperture,
    z > ~2000 m at 0.6 µm).

    Parameters
    ----------
    aperture_width : float — full linear extent of the aperture, µm
    wavelength     : float — wavelength, µm

    Returns
    -------
    z_min : float — minimum propagation distance (µm) for Fraunhofer
        validity per the antenna designer's formula

    Notes
    -----
    Explicit float() cast on return: aperture_width**2 / wavelength
    produces a numpy.floating if either input happens to be a numpy
    scalar (e.g. pulled from an array), not a plain Python float, even
    though the annotation says float. Same numpy-scalar leak Grid1D hit
    in Week 2 (self.dx cast) — fixed the same way, at the source, so it
    can't propagate into later comparisons.
    """
    return float(2.0 * aperture_width ** 2 / wavelength)


def check_fraunhofer_validity(aperture_width: float, wavelength: float, z: float) -> bool:
    """
    True if z clears the far-field distance from fraunhofer_far_field_distance
    for the given aperture width and wavelength (Goodman Eq. 4-24 region).
    Every pattern-generation call in scripts/ should check this before
    trusting the output, since fraunhofer_pattern below will happily
    compute a (physically invalid) answer at any z.

    Notes
    -----
    Explicit bool() cast for the same reason fraunhofer_far_field_distance
    casts to float: a comparison against a numpy scalar can return
    numpy.bool_ rather than Python's bool, which is fine for `if` checks
    but can behave unexpectedly if the caller ever compares the result
    with `is True`/`is False` (the failure mode Week 2's test suite hit).
    """
    return bool(z > fraunhofer_far_field_distance(aperture_width, wavelength))


def fraunhofer_pattern(mask: np.ndarray, grid, wavelength: float, z: float):
    """
    Compute the Fraunhofer (far-field) diffraction pattern of an arbitrary
    1D aperture, directly implementing Goodman Eq. (4-25):

        U(x) = [1/(j*lambda*z)] * exp(jkz) * exp[j(k/2z)x^2]
               * FT{t(xi)}(fx = x/(lambda*z))

    Since only the intensity |U(x)|^2 is of interest (per Eq. 4-26's own
    reduction), the unit-magnitude phase prefactors (exp(jkz), the
    quadratic phase, and 1/j) all drop out of |U|, so this function
    returns the field magnitude/intensity without carrying them.

    HOW THIS WORKS NUMERICALLY (arbitrary aperture, not just analytic
    special cases)
    ------------------------------------------------------------------
    fft_engine.fft1d already returns the DFT normalized by 1/N (DC = mean
    of the signal), following the Week 1/2 convention. The *continuous*
    Fourier transform in Eq. 4-25 is instead approximated by the DFT times
    the sample spacing: G(f) ~ dx * sum_n mask[n] exp(-j2*pi*f*x_n)
                              = dx * N * fft1d(mask, dx)
                              = grid.L * fft1d(mask, dx)
    (dx*N = L exactly, by construction of Grid1D/make_grid). Multiplying
    by grid.L converts fft1d's dimensionless-per-N output back into a
    physically-scaled spectrum with units of length (µm), matching e.g.
    the analytic single-slit spectrum width*sinc(width*fx), which also has
    units of µm. This rescaling is what makes fraunhofer_pattern work for
    *any* mask array (grating, two lines, or a mask no closed form exists
    for) — not just the special cases in analytic_slit_intensity below.

    The far-field spatial axis then comes directly from Grid1D's existing
    frequency axis: since Eq. 4-25 evaluates the spectrum at
    fx = x/(lambda*z), the *frequency* axis IS the far-field *position*
    axis after the substitution x_obs = f * lambda * z. No interpolation
    is needed — grid.f is just relabeled.

    Parameters
    ----------
    mask       : ndarray, shape (N,) — aperture transmission (0/1), on grid.x
    grid       : Grid1D — the aperture's spatial/frequency grid (from grid.py)
    wavelength : float — wavelength, µm
    z          : float — propagation distance to the observation plane, µm
                 (should satisfy check_fraunhofer_validity for the result
                 to be physically meaningful)

    Returns
    -------
    x_obs     : ndarray, shape (N,) — far-field position axis, µm
                 (x_obs = grid.f * wavelength * z)
    field     : ndarray, shape (N,), complex — field amplitude U(x_obs),
                 with phase prefactors dropped (magnitude-correct, not
                 phase-correct — see note above)
    intensity : ndarray, shape (N,) — |field|^2, normalized so the pattern
                 peak equals 1 (project convention)
    """
    G_phys = fft1d(mask, grid.dx) * grid.L
    x_obs = grid.f * wavelength * z
    field = G_phys / (wavelength * z)
    intensity = np.abs(field) ** 2
    peak = intensity.max()
    if peak > 0:
        intensity = intensity / peak
    return x_obs, field, intensity


def analytic_slit_intensity(x_obs: np.ndarray, width: float, wavelength: float, z: float) -> np.ndarray:
    """
    Closed-form Fraunhofer intensity of a single slit of full width `width`,
    directly from Goodman Eq. (4-26)/(4-27) reduced to 1D:

        I(x) proportional to [width * sinc(width * x / (lambda*z))]^2

    Normalized to peak = 1 (sinc(0) = 1), matching fraunhofer_pattern's
    normalization convention so the two can be compared directly without
    tracking the (width/(lambda*z))^2 prefactor.

    numpy's np.sinc(u) = sin(pi*u)/(pi*u) already matches Goodman's own
    sinc definition (Table 2.1: sinc(x) = sin(pi*x)/(pi*x)), so no
    rescaling of the sinc argument is needed beyond width*x/(lambda*z).

    Parameters
    ----------
    x_obs      : ndarray — far-field position axis, µm
    width      : float — full width of the slit, µm
    wavelength : float — wavelength, µm
    z          : float — propagation distance, µm

    Returns
    -------
    intensity : ndarray — analytic sinc^2 pattern, peak-normalized to 1
    """
    return np.sinc(width * x_obs / (wavelength * z)) ** 2


def analytic_grating_order_positions(pitch: float, wavelength: float, z: float, max_order: int) -> np.ndarray:
    """
    Positions of the diffraction orders of a grating with period `pitch`,
    from Goodman Eq. (4-34)/(4-36): the grating's spatial frequency comb
    sits at f = n/pitch (n = fundamental frequency f0 = 1/pitch and its
    harmonics), so via x_obs = f*lambda*z the orders land at

        x_n = n * lambda * z / pitch,   n = -max_order, ..., max_order

    This is the "grating order spacing" the Week 8 definition of done asks
    to verify: order spacing Delta_x = lambda*z/pitch, independent of duty
    cycle or whether the grating is sinusoidal (Goodman's worked example)
    or square-wave (this project's line_space_grating) — the position of
    the delta-function comb in frequency only depends on the period, per
    the Fourier series of any periodic function.

    Parameters
    ----------
    pitch      : float — grating period, µm
    wavelength : float — wavelength, µm
    z          : float — propagation distance, µm
    max_order  : int — highest order (n) to include, symmetric about 0

    Returns
    -------
    x_orders : ndarray, shape (2*max_order+1,) — far-field positions of
        orders n = -max_order..max_order, µm
    """
    n = np.arange(-max_order, max_order + 1)
    return n * wavelength * z / pitch


def analytic_grating_relative_intensity(order: int, duty_cycle: float) -> float:
    """
    Relative intensity of grating diffraction order `order` compared to the
    zero order, for a square-wave (binary) line-space grating of the given
    duty cycle.

    Goodman's own worked grating example (§4.4.3) is for a *sinusoidal*
    amplitude grating and only gives 3 orders (0, +1, -1) in closed form
    (Eq. 4-37). This project's line_space_grating is a square wave instead,
    which has an infinite Fourier series rather than 3 terms. The general
    result needed here — diffraction efficiency of order k equals |c_k|^2,
    where c_k are the grating's Fourier series coefficients — is stated
    explicitly (not derived in the main text) in Problem 4-12(a): "the
    diffraction efficiency into the kth order of the grating is simply
    v_k = |c_k|^2". Combined with the standard Fourier series coefficient
    of a duty-cycle-d square wave, c_k = d * sinc(k*d), this gives

        I_order / I_0 = |c_order / c_0|^2 = sinc(order * duty_cycle)^2

    (c_0 = d*sinc(0) = d, so the ratio cancels the duty cycle entirely).
    This is the closed-form check used against the numerically simulated
    grating's peak heights at each order position.

    Parameters
    ----------
    order      : int — diffraction order (0, ±1, ±2, ...)
    duty_cycle : float — grating duty cycle (matches masks.line_space_grating)

    Returns
    -------
    relative_intensity : float — I_order / I_0, in [0, 1]

    Notes
    -----
    Explicit float() cast on return: np.sinc(...) ** 2 returns a
    numpy.floating even for scalar inputs, not a plain Python float —
    this is the leak Pylance flagged (reportReturnType). Same fix pattern
    as fraunhofer_far_field_distance above.
    """
    return float(np.sinc(order * duty_cycle) ** 2)