"""
physics/propagation.py
-----------------------
Angular spectrum propagator: propagates a 1D field a distance z along the
optical axis, separating propagating plane-wave components from
evanescent ones.

WHY THIS MODULE EXISTS IN THE PIPELINE
---------------------------------------
masks.py builds the field right at the mask plane (z = 0). fft_engine.py
and grid.py give us the machinery to go in and out of the frequency
domain with correctly labeled physical frequencies. Neither of those
modules tells us what the field looks like anywhere *else* along z — that
is the missing physical step this module fills in, and it is the first
piece of the simulator that sits between "mask" and "lens": light leaves
the mask, diffracts, and spreads as it travels before it ever reaches the
lens's entrance pupil. Every later stage (lens.py's Fourier transform,
the pupil filtering, the aerial image) implicitly assumes a propagated
field is available to feed into it; this module is what produces that
field.

This is also the first module to actually *consume* Grid1D and
fft_engine's fft1d/ifft1d as core dependencies rather than as standalone
utilities, per the Week 2 carryover note.

PHYSICS (Goodman §3.10.1-3.10.4)
---------------------------------
Decompose the field at z=0 into plane waves via its Fourier spectrum
A(f; 0) (Eq. 3-58 in 2D; here 1D in f = fx). Each spectral component at
spatial frequency f is a single plane wave traveling at a fixed angle.
Propagating that plane wave by a distance z changes only its phase (Eq.
3-66), provided its direction cosine satisfies the physical constraint

    alpha^2 + beta^2 < 1    (Eq. 3-67, here just (lambda*f)^2 < 1)

Components that violate this inequality are not real propagation angles
at all -- the "direction cosine" becomes imaginary, and Eq. (3-66)'s phase
term becomes a real, rapidly decaying exponential instead of an
oscillating one. These are the evanescent waves: they carry no energy
away from the aperture and are negligible after a few wavelengths of
travel (Goodman's footnote 7, §3.10.2). Eq. (3-69) packages the cutoff
explicitly using a circ() function limiting which frequencies are even
allowed to propagate; §3.10.4 then identifies the whole z-propagation
operation as a linear, space-invariant filter with transfer function

            { exp[ j2*pi*z*sqrt(1/lambda^2 - f^2) ],   f^2 < 1/lambda^2
    H(f;z) = {
            { 0   (or exponentially decaying),          f^2 >= 1/lambda^2

(this module keeps the evanescent branch as a decaying real exponential
rather than hard-zeroing it, so that the decay itself -- not just a
binary cutoff -- is visible in the propagated spectrum, matching Eq.
3-66's actual imaginary-square-root behavior rather than Eq. 3-69's
circ-function simplification of it).

All spatial coordinates: µm. All spatial frequencies: cycles/µm (µm⁻¹).
Wavelength: µm.
"""

import numpy as np

from fft_engine import fft1d, ifft1d, freq_axis


def transfer_function(f: np.ndarray, wavelength: float, z: float):
    """
    Compute the angular-spectrum propagation transfer function H(f; z)
    and a boolean mask of which frequency components are propagating
    (as opposed to evanescent).

    Goodman connection
    -------------------
    Direct implementation of the propagation phase term in Eq. (3-66),
    extended into the evanescent regime per the discussion following Eq.
    (3-67)-(3-69) and the transfer function of §3.10.4. Define

        kz(f) = sqrt(1/lambda^2 - f^2)

    For f^2 < 1/lambda^2, kz is real and positive: this is the
    z-component of the wave vector (in cycles/µm) for the plane wave at
    spatial frequency f, and propagation over distance z simply
    accumulates phase j*2*pi*z*kz(f), exactly Eq. (3-66) with no
    amplitude change -- consistent with the handwritten note "these plane
    waves don't change shape or strength as they propagate, they just
    accumulate phase."

    For f^2 >= 1/lambda^2, kz is imaginary: write kz = j*kappa with
    kappa = sqrt(f^2 - 1/lambda^2) real and positive. Then
    exp(j*2*pi*z*kz) = exp(j*2*pi*z*j*kappa) = exp(-2*pi*z*kappa), a real,
    monotonically decaying exponential in z -- the evanescent decay
    Goodman describes as "rapidly attenuated by the propagation
    phenomenon" (§3.10.2) and "quite analogous to the waves produced in a
    microwave waveguide driven below its cutoff frequency."

    Parameters
    ----------
    f          : ndarray — spatial frequencies (cycles/µm), e.g. from
                  fft_engine.freq_axis or Grid1D.f
    wavelength : float — wavelength in µm
    z          : float — propagation distance in µm (z=0 returns H == 1
                  everywhere)

    Returns
    -------
    H              : ndarray, complex, same shape as f
                      Propagation transfer function.
    is_propagating : ndarray, bool, same shape as f
                      True where f^2 < 1/lambda^2 (real direction cosine,
                      i.e. a physically propagating plane wave); False
                      where the component is evanescent.
    """
    f = np.asarray(f, dtype=float)
    cutoff_sq = 1.0 / wavelength**2
    is_propagating = f**2 < cutoff_sq

    H = np.empty_like(f, dtype=complex)

    # Propagating branch: real kz, pure phase (Eq. 3-66)
    kz_prop = np.sqrt(np.clip(cutoff_sq - f**2, 0.0, None))
    H_prop = np.exp(1j * 2.0 * np.pi * z * kz_prop)

    # Evanescent branch: imaginary kz -> real decaying exponential
    kappa = np.sqrt(np.clip(f**2 - cutoff_sq, 0.0, None))
    H_evan = np.exp(-2.0 * np.pi * np.abs(z) * kappa).astype(complex)

    H = np.where(is_propagating, H_prop, H_evan)
    return H, is_propagating


def propagate_angular_spectrum(
    field: np.ndarray, dx: float, wavelength: float, z: float
) -> np.ndarray:
    """
    Propagate a 1D complex field a distance z using the angular spectrum
    method.

    Goodman connection
    -------------------
    Implements the full angular-spectrum prescription of §3.10.2:
    Fourier transform the field at z=0 to get its angular spectrum
    A(f; 0) (Eq. 3-58), multiply by the propagation transfer function
    H(f; z) (Eq. 3-66/3-70) to get A(f; z), then inverse transform to
    recover the field U(x; z) (Eq. 3-69). This is exactly the "each
    component of A is an individual plane wave travelling at a fixed
    angle; they don't change shape or strength, they just accumulate
    phase" picture from the handwritten notes, summed back together via
    the inverse FFT.

    Pipeline connection
    --------------------
    Input: the mask transmission field at z=0 (physics/masks.py).
    Output: the field some distance z downstream, still in 1D, still
    needing the lens (physics/lens.py, next week) to apply its own
    Fourier transform and finite-aperture (pupil) filtering. This module
    does not model any lens -- it is free-space propagation only.

    Parameters
    ----------
    field      : ndarray, shape (N,) — complex (or real) field at z=0
    dx         : float — spatial sample spacing in µm (must match the
                  grid the field was defined on, e.g. Grid1D.dx)
    wavelength : float — wavelength in µm
    z          : float — propagation distance in µm

    Returns
    -------
    propagated : ndarray, shape (N,), complex
        Field at distance z. Use np.abs(propagated)**2 for intensity.
    """
    N = len(field)
    f = freq_axis(N, dx)
    H, _ = transfer_function(f, wavelength, z)

    spectrum0 = fft1d(field, dx)
    spectrum_z = spectrum0 * H
    propagated = ifft1d(spectrum_z, dx)
    return propagated


def propagated_spectrum(
    field: np.ndarray, dx: float, wavelength: float, z: float
):
    """
    Return the angular spectrum of `field` after propagating to distance
    z, along with the frequency axis and the propagating/evanescent
    split -- everything needed to plot "spectrum content evolving with
    propagation distance" without re-deriving H by hand.

    This is a thin convenience wrapper: it does the same FFT + H
    multiply as propagate_angular_spectrum, but returns the frequency-
    domain result (and the propagating mask) instead of inverse
    transforming back to space, since the build goal explicitly asks to
    visualize spectrum evolution, not just the propagated field.

    Parameters
    ----------
    field      : ndarray, shape (N,) — field at z=0
    dx         : float — spatial sample spacing in µm
    wavelength : float — wavelength in µm
    z          : float — propagation distance in µm

    Returns
    -------
    f              : ndarray, shape (N,) — frequency axis (cycles/µm)
    spectrum_z     : ndarray, shape (N,), complex — A(f; z)
    is_propagating : ndarray, shape (N,), bool — True where the
                      component is propagating (not evanescent) at this
                      wavelength
    """
    N = len(field)
    f = freq_axis(N, dx)
    H, is_propagating = transfer_function(f, wavelength, z)
    spectrum0 = fft1d(field, dx)
    spectrum_z = spectrum0 * H
    return f, spectrum_z, is_propagating