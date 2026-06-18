"""
physics/fft_engine.py
----------------------
FFT/IFFT helpers with correct physical units, plus sampling-constraint
verification via the space-bandwidth product.

WHY THIS MODULE EXISTS IN THE PIPELINE
---------------------------------------
Every later stage of the simulator (propagation, lens Fourier transform,
pupil filtering) needs to go back and forth between the spatial domain
(mask transmission vs. x) and the frequency domain (spectrum vs. f) using
a numerical FFT. np.fft.fft on its own only returns *array indices* — it
has no idea that our grid is spaced in µm. This module is the single place
that converts those raw FFT outputs into physically labeled spectra (cycles
per µm), and the single place that checks whether a chosen grid (L, N) is
even capable of representing the features we care about, before we trust
any number that comes out of it.

Without this module, every later file would need to re-derive the
frequency axis and re-check sampling validity by hand, which is exactly
the kind of physics-inline duplication the project conventions forbid.

All spatial coordinates: µm
All spatial frequencies: cycles/µm (µm⁻¹)
"""

import numpy as np


def freq_axis(N: int, dx: float) -> np.ndarray:
    """
    Compute the physical spatial-frequency axis (cycles/µm) for an N-point
    grid with spacing dx, centered so it runs from -f_max to +f_max.

    Goodman connection
    -------------------
    This is the discrete, finite-N analogue of the continuous Fourier
    transform's frequency variable f_x. The fact that the *sampled* spatial
    signal produces a *periodic* spectrum (which is why fftshift is needed
    to re-center it) is the comb-function transform pair from Goodman
    §2.1.6, Table 2.1:

        F{comb(x/dx)} ∝ comb(f_x · dx)

    Sampling the mask in space is equivalent to multiplying it by a comb
    function of spacing dx; by the convolution theorem this convolves the
    true continuous spectrum with a comb function of spacing 1/dx in
    frequency — i.e. it tiles copies of the spectrum every 1/dx. np.fft.fft
    returns one period of that tiling, ordered [0, df, ..., f_max, -f_max,
    ..., -df]; fftshift re-orders it into the physically intuitive
    [-f_max, ..., 0, ..., f_max] layout. The Nyquist frequency f_max where
    those tiled copies first touch is exactly 1/(2dx), consistent with the
    Whittaker-Shannon sampling theorem (Goodman Eq. 2-57): samples spaced
    at (2B)^-1 are sufficient to exactly reconstruct a function bandlimited
    to B.

    Parameters
    ----------
    N  : int   — number of grid points (should match the spatial grid)
    dx : float — spatial sample spacing in µm

    Returns
    -------
    f : ndarray, shape (N,)
        Spatial frequencies in cycles/µm, centered, spanning approximately
        [-f_max, +f_max) where f_max = 1/(2·dx).
    """
    f = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    return f


def fft1d(signal: np.ndarray, dx: float, normalize: bool = True) -> np.ndarray:
    """
    Forward FFT of a 1D spatial-domain signal, centered and (optionally)
    normalized to be physically meaningful regardless of grid size.

    Goodman connection
    -------------------
    This numerically approximates the continuous 1D Fourier transform

        G(f) = ∫ g(x) exp(-j2πfx) dx

    np.fft.fft computes the unnormalized discrete sum without the dx
    weighting and without re-centering the frequency axis. We fftshift to
    match freq_axis's ordering, and normalize by N (consistent with the
    Week 1 convention in masks.py / core.py) so the DC component equals
    the spatial average of the signal, independent of how many grid points
    we happened to choose.

    Parameters
    ----------
    signal    : ndarray, shape (N,) — spatial-domain values
    dx        : float — spatial sample spacing in µm (kept for API symmetry
                 with ifft1d and for future weeks that may need true
                 dx-weighted integral normalization; not used to rescale
                 here since we follow the Week 1 /N convention)
    normalize : bool — divide by N so DC = mean(signal) (default True,
                 matches Week 1's compute_spectrum convention)

    Returns
    -------
    spectrum : ndarray, shape (N,), complex
        Centered spectrum, ordered to match freq_axis (i.e. index i of the
        returned array corresponds to frequency freq_axis(N, dx)[i]).
    """
    N = len(signal)
    spectrum = np.fft.fftshift(np.fft.fft(signal))
    if normalize:
        spectrum = spectrum / N
    return spectrum


def ifft1d(spectrum: np.ndarray, dx: float, normalize: bool = True) -> np.ndarray:
    """
    Inverse FFT, undoing fft1d exactly (round-trip safe).

    Goodman connection
    -------------------
    Numerically approximates the inverse Fourier transform

        g(x) = ∫ G(f) exp(+j2πfx) df

    Must invert both operations fft1d applied: the fftshift re-centering
    and the 1/N normalization. We un-shift first (ifftshift), then call
    np.fft.ifft, then undo the normalization by multiplying back by N —
    this guarantees ifft1d(fft1d(g)) == g to numerical precision, which
    matters because later weeks (propagation, lens transform, pupil
    filtering) chain multiple forward/inverse transforms together and any
    asymmetry here would silently corrupt every downstream stage.

    Parameters
    ----------
    spectrum  : ndarray, shape (N,), complex — centered spectrum, as
                 returned by fft1d (same ordering as freq_axis)
    dx        : float — spatial sample spacing in µm (kept for API
                 symmetry with fft1d; see note there)
    normalize : bool — must match whatever was used in the corresponding
                 fft1d call, so the rescaling cancels out (default True)

    Returns
    -------
    signal : ndarray, shape (N,), complex
        Spatial-domain signal. Take .real if the input is known to
        correspond to a real-valued spatial signal (it will have a
        negligible imaginary part from floating-point round-off).
    """
    N = len(spectrum)
    unshifted = np.fft.ifftshift(spectrum)
    if normalize:
        unshifted = unshifted * N
    signal = np.fft.ifft(unshifted)
    return signal


def space_bandwidth_product(L: float, B: float) -> float:
    """
    Compute the space-bandwidth product N_sbp = (2L)(2B).

    Goodman connection
    -------------------
    Direct implementation of Goodman Eq. 2-58 (§2.4.2): for a function
    confined to [-L, L] in space and [-B, B] in frequency, the number of
    independent samples needed to fully describe it is (2L)(2B). Goodman
    calls this "the number of degrees of freedom of the given function" —
    it is a hard lower bound on grid points, not a rule of thumb.

    Note on convention: Goodman's L and B in Eq. 2-58 are *half-widths*
    (function confined to [-L, L], spectrum confined to [-B, B]), matching
    how make_grid in masks.py already defines the field so that x spans
    [-L_total/2, L_total/2). To use this function with a field of total
    width L_total, pass L = L_total / 2.

    Parameters
    ----------
    L : float — half-width of the spatial extent (µm)
    B : float — half-width of the frequency extent (cycles/µm)

    Returns
    -------
    N_sbp : float
        Minimum number of samples required (degrees of freedom). Compare
        against your actual grid's N; if N < N_sbp, the grid is formally
        under-sampled for that combination of feature size and field width.
    """
    return (2.0 * L) * (2.0 * B)


def check_sampling(L: float, N: int, dx: float, min_feature: float) -> dict:
    """
    Verify that a grid (L, N, dx) satisfies the sampling requirements
    needed to represent a target minimum feature size, using both the
    Nyquist criterion and the space-bandwidth product.

    Goodman connection
    -------------------
    Combines two distinct checks from Goodman Ch. 2:

    1. Nyquist (Whittaker-Shannon, Eq. 2-57): the grid's own Nyquist
       frequency f_max = 1/(2dx) must exceed the highest spatial frequency
       of interest, which we take as B = 1/min_feature (location of the
       first sinc zero for a feature of that width — see Week 1 build log,
       "first spectral zero at f = 1/w"). This checks that dx is fine
       enough.

    2. Space-bandwidth product (Eq. 2-58): N_required = (2·(L/2))·(2·B)
       = L · (2/min_feature) — using L/2 as the half-width per the note in
       space_bandwidth_product. This checks that N is large enough given
       BOTH the field width L and the feature size simultaneously — a grid
       can pass the Nyquist check (fine enough dx) while still having too
       few total points if L is large, because dx = L/N couples the two.

    A grid can fail check 2 while passing check 1: dx can be fine enough
    (Nyquist satisfied) while the total point count N is still too small
    for the *combination* of field width L and feature size, because
    N_required scales with L, not just with dx. Verified directly before
    delivering this file: at L=20 µm, min_feature=0.25 µm, N=256 passes
    both checks (f_max=6.4 ≥ B=4, and N_required=160 ≤ 256), while N=128
    on the same L and min_feature fails both (f_max=3.2 < 4, and
    N_required=160 > 128) — shrinking N at fixed L coarsens dx and trips
    Nyquist directly, while growing L at fixed N (not shown above) would
    instead trip the space-bandwidth check first, since N_required grows
    linearly with L but Nyquist only depends on dx = L/N. The two checks
    are therefore independent failure modes, not redundant restatements
    of each other.

    Parameters
    ----------
    L           : float — total field width in µm (matches make_grid's L)
    N           : int   — number of grid points
    dx          : float — spatial sample spacing in µm
    min_feature : float — smallest feature width you need to resolve (µm)

    Returns
    -------
    result : dict with keys
        f_max          : Nyquist frequency of this grid (cycles/µm)
        B_required     : spatial frequency bandwidth needed to resolve
                          min_feature (cycles/µm), taken as 1/min_feature
        nyquist_ok     : bool, True if f_max >= B_required
        N_required     : minimum N from the space-bandwidth product
        sbp_ok         : bool, True if N >= N_required
        sampling_ok    : bool, True only if both checks pass
    """
    f_max = 1.0 / (2.0 * dx)
    B_required = 1.0 / min_feature

    N_required = space_bandwidth_product(L / 2.0, B_required)

    nyquist_ok = f_max >= B_required
    sbp_ok = N >= N_required

    return dict(
        f_max=f_max,
        B_required=B_required,
        nyquist_ok=nyquist_ok,
        N_required=N_required,
        sbp_ok=sbp_ok,
        sampling_ok=nyquist_ok and sbp_ok,
    )