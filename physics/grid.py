"""
physics/grid.py
----------------
Reusable grid/resolution parameter system.

WHY THIS MODULE EXISTS IN THE PIPELINE
---------------------------------------
Week 1 gave us a bare spatial array `x` (from masks.make_grid) and a
snapshot dict of derived quantities (masks.grid_info). That's fine for one
script, but every later stage of the pipeline — propagation, the lens
Fourier transform, pupil filtering, the imaging/thresholding step — needs
the *same* dx, N, frequency axis, and f_max over and over, and needs to
re-verify sampling validity every time the feature size of interest
changes (e.g. when sweeping line widths). Recomputing all of that from a
raw array in five different files is exactly the inline-physics
duplication the project conventions forbid.

Grid1D is a single object that:
  - builds the spatial grid (by calling masks.make_grid — not
    reimplementing it),
  - exposes the frequency axis and Nyquist frequency (by calling
    fft_engine.freq_axis — not reimplementing it),
  - and exposes a thin wrapper around fft_engine.check_sampling so later
    weeks can ask "grid.verify_sampling(min_feature)" on one object
    instead of threading L, N, dx through every call site by hand.

This module contains no new physics of its own — it is bookkeeping over
masks.py and fft_engine.py, which is why it is low-risk but still
verified below before delivery.
"""

import numpy as np

from masks import make_grid
from fft_engine import freq_axis, check_sampling


class Grid1D:
    """
    Bundles a 1D spatial grid with its derived spacing, frequency axis,
    and sampling-verification capability.

    Parameters
    ----------
    L : float
        Total field width in µm (matches masks.make_grid's L).
    N : int
        Number of grid points (power of 2 recommended for FFT efficiency,
        per the Week 1 convention).

    Attributes
    ----------
    L  : float — total field width (µm), as given
    N  : int   — number of grid points, as given
    x  : ndarray, shape (N,) — spatial coordinates (µm), from masks.make_grid
    dx : float — spatial sample spacing (µm)
    f  : ndarray, shape (N,) — centered frequency axis (cycles/µm), from
          fft_engine.freq_axis; index-aligned with x's FFT/IFFT pair
    df : float — frequency resolution (cycles/µm), equal to 1/L
    f_max : float — Nyquist frequency (cycles/µm), equal to 1/(2·dx)

    Notes
    -----
    All derived quantities are computed once at construction time and
    cached as attributes, since L and N together fully determine them —
    there is no later operation in this project that mutates a grid in
    place. If you need a different L or N, construct a new Grid1D rather
    than modifying an existing one.
    """

    def __init__(self, L: float, N: int):
        self.L = L
        self.N = N

        self.x = make_grid(L, N)
        # Cast to plain Python float: self.x is a numpy array, so
        # self.x[1] - self.x[0] is a numpy scalar by default. Left as-is,
        # that numpy-ness silently propagates into every later computation
        # that uses self.dx (e.g. check_sampling's f_max >= B_required
        # comparison returns np.bool_ instead of bool). Casting once here
        # keeps Grid1D's public attributes plain Python types throughout.
        self.dx = float(self.x[1] - self.x[0])

        self.f = freq_axis(N, self.dx)
        self.df = 1.0 / L
        self.f_max = 1.0 / (2.0 * self.dx)

    def verify_sampling(self, min_feature: float) -> dict:
        """
        Check whether this grid satisfies the Nyquist and space-bandwidth
        sampling requirements needed to resolve a target minimum feature
        size.

        Thin wrapper around fft_engine.check_sampling using this grid's
        own L, N, and dx — see that function's docstring for the full
        Goodman Eq. 2-57 / 2-58 derivation. Provided here so calling code
        only needs a Grid1D and a feature size, not four separate
        arguments threaded through by hand.

        Parameters
        ----------
        min_feature : float — smallest feature width to resolve (µm)

        Returns
        -------
        result : dict — same keys as fft_engine.check_sampling:
            f_max, B_required, nyquist_ok, N_required, sbp_ok, sampling_ok
        """
        return check_sampling(self.L, self.N, self.dx, min_feature)

    def __repr__(self) -> str:
        return (
            f"Grid1D(L={self.L} µm, N={self.N}, dx={self.dx:.5g} µm, "
            f"f_max={self.f_max:.5g} µm⁻¹, df={self.df:.5g} µm⁻¹)"
        )