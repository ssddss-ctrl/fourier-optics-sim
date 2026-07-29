"""
physics/grid2d.py
--------------------
2D generalization of grid.py's Grid1D: bundles a 2D spatial grid with its
derived spacing, frequency axes, and sampling-verification capability.

WHY THIS MODULE EXISTS IN THE PIPELINE
-----------------------------------------
grid.py's Grid1D bundles everything the 1D pipeline (propagation, lens,
imaging, opc) needs so no downstream module re-derives L/N/dx/f/df/f_max
by hand. This is the same bookkeeping, extended to two spatial dimensions,
for the 2D pipeline (masks2d.py, lens2d.py, imaging2d.py): a mask/target
lives on a meshgrid (X, Y), not a bare 1D x array, and pupil filtering
needs a 2D frequency meshgrid (FX, FY) rather than a single f axis.

This module contains no new physics of its own -- it is bookkeeping over
masks.py and fft_engine.py, exactly like Grid1D, extended to two axes via
np.meshgrid.

SIMPLIFYING ASSUMPTION (stated explicitly)
---------------------------------------------
Square field, square grid only: a single L and N apply to BOTH the x and y
axes (Lx = Ly = L, Nx = Ny = N), not independent per-axis field widths or
sample counts. Every 2D mask pattern this project defines (contact-hole
arrays, simple rectangular layouts) reads naturally on a square field, and
a square grid keeps fft2d/ifft2d's dx-only (not dx, dy) signature valid
without complicating every downstream 2D function with two spacing
parameters. A non-square field is not supported by this class.

All spatial coordinates: µm
All spatial frequencies: cycles/µm (µm⁻¹)
"""

import numpy as np

from masks import make_grid
from fft_engine import freq_axis, check_sampling


class Grid2D:
    """
    Bundles a 2D (square) spatial grid with its derived spacing, 2D
    frequency meshgrid, and sampling-verification capability -- the 2D
    analogue of grid.Grid1D.

    Parameters
    ----------
    L : float
        Total field width in µm, applied to BOTH axes (matches
        masks.make_grid's L; see module docstring's square-grid
        assumption).
    N : int
        Number of grid points per axis (matches masks.make_grid's N;
        applied to both axes, so the full grid has N*N points total).

    Attributes
    ----------
    L, N    : as given
    x, y    : ndarray, shape (N,) — 1D spatial coordinates (µm) per axis,
               from masks.make_grid; identical since the grid is square
               (y is a separate attribute, not just an alias for x, so
               callers/tests can catch an accidental x/y mixup by comparing
               against the wrong one deliberately)
    X, Y    : ndarray, shape (N, N) — 2D coordinate meshgrids, via
               np.meshgrid(x, y) (default 'xy' indexing: X varies along
               columns, Y varies along rows -- row index -> y, column
               index -> x). This indexing convention matches Plotly's
               go.Heatmap (z[row][col], row <-> y) used by the frontend.
    dx      : float — spatial sample spacing (µm), shared by both axes
    fx, fy  : ndarray, shape (N,) — 1D frequency axes (cycles/µm) per
               axis, from fft_engine.freq_axis; identical since the grid
               is square
    FX, FY  : ndarray, shape (N, N) — 2D frequency meshgrids, via
               np.meshgrid(fx, fy), same indexing convention as X/Y
    df      : float — frequency resolution (cycles/µm), equal to 1/L
    f_max   : float — Nyquist frequency (cycles/µm), equal to 1/(2*dx)

    Notes
    -----
    All derived quantities are computed once at construction time and
    cached as attributes, mirroring Grid1D's own immutability convention:
    construct a new Grid2D rather than mutating an existing one.
    """

    def __init__(self, L: float, N: int):
        self.L = L
        self.N = N

        self.x = make_grid(L, N)
        self.y = make_grid(L, N)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Cast to plain Python float, matching Grid1D's own reasoning:
        # self.x[1] - self.x[0] is a numpy scalar by default, and that
        # numpy-ness would otherwise silently propagate into every later
        # 2D computation that uses self.dx.
        self.dx = float(self.x[1] - self.x[0])

        self.fx = freq_axis(N, self.dx)
        self.fy = freq_axis(N, self.dx)
        self.FX, self.FY = np.meshgrid(self.fx, self.fy)

        self.df = 1.0 / L
        self.f_max = 1.0 / (2.0 * self.dx)

    def verify_sampling(self, min_feature: float) -> dict:
        """
        Check whether this grid satisfies the Nyquist and space-bandwidth
        sampling requirements needed to resolve a target minimum feature
        size, along either axis (the grid is square, so one axis's check
        is exactly the other axis's check).

        Thin wrapper around fft_engine.check_sampling using this grid's
        own L, N, and dx -- identical to Grid1D.verify_sampling, since
        check_sampling itself has no axis-count dependence (it operates
        on scalar L/N/dx already, per axis).

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
            f"Grid2D(L={self.L} µm, N={self.N}x{self.N}, dx={self.dx:.5g} µm, "
            f"f_max={self.f_max:.5g} µm⁻¹, df={self.df:.5g} µm⁻¹)"
        )
