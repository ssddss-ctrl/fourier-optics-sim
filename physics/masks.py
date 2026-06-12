"""
physics/masks.py
----------------
1D binary mask and target pattern generation.

A "mask" in lithography is the physical template that blocks or passes light.
In simulation, it's just a 1D array of 0s (opaque) and 1s (transparent).

All spatial coordinates are in micrometers (µm).
"""

import numpy as np


def make_grid(L: float, N: int) -> np.ndarray:
    """
    Create a 1D spatial coordinate array centered at zero.

    Parameters
    ----------
    L : float
        Total field width in µm.
    N : int
        Number of grid points (use a power of 2 for FFT efficiency).

    Returns
    -------
    x : ndarray, shape (N,)
        Spatial coordinates in µm, centered at 0, spanning [-L/2, L/2).

    Notes
    -----
    We center the grid at zero so that symmetric features (like an isolated
    line centered in the field) have their center at x=0. This makes the
    Fourier transform real-valued and symmetric for even functions, which
    is easier to interpret.
    """
    dx = L / N
    x = np.arange(N) * dx - L / 2
    return x


def single_line(x: np.ndarray, width: float, center: float = 0.0) -> np.ndarray:
    """
    Create a binary mask with a single opaque line (dark field: 1 outside, 0 inside).

    Wait — in lithography conventions:
      - "bright field" mask: background is 1 (transparent), line is 0 (opaque chrome)
      - "dark field" mask: background is 0, line is 1

    Here we use DARK FIELD: the line itself is 1 (transparent opening),
    background is 0. This is the "clear line on opaque background" convention
    used in positive-tone lithography.

    Parameters
    ----------
    x      : ndarray — spatial grid (µm)
    width  : float   — line width in µm
    center : float   — center position of the line (µm), default 0

    Returns
    -------
    mask : ndarray of 0s and 1s
    """
    mask = np.zeros_like(x)
    mask[np.abs(x - center) <= width / 2] = 1.0
    return mask


def line_space_grating(x: np.ndarray, pitch: float, duty_cycle: float = 0.5) -> np.ndarray:
    """
    Create a periodic binary line-space grating.

    A grating is the simplest model of a dense pattern — like the lines in a
    DRAM array. The pitch is the center-to-center distance between lines.
    Duty cycle = (line width) / pitch. At 0.5, lines and spaces are equal.

    Parameters
    ----------
    x          : ndarray — spatial grid (µm)
    pitch      : float   — period of the grating in µm
    duty_cycle : float   — fraction of period that is "line" (transparent), default 0.5

    Returns
    -------
    mask : ndarray of 0s and 1s

    Notes
    -----
    Uses modular arithmetic: x mod pitch gives position within one period.
    Points where that position < line_width are "on" (transparent).
    """
    # Shift so modulo is taken symmetrically (optional, makes plot look clean)
    x_shifted = x + pitch / 2
    # Position within one period: 0 to pitch
    x_mod = np.mod(x_shifted, pitch)
    line_width = duty_cycle * pitch
    mask = (x_mod < line_width).astype(float)
    return mask


def grid_info(x: np.ndarray) -> dict:
    """
    Return key grid parameters as a dict. Useful for sanity-checking sampling.

    Returns dx, L, N, df (frequency resolution), f_max (Nyquist limit).
    """
    N = len(x)
    dx = x[1] - x[0]
    L = N * dx
    df = 1.0 / L
    f_max = 1.0 / (2.0 * dx)
    return dict(N=N, dx=dx, L=L, df=df, f_max=f_max)
