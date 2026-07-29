"""
physics/masks2d.py
---------------------
2D binary mask pattern generation -- the 2D generalization of masks.py.

WHY THIS MODULE EXISTS IN THE PIPELINE
-----------------------------------------
masks.py's 1D patterns (single_line, line_space_grating, two_lines) are all
built directly against a 1D spatial array x. Real lithography masks are
2D: a contact-hole array is a 2D grid of circular openings, not a 1D
line-space grating; a chip interconnect layout is a set of 2D rectangles,
not a single line. This module adds that second spatial dimension,
mirroring masks.py's own conventions (dark-field: mask value 1 = clear/
transparent opening, 0 = opaque background) but built against a Grid2D's
(X, Y) meshgrid instead of a bare 1D x array.

Nothing in masks.py is modified -- physics/grid2d.py's Grid2D is the only
new grid-construction machinery this module needs, and it in turn reuses
masks.make_grid unchanged per axis.

All spatial coordinates are in micrometers (µm).
"""

from typing import List, Tuple

import numpy as np


def single_hole(X: np.ndarray, Y: np.ndarray, diameter: float,
                 center: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """
    Create a 2D binary mask with a single circular opaque-field opening
    (dark field: 1 inside the hole, 0 outside) -- the 2D analogue of
    masks.single_line.

    Physically: an isolated contact hole, the simplest 2D lithography
    feature (the 2D equivalent of an isolated line), and the pattern used
    directly by lens2d.py's Airy-disk verification (a circular aperture's
    diffraction pattern is only cleanly comparable to the closed-form Airy
    profile for a single isolated circular opening, not a periodic array).

    Parameters
    ----------
    X, Y    : ndarray, shape (N, N) — 2D coordinate meshgrids (µm), from a
               Grid2D's .X, .Y
    diameter : float — hole diameter, µm
    center   : (cx, cy) — hole center position, µm (default origin)

    Returns
    -------
    mask : ndarray of 0.0/1.0, same shape as X
    """
    cx, cy = center
    radius_sq = (diameter / 2.0) ** 2
    return (((X - cx) ** 2 + (Y - cy) ** 2) <= radius_sq).astype(float)


def contact_hole_array(X: np.ndarray, Y: np.ndarray, hole_diameter: float,
                        pitch: float, center: Tuple[float, float] = (0.0, 0.0)
                        ) -> np.ndarray:
    """
    Create a periodic 2D array of circular openings on a square lattice --
    the 2D analogue of masks.line_space_grating.

    Physically: a contact/via array, one of the most common real
    lithography test patterns (memory arrays, interconnect vias) -- a
    dense, periodic 2D feature, in contrast to single_hole's isolated
    feature.

    Parameters
    ----------
    X, Y          : ndarray, shape (N, N) — 2D coordinate meshgrids (µm)
    hole_diameter : float — diameter of each hole, µm
    pitch         : float — center-to-center spacing between holes along
                     BOTH axes (square lattice), µm
    center        : (cx, cy) — lattice reference point, µm (default origin)

    Returns
    -------
    mask : ndarray of 0.0/1.0, same shape as X

    Notes
    -----
    Uses the same "shift by pitch/2, then modulo, then re-center" trick as
    masks.line_space_grating, applied independently to both axes: each
    point is folded into its own lattice cell (a square of side `pitch`
    centered on the nearest lattice point), then tested against the hole
    radius within that cell -- equivalent to line_space_grating's 1D
    modular-arithmetic approach, extruded to 2D.
    """
    cx, cy = center
    radius_sq = (hole_diameter / 2.0) ** 2

    x_mod = np.mod(X - cx + pitch / 2.0, pitch) - pitch / 2.0
    y_mod = np.mod(Y - cy + pitch / 2.0, pitch) - pitch / 2.0

    return ((x_mod ** 2 + y_mod ** 2) <= radius_sq).astype(float)


def rectangle(X: np.ndarray, Y: np.ndarray, x0: float, y0: float,
              width: float, height: float) -> np.ndarray:
    """
    Create a single axis-aligned rectangular opening (dark field: 1
    inside, 0 outside) -- a 2D primitive with no direct 1D analogue (a 1D
    mask has no independent second-axis extent), used as the building
    block for chip_block_layout below.

    Parameters
    ----------
    X, Y   : ndarray, shape (N, N) — 2D coordinate meshgrids (µm)
    x0, y0 : float — rectangle center position, µm
    width  : float — full extent along x, µm
    height : float — full extent along y, µm

    Returns
    -------
    mask : ndarray of 0.0/1.0, same shape as X
    """
    in_x = np.abs(X - x0) <= width / 2.0
    in_y = np.abs(Y - y0) <= height / 2.0
    return (in_x & in_y).astype(float)


# (x0, y0, width, height) -- one horizontal "bus" bar plus two vertical
# "stub" traces dropping from it at different x positions, a stylized
# stand-in for a simple chip interconnect layout cross-section. Fixed for
# v1 (no per-rectangle tuning UI -- see physics_assumptions.md's 2D
# Extension Assumptions section for why).
DEFAULT_RECTS: List[Tuple[float, float, float, float]] = [
    (0.0, 3.0, 8.0, 0.8),
    (-2.5, 0.1, 0.6, 5.0),
    (1.8, 0.85, 0.6, 3.5),
]


def chip_block_layout(X: np.ndarray, Y: np.ndarray,
                       rects: List[Tuple[float, float, float, float]] = None
                       ) -> np.ndarray:
    """
    Create a 2D mask from a fixed set of rectangles, combined into a
    single layout -- a stylized stand-in for a simple chip interconnect
    cross-section (a "bus" trace with a couple of "stub" traces dropping
    off it), NOT a real IC layout import.

    Physically: real lithography masks for interconnect/metal layers are
    sets of rectangles (Manhattan-geometry polygons) exactly like this,
    just far more numerous and irregular; this is the simplest
    illustrative case with the same underlying representation.

    Parameters
    ----------
    X, Y  : ndarray, shape (N, N) — 2D coordinate meshgrids (µm)
    rects : list of (x0, y0, width, height) tuples, µm -- defaults to
             DEFAULT_RECTS if None

    Returns
    -------
    mask : ndarray of 0.0/1.0, same shape as X

    Notes
    -----
    Combines individual rectangle() masks via np.maximum, mirroring
    masks.two_lines' own combination convention (so overlapping rectangles
    don't double-count transmission above 1.0).
    """
    if rects is None:
        rects = DEFAULT_RECTS

    mask = np.zeros_like(X)
    for x0, y0, width, height in rects:
        mask = np.maximum(mask, rectangle(X, Y, x0, y0, width, height))
    return mask
