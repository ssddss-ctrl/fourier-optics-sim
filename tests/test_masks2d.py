"""
tests/test_masks2d.py
-------------------------
Unit tests for physics/masks2d.py.

Test organization mirrors tests/test_masks.py in spirit: exact,
hand-computable checks where the geometry allows it (rectangles, whose
pixel count is an exact separable product of two independent 1D extents),
and symmetry/approximate-area checks for circular features (a discretized
circle's pixel count has no simple exact closed form, the same reason
test_lens.py's own grating checks use an approximate duty-cycle-squared
comparison rather than an exact pixel count).
"""

import numpy as np
import pytest

from grid2d import Grid2D
from masks import make_grid
from masks2d import single_hole, contact_hole_array, rectangle, chip_block_layout, DEFAULT_RECTS


@pytest.fixture
def grid():
    # L=8 um, N=64 -> dx=0.125 um, an exact binary fraction so grid-aligned
    # feature boundaries land exactly on sample points (no floating-point
    # boundary ambiguity in the <= comparisons below).
    return Grid2D(L=8.0, N=64)


# ── single_hole ───────────────────────────────────────────────────────────

def test_single_hole_is_binary(grid):
    mask = single_hole(grid.X, grid.Y, diameter=2.0)
    assert set(np.unique(mask)).issubset({0.0, 1.0})


def test_single_hole_symmetric_under_reflection(grid):
    """A hole centered at the origin must be symmetric under reflection
    about either axis -- a row/column mixup in the underlying (X-cx)^2 +
    (Y-cy)^2 computation would break this.

    Note: masks.make_grid's convention (x = arange(N)*dx - L/2) includes
    -L/2 but not +L/2 for even N -- the same one-unmatched-endpoint
    asymmetry test_imaging.py's test_otf_is_even_symmetric already
    documents and excludes for the frequency axis. Index 0 is excluded
    here for the same reason before checking reflection symmetry."""
    mask = single_hole(grid.X, grid.Y, diameter=2.0, center=(0.0, 0.0))
    interior = mask[1:, 1:]
    assert np.array_equal(interior, interior[:, ::-1])  # reflect x
    assert np.array_equal(interior, interior[::-1, :])  # reflect y


def test_single_hole_off_center_breaks_symmetry_correctly(grid):
    """An off-center hole must shift in the direction matching its stated
    (cx, cy) -- verified by checking which half-plane contains it, so an
    X/Y axis swap bug would be caught (a swap would shift it along the
    wrong axis)."""
    mask = single_hole(grid.X, grid.Y, diameter=1.0, center=(2.0, 0.0))
    # Center shifted in +x: mass should be concentrated at grid.X > 0.
    on_pixels_x = grid.X[mask > 0.5]
    on_pixels_y = grid.Y[mask > 0.5]
    assert np.all(on_pixels_x > 0.0)
    assert np.isclose(np.mean(on_pixels_y), 0.0, atol=1e-9)


def test_single_hole_area_approximately_matches_circle_area(grid):
    """Discretized circle area has no simple exact pixel-count formula
    (unlike a grid-aligned rectangle), so this is an approximate check,
    same convention as test_lens.py's grating duty-cycle comparisons."""
    diameter = 2.0
    mask = single_hole(grid.X, grid.Y, diameter=diameter, center=(0.0, 0.0))
    measured_area = mask.sum() * grid.dx ** 2
    expected_area = np.pi * (diameter / 2.0) ** 2
    assert measured_area == pytest.approx(expected_area, rel=0.05)


# ── contact_hole_array ────────────────────────────────────────────────────

def test_contact_hole_array_is_binary(grid):
    mask = contact_hole_array(grid.X, grid.Y, hole_diameter=0.6, pitch=2.0)
    assert set(np.unique(mask)).issubset({0.0, 1.0})


def test_contact_hole_array_symmetric_about_origin(grid):
    """Same one-unmatched-endpoint exclusion as test_single_hole_symmetric_under_reflection."""
    mask = contact_hole_array(grid.X, grid.Y, hole_diameter=0.6, pitch=2.0, center=(0.0, 0.0))
    interior = mask[1:, 1:]
    assert np.array_equal(interior, interior[:, ::-1])
    assert np.array_equal(interior, interior[::-1, :])


def test_contact_hole_array_total_area_approximately_matches_n_holes():
    """L=8, pitch=2 -> exactly 4 periods per axis (grid-aligned choice), so
    total hole area should approximate n_holes * single-hole area. Uses a
    finer grid than the shared fixture (dx=0.03125 vs. 0.125) so the
    hole_diameter=0.6 circle is resolved by ~19 samples across rather than
    ~5 -- a coarser grid makes this approximation's discretization error
    too large relative to a tight tolerance (a circle only ~5 pixels
    across is dominated by boundary staircasing)."""
    fine_grid = Grid2D(L=8.0, N=256)
    hole_diameter, pitch = 0.6, 2.0
    mask = contact_hole_array(fine_grid.X, fine_grid.Y, hole_diameter=hole_diameter, pitch=pitch)
    measured_area = mask.sum() * fine_grid.dx ** 2

    n_periods = fine_grid.L / pitch  # 4.0, exact for this grid/pitch choice
    assert n_periods == pytest.approx(4.0)
    expected_area = (n_periods ** 2) * np.pi * (hole_diameter / 2.0) ** 2
    assert measured_area == pytest.approx(expected_area, rel=0.05)


# ── rectangle ─────────────────────────────────────────────────────────────

def test_rectangle_is_binary(grid):
    mask = rectangle(grid.X, grid.Y, x0=0.0, y0=0.0, width=2.0, height=1.0)
    assert set(np.unique(mask)).issubset({0.0, 1.0})


def test_rectangle_exact_pixel_count_matches_separable_1d_extents(grid):
    """
    Key hand-verified check: rectangle() is exactly the outer product of
    two independent 1D indicator functions (in_x depends only on the X
    column value, in_y only on the Y row value), so its total pixel count
    must exactly equal (# grid.x satisfying |x-x0|<=width/2) times
    (# grid.y satisfying |y-y0|<=height/2) -- computed independently here
    via masks.make_grid directly, not re-deriving rectangle()'s own logic.
    """
    x0, y0, width, height = 0.0, 0.0, 2.0, 1.0
    x = make_grid(grid.L, grid.N)
    count_x = np.sum(np.abs(x - x0) <= width / 2.0)
    count_y = np.sum(np.abs(x - y0) <= height / 2.0)  # same 1D axis, square grid
    expected_total = int(count_x) * int(count_y)

    mask = rectangle(grid.X, grid.Y, x0, y0, width, height)
    assert int(mask.sum()) == expected_total


def test_rectangle_off_center_position(grid):
    mask = rectangle(grid.X, grid.Y, x0=2.0, y0=-1.0, width=1.0, height=1.0)
    on_pixels_x = grid.X[mask > 0.5]
    on_pixels_y = grid.Y[mask > 0.5]
    assert np.isclose(np.mean(on_pixels_x), 2.0, atol=1e-9)
    assert np.isclose(np.mean(on_pixels_y), -1.0, atol=1e-9)


# ── chip_block_layout ─────────────────────────────────────────────────────

def test_chip_block_layout_is_binary(grid):
    mask = chip_block_layout(grid.X, grid.Y)
    assert set(np.unique(mask)).issubset({0.0, 1.0})


def test_chip_block_layout_matches_union_of_individual_rectangles(grid):
    """chip_block_layout must be exactly the elementwise max (union) of
    its constituent rectangle() calls -- a plumbing check that the
    combination logic matches its own stated definition."""
    mask = chip_block_layout(grid.X, grid.Y, rects=DEFAULT_RECTS)

    expected = np.zeros_like(grid.X)
    for x0, y0, width, height in DEFAULT_RECTS:
        expected = np.maximum(expected, rectangle(grid.X, grid.Y, x0, y0, width, height))

    assert np.array_equal(mask, expected)


def test_chip_block_layout_custom_rects():
    grid = Grid2D(L=8.0, N=64)
    custom_rects = [(0.0, 0.0, 2.0, 2.0)]
    mask = chip_block_layout(grid.X, grid.Y, rects=custom_rects)
    expected = rectangle(grid.X, grid.Y, 0.0, 0.0, 2.0, 2.0)
    assert np.array_equal(mask, expected)


def test_chip_block_layout_default_uses_default_rects(grid):
    """Calling with no `rects` argument at all must match DEFAULT_RECTS explicitly."""
    mask_default = chip_block_layout(grid.X, grid.Y)
    mask_explicit = chip_block_layout(grid.X, grid.Y, rects=DEFAULT_RECTS)
    assert np.array_equal(mask_default, mask_explicit)
