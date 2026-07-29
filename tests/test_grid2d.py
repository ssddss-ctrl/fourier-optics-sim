"""
tests/test_grid2d.py
------------------------
Unit tests for physics/grid2d.py (Grid2D).

Grid2D introduces no new physics -- it wraps masks.make_grid and
fft_engine.freq_axis/check_sampling exactly like Grid1D, just meshgridded
onto two axes -- so these tests focus on consistency with Grid1D and the
underlying functions rather than re-deriving physical results already
covered in test_fft_engine.py.
"""

import numpy as np
import pytest

from grid import Grid1D
from grid2d import Grid2D
from masks import make_grid
from fft_engine import freq_axis, check_sampling


def test_grid2d_x_y_match_grid1d():
    g1 = Grid1D(L=20.0, N=256)
    g2 = Grid2D(L=20.0, N=256)
    assert np.allclose(g2.x, g1.x)
    assert np.allclose(g2.y, g1.x)
    assert np.isclose(g2.dx, g1.dx)


def test_grid2d_meshgrid_shapes():
    g = Grid2D(L=20.0, N=256)
    assert g.X.shape == (256, 256)
    assert g.Y.shape == (256, 256)
    assert g.FX.shape == (256, 256)
    assert g.FY.shape == (256, 256)


def test_grid2d_meshgrid_indexing_convention():
    """
    X must vary along columns (each row constant), Y must vary along rows
    (each column constant) -- np.meshgrid's default 'xy' indexing, matching
    Plotly's go.Heatmap row<->y / column<->x convention documented in this
    module. Verified directly (not just shape) since a row/column mixup
    here would silently corrupt every 2D mask/pupil built on this grid.
    """
    g = Grid2D(L=20.0, N=8)
    # Every row of X must be identical (X doesn't vary with row index i.e. y)
    assert np.allclose(g.X[0, :], g.X[-1, :])
    assert np.allclose(g.X[0, :], g.x)
    # Every column of Y must be identical (Y doesn't vary with column index i.e. x)
    assert np.allclose(g.Y[:, 0], g.Y[:, -1])
    assert np.allclose(g.Y[:, 0], g.y)


def test_grid2d_frequency_axes_match_fft_engine():
    g = Grid2D(L=20.0, N=256)
    assert np.allclose(g.fx, freq_axis(256, g.dx))
    assert np.allclose(g.fy, freq_axis(256, g.dx))


def test_verify_sampling_matches_check_sampling_directly():
    g = Grid2D(L=20.0, N=256)
    assert g.verify_sampling(min_feature=0.25) == check_sampling(
        g.L, g.N, g.dx, 0.25
    )


def test_verify_sampling_matches_grid1d_at_same_params():
    """Grid2D's per-axis sampling check must agree exactly with Grid1D's,
    since check_sampling has no axis-count dependence."""
    g1 = Grid1D(L=20.0, N=256)
    g2 = Grid2D(L=20.0, N=256)
    assert g1.verify_sampling(0.25) == g2.verify_sampling(0.25)


def test_verify_sampling_flags_undersampled_grid():
    g = Grid2D(L=20.0, N=128)
    result = g.verify_sampling(min_feature=0.25)
    assert result["sampling_ok"] == False


def test_repr_contains_key_parameters():
    g = Grid2D(L=20.0, N=256)
    r = repr(g)
    assert "L=20.0" in r
    assert "N=256x256" in r


def test_different_grids_are_independent():
    """Constructing a second Grid2D must not mutate or alias the first."""
    g1 = Grid2D(L=20.0, N=256)
    g2 = Grid2D(L=10.0, N=128)
    assert g1.L != g2.L
    assert g1.N != g2.N
    assert not np.array_equal(g1.X, g2.X)
