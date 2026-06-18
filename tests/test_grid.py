"""
tests/test_grid.py
--------------------
Unit tests for physics/grid.py (Grid1D).

Grid1D introduces no new physics -- it wraps masks.make_grid and
fft_engine.freq_axis/check_sampling -- so these tests focus on consistency
with those underlying functions rather than re-deriving physical results
already covered in test_fft_engine.py.
"""

import numpy as np
import pytest

from grid import Grid1D
from masks import make_grid, grid_info, line_space_grating
from fft_engine import freq_axis, fft1d, ifft1d, check_sampling


def test_grid1d_x_matches_make_grid():
    g = Grid1D(L=20.0, N=256)
    assert np.allclose(g.x, make_grid(20.0, 256))


def test_grid1d_derived_quantities_match_grid_info():
    g = Grid1D(L=20.0, N=256)
    info = grid_info(g.x)
    assert np.isclose(g.dx, info["dx"])
    assert np.isclose(g.df, info["df"])
    assert np.isclose(g.f_max, info["f_max"])


def test_grid1d_frequency_axis_matches_fft_engine():
    g = Grid1D(L=20.0, N=256)
    assert np.allclose(g.f, freq_axis(256, g.dx))


def test_grid1d_end_to_end_round_trip():
    g = Grid1D(L=20.0, N=256)
    mask = line_space_grating(g.x, pitch=2.0, duty_cycle=0.5)
    spectrum = fft1d(mask, dx=g.dx)
    recovered = ifft1d(spectrum, dx=g.dx)
    assert np.max(np.abs(recovered.real - mask)) < 1e-10


def test_verify_sampling_matches_check_sampling_directly():
    g = Grid1D(L=20.0, N=256)
    assert g.verify_sampling(min_feature=0.25) == check_sampling(
        g.L, g.N, g.dx, 0.25
    )


def test_verify_sampling_flags_undersampled_grid():
    g = Grid1D(L=20.0, N=128)
    result = g.verify_sampling(min_feature=0.25)
    assert result["sampling_ok"] == False


def test_repr_contains_key_parameters():
    g = Grid1D(L=20.0, N=256)
    r = repr(g)
    assert "L=20.0" in r
    assert "N=256" in r


def test_different_grids_are_independent():
    """Constructing a second Grid1D must not mutate or alias the first."""
    g1 = Grid1D(L=20.0, N=256)
    g2 = Grid1D(L=10.0, N=128)
    assert g1.L != g2.L
    assert g1.N != g2.N
    assert not np.array_equal(g1.x, g2.x)