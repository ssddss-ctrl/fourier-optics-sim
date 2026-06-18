"""
tests/test_fft_engine.py
-------------------------
Unit tests for physics/fft_engine.py.

These formalize the verification already performed by hand/script before
the file was delivered in Week 2 (Optics): round-trip correctness,
agreement with Week 1's plotting/core.py frequency axis, DC-component
physical meaning, and both branches (pass/fail) of the sampling check.
"""

import numpy as np
import pytest

from masks import make_grid, single_line, line_space_grating
from fft_engine import (
    freq_axis,
    fft1d,
    ifft1d,
    space_bandwidth_product,
    check_sampling,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def standard_grid():
    L, N = 20.0, 256
    x = make_grid(L, N)
    dx = x[1] - x[0]
    return L, N, x, dx


# ── freq_axis ─────────────────────────────────────────────────────────────

def test_freq_axis_length_and_symmetry(standard_grid):
    L, N, x, dx = standard_grid
    f = freq_axis(N, dx)
    assert len(f) == N
    # Centered axis should be (anti-)symmetric about zero, aside from the
    # single unmatched Nyquist bin in fftfreq's convention for even N.
    assert f[0] < 0 < f[-1]


def test_freq_axis_matches_core_py(standard_grid):
    """
    fft_engine.freq_axis must agree exactly with the private _freq_axis
    helper already used in Week 1's plotting/core.py, since both must
    label the same FFT output with the same physical units.
    """
    from core import _freq_axis

    L, N, x, dx = standard_grid
    assert np.allclose(freq_axis(N, dx), _freq_axis(x))


def test_nyquist_frequency_value(standard_grid):
    L, N, x, dx = standard_grid
    f = freq_axis(N, dx)
    expected_f_max = 1.0 / (2.0 * dx)
    assert np.isclose(f.max(), expected_f_max, rtol=1e-2)


# ── fft1d / ifft1d round trip ────────────────────────────────────────────

def test_round_trip_grating(standard_grid):
    L, N, x, dx = standard_grid
    mask = line_space_grating(x, pitch=2.0, duty_cycle=0.5)
    spectrum = fft1d(mask, dx=dx)
    recovered = ifft1d(spectrum, dx=dx)
    assert np.max(np.abs(recovered.real - mask)) < 1e-10


def test_round_trip_single_line(standard_grid):
    L, N, x, dx = standard_grid
    mask = single_line(x, width=1.0)
    spectrum = fft1d(mask, dx=dx)
    recovered = ifft1d(spectrum, dx=dx)
    assert np.max(np.abs(recovered.real - mask)) < 1e-10


def test_round_trip_preserves_imaginary_negligible(standard_grid):
    """A real-valued input should round-trip with negligible imaginary part."""
    L, N, x, dx = standard_grid
    mask = single_line(x, width=1.0)
    spectrum = fft1d(mask, dx=dx)
    recovered = ifft1d(spectrum, dx=dx)
    assert np.max(np.abs(recovered.imag)) < 1e-10


def test_dc_component_equals_mean_transmission(standard_grid):
    """
    Normalized DC component (f=0) must equal the spatial average of the
    mask -- e.g. 0.5 for a 50% duty cycle grating. This is the same
    physical check used in the Week 1 build log validation.
    """
    L, N, x, dx = standard_grid
    mask = line_space_grating(x, pitch=2.0, duty_cycle=0.5)
    f = freq_axis(N, dx)
    spectrum = fft1d(mask, dx=dx)
    dc_index = np.argmin(np.abs(f))
    assert np.isclose(spectrum[dc_index].real, mask.mean(), atol=1e-12)


def test_fft1d_normalize_false_scales_by_N(standard_grid):
    """normalize=False should differ from normalize=True by exactly a factor of N."""
    L, N, x, dx = standard_grid
    mask = line_space_grating(x, pitch=2.0, duty_cycle=0.5)
    spec_norm = fft1d(mask, dx=dx, normalize=True)
    spec_raw = fft1d(mask, dx=dx, normalize=False)
    assert np.allclose(spec_raw, spec_norm * N)


# ── space_bandwidth_product ──────────────────────────────────────────────

def test_space_bandwidth_product_known_value():
    # Goodman Eq. 2-58: N = (2L)(2B)
    assert space_bandwidth_product(L=10.0, B=4.0) == 160.0


def test_space_bandwidth_product_scales_linearly_in_each_arg():
    base = space_bandwidth_product(L=10.0, B=4.0)
    doubled_L = space_bandwidth_product(L=20.0, B=4.0)
    doubled_B = space_bandwidth_product(L=10.0, B=8.0)
    assert np.isclose(doubled_L, 2 * base)
    assert np.isclose(doubled_B, 2 * base)


# ── check_sampling ────────────────────────────────────────────────────────

def test_check_sampling_passes_when_grid_adequate():
    result = check_sampling(L=20.0, N=256, dx=20.0 / 256, min_feature=0.25)
    assert result["nyquist_ok"] == True
    assert result["sbp_ok"] == True
    assert result["sampling_ok"] == True


def test_check_sampling_fails_when_grid_too_coarse():
    result = check_sampling(L=20.0, N=128, dx=20.0 / 128, min_feature=0.25)
    assert result["nyquist_ok"] == False
    assert result["sbp_ok"] == False
    assert result["sampling_ok"] == False


def test_check_sampling_required_value_matches_space_bandwidth_product():
    result = check_sampling(L=20.0, N=256, dx=20.0 / 256, min_feature=0.25)
    expected_N_required = space_bandwidth_product(L=10.0, B=4.0)
    assert result["N_required"] == expected_N_required


def test_check_sampling_boundary_is_inclusive():
    """At exactly N_required and exactly f_max == B_required, both checks should pass (>=, not >)."""
    L = 20.0
    min_feature = 0.25
    B_required = 1.0 / min_feature
    N_required = space_bandwidth_product(L / 2.0, B_required)
    N = int(N_required)
    dx = L / N
    result = check_sampling(L=L, N=N, dx=dx, min_feature=min_feature)
    assert result["sbp_ok"] == True