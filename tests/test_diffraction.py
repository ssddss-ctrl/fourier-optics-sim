"""
tests/test_diffraction.py
--------------------------
Unit tests for physics/diffraction.py: validates the numerical Fraunhofer
simulator (fraunhofer_pattern) against the closed-form analytic predictions
from Goodman Ch. 4 for all three Week 8 pattern-library targets (single
line, line-space grating, two nearby lines).
"""

import numpy as np
import pytest

from masks import make_grid, single_line, line_space_grating, two_lines
from grid import Grid1D
import diffraction as diff


# Shared grid for all tests: fine enough that discretizing an aperture
# width to the nearest grid point introduces negligible error once widths
# are snapped to a multiple of dx (see _snap below).
L, N = 200.0, 8192
WAVELENGTH = 0.5  # µm


@pytest.fixture
def grid():
    return Grid1D(L, N)


def _snap(value: float, dx: float) -> float:
    """
    Round a physical length to the nearest multiple of the grid spacing.

    Not physics — a test-only helper for aperture *widths*. single_line
    and two_lines build binary masks on a discrete grid, so a width that
    isn't an exact multiple of dx gets silently widened or narrowed by up
    to dx/2 when it's rasterized (confirmed by hand before writing these
    tests: for L=200, N=8192, requesting width=1.0 actually rasterizes to
    1.025 µm, i.e. mask.sum()*dx != 1.0). Snapping removes that
    rasterization error from the comparison so slit/two-line tests check
    the Fraunhofer math, not grid quantization.

    NOTE: this is deliberately NOT used for grating pitch — see the
    comment in test_grating_order_spacing_matches_lambda_z_over_pitch for
    why pitch needs a different (frequency-bin-alignment) treatment.
    """
    return round(value / dx) * dx


def _find_peaks(y: np.ndarray, min_height: float) -> np.ndarray:
    """Test-only local-maximum finder (no scipy dependency in this project)."""
    peaks = []
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1] and y[i] > min_height:
            peaks.append(i)
    return np.array(peaks, dtype=int)


# ── fraunhofer_far_field_distance / check_fraunhofer_validity ───────────────

def test_far_field_distance_formula(grid):
    """z_min = 2*D^2/lambda, the antenna designer's formula (Goodman §4.3)."""
    D = 2.0
    z_min = diff.fraunhofer_far_field_distance(D, WAVELENGTH)
    assert z_min == pytest.approx(2.0 * D**2 / WAVELENGTH)
    assert isinstance(z_min, float)  # not numpy.floating


def test_check_fraunhofer_validity_true_and_false(grid):
    D = 1.0
    z_min = diff.fraunhofer_far_field_distance(D, WAVELENGTH)
    assert diff.check_fraunhofer_validity(D, WAVELENGTH, z_min * 2) is True
    assert diff.check_fraunhofer_validity(D, WAVELENGTH, z_min * 0.5) is False
    assert isinstance(diff.check_fraunhofer_validity(D, WAVELENGTH, z_min * 2), bool)


# ── single slit vs analytic sinc^2 (Eq. 4-26/4-27) ──────────────────────────

def test_slit_matches_analytic_sinc2(grid):
    width = _snap(1.0, grid.dx)
    z = 200.0
    assert diff.check_fraunhofer_validity(width, WAVELENGTH, z)

    mask = single_line(grid.x, width)
    x_obs, field, I = diff.fraunhofer_pattern(mask, grid, WAVELENGTH, z)
    I_analytic = diff.analytic_slit_intensity(x_obs, width, WAVELENGTH, z)

    assert np.max(np.abs(I - I_analytic)) < 5e-3


def test_slit_first_zero_at_lambda_z_over_width(grid):
    """
    First zero of the sinc^2 main lobe should sit at x = lambda*z/width.
    Checked by sampling I at the grid point nearest the predicted zero
    (not by scanning for the first point below a threshold — that method
    was tried by hand first and found to trigger early, on the descending
    sidelobe tail well before the true zero, since intensity is already
    small approaching the zero and not just at it).
    """
    width = _snap(1.0, grid.dx)
    z = 200.0
    mask = single_line(grid.x, width)
    x_obs, field, I = diff.fraunhofer_pattern(mask, grid, WAVELENGTH, z)

    predicted_zero = WAVELENGTH * z / width
    idx = np.argmin(np.abs(x_obs - predicted_zero))
    assert I[idx] < 1e-3


def test_slit_peak_normalized_to_one(grid):
    mask = single_line(grid.x, _snap(1.0, grid.dx))
    _, _, I = diff.fraunhofer_pattern(mask, grid, WAVELENGTH, 200.0)
    assert I.max() == pytest.approx(1.0)


# ── grating order spacing and relative heights (Eq. 4-34/4-36, Prob. 4-12) ──

def test_grating_order_spacing_matches_lambda_z_over_pitch(grid):
    """
    pitch is deliberately left at an exact value (2.0 µm) with L/pitch an
    exact integer (100 periods exactly fill the 200 µm field), rather than
    snapped to a multiple of dx the way slit widths are. Snapping pitch to
    dx was tried by hand first and found to misalign the grating's
    fundamental frequency 1/pitch from the FFT's frequency bins
    (df = 1/L), leaking ~3% of each order's power into its neighbors and
    failing a tight tolerance. Choosing L/pitch to be an integer instead
    guarantees 1/pitch lands exactly on a frequency bin, eliminating that
    leakage — confirmed by hand (order 1 height changed from a ~3% error
    down to a ~1e-7 relative error after switching from dx-snapping to
    integer-period alignment).
    """
    pitch = 2.0
    assert L / pitch == int(L / pitch)
    duty = 0.5
    z = 50.0
    mask = line_space_grating(grid.x, pitch, duty)
    x_obs, _, I = diff.fraunhofer_pattern(mask, grid, WAVELENGTH, z)

    predicted = diff.analytic_grating_order_positions(pitch, WAVELENGTH, z, max_order=3)
    x_obs_spacing = WAVELENGTH * z / L
    tol = 2 * x_obs_spacing

    for x_pred in predicted:
        order = round(x_pred / (WAVELENGTH * z / pitch))
        rel = diff.analytic_grating_relative_intensity(order, duty)
        if rel < 1e-6:
            continue  # order 2 vanishes exactly for duty_cycle=0.5
        idx = np.argmin(np.abs(x_obs - x_pred))
        assert x_obs[idx] == pytest.approx(x_pred, abs=tol)


def test_grating_relative_intensities_match_fourier_series(grid):
    """
    I_order/I_0 = sinc(order*duty_cycle)^2, per Problem 4-12(a)'s
    v_k = |c_k|^2 applied to a square-wave grating's Fourier coefficients.
    """
    pitch = 2.0
    duty = 0.5
    z = 50.0
    mask = line_space_grating(grid.x, pitch, duty)
    x_obs, _, I = diff.fraunhofer_pattern(mask, grid, WAVELENGTH, z)

    for order in [0, 1, 3]:  # skip order 2: exactly zero for duty_cycle=0.5
        x_pred = order * WAVELENGTH * z / pitch
        idx = np.argmin(np.abs(x_obs - x_pred))
        numeric_rel = I[idx]
        analytic_rel = diff.analytic_grating_relative_intensity(order, duty)
        assert numeric_rel == pytest.approx(analytic_rel, abs=1e-3)


# ── two nearby lines vs shift-theorem interference prediction ───────────────

def test_two_lines_matches_envelope_times_interference(grid):
    """
    Two lines = single_line shifted by ±sep/2; by the shift theorem
    (Goodman §2.1.3) their spectrum is the single-line spectrum times
    2*cos(pi*f*sep), so intensity = envelope(x) * cos^2(pi*x*sep/(lambda*z)).
    """
    width = _snap(0.5, grid.dx)
    sep = _snap(3.0, grid.dx)
    z = 200.0
    mask = two_lines(grid.x, width, sep)
    x_obs, field, I = diff.fraunhofer_pattern(mask, grid, WAVELENGTH, z)

    envelope = diff.analytic_slit_intensity(x_obs, width, WAVELENGTH, z)
    fx = x_obs / (WAVELENGTH * z)
    interference = np.cos(np.pi * fx * sep) ** 2
    predicted = envelope * interference
    predicted = predicted / predicted.max()  # match fraunhofer_pattern's peak=1 convention

    assert np.max(np.abs(I - predicted)) < 1e-2


def test_two_lines_reduces_to_single_line_at_zero_separation(grid):
    """
    Sanity/edge case: as separation -> 0, two_lines becomes (approximately)
    a single line of the same width, so its pattern should approach the
    plain single-slit sinc^2 (cos^2 term -> 1 for all x as sep -> 0).
    """
    width = _snap(0.5, grid.dx)
    tiny_sep = grid.dx  # smallest resolvable separation
    z = 200.0
    mask = two_lines(grid.x, width, tiny_sep)
    x_obs, _, I = diff.fraunhofer_pattern(mask, grid, WAVELENGTH, z)
    I_single = diff.analytic_slit_intensity(x_obs, width, WAVELENGTH, z)

    assert np.max(np.abs(I - I_single)) < 5e-2


# ── general sanity checks on fraunhofer_pattern's output ────────────────────

def test_fraunhofer_pattern_output_shapes_and_types(grid):
    mask = single_line(grid.x, _snap(1.0, grid.dx))
    x_obs, field, I = diff.fraunhofer_pattern(mask, grid, WAVELENGTH, 200.0)
    assert x_obs.shape == (N,)
    assert field.shape == (N,)
    assert I.shape == (N,)
    assert np.iscomplexobj(field)
    assert not np.iscomplexobj(I)
    assert np.all(I >= 0)


def test_all_zero_mask_gives_zero_intensity_without_division_error(grid):
    """Edge case: an all-opaque mask (no transmission) shouldn't raise or divide by zero."""
    mask = np.zeros_like(grid.x)
    x_obs, field, I = diff.fraunhofer_pattern(mask, grid, WAVELENGTH, 200.0)
    assert np.allclose(I, 0.0)