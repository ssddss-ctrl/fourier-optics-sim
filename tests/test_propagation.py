"""
tests/test_propagation.py
--------------------------
Unit tests for physics/propagation.py.

Mirrors the verification approach used in test_fft_engine.py and
test_grid.py: physical sanity checks derived directly from Goodman
Eq. (3-66)-(3-70), not just "does it run."
"""

import numpy as np
import pytest

from grid import Grid1D
from masks import single_line
from propagation import (
    transfer_function,
    propagate_angular_spectrum,
    propagated_spectrum,
)


WAVELENGTH = 0.5  # µm


@pytest.fixture
def grid():
    return Grid1D(L=20.0, N=256)


@pytest.fixture
def slit(grid):
    return single_line(grid.x, width=2.0).astype(complex)


# ---------------------------------------------------------------------
# transfer_function
# ---------------------------------------------------------------------

def test_transfer_function_unity_at_z_zero(grid):
    H, is_prop = transfer_function(grid.f, WAVELENGTH, z=0.0)
    assert np.allclose(H, 1.0)


def test_transfer_function_cutoff_matches_1_over_lambda(grid):
    # Goodman Eq. (3-67): propagating iff f^2 < 1/lambda^2
    _, is_prop = transfer_function(grid.f, WAVELENGTH, z=10.0)
    cutoff = 1.0 / WAVELENGTH
    expected = np.abs(grid.f) < cutoff
    assert np.array_equal(is_prop, expected)


def test_transfer_function_unit_magnitude_when_propagating(grid):
    # Eq. (3-66): propagating components only accumulate phase, no
    # amplitude change, regardless of z.
    H, is_prop = transfer_function(grid.f, WAVELENGTH, z=15.0)
    assert np.allclose(np.abs(H[is_prop]), 1.0)


def test_transfer_function_decays_when_evanescent(grid):
    # Eq. (3-66) imaginary branch: |H| < 1 and strictly decreasing in z
    f = np.array([3.0])  # > 1/0.5 = 2.0 -> evanescent at this wavelength
    H_near, _ = transfer_function(f, WAVELENGTH, z=1.0)
    H_far, _ = transfer_function(f, WAVELENGTH, z=10.0)
    assert np.abs(H_near[0]) < 1.0
    assert np.abs(H_far[0]) < np.abs(H_near[0])


def test_transfer_function_hand_calc_propagating():
    # Hand calc at f=0: kz = 1/lambda, H = exp(j*2*pi*z*kz)
    f = np.array([0.0])
    z = 10.0
    H, is_prop = transfer_function(f, WAVELENGTH, z)
    expected = np.exp(1j * 2 * np.pi * z * (1.0 / WAVELENGTH))
    assert is_prop[0]
    assert np.allclose(H, expected)


def test_transfer_function_hand_calc_evanescent():
    # Hand calc at f=3, lambda=0.5 (cutoff=2): kappa = sqrt(3^2 - 2^2)
    f = np.array([3.0])
    z = 10.0
    H, is_prop = transfer_function(f, WAVELENGTH, z)
    kappa = np.sqrt(3.0**2 - (1.0 / WAVELENGTH) ** 2)
    expected = np.exp(-2 * np.pi * z * kappa)
    assert not is_prop[0]
    assert np.allclose(H.real, expected)  # type: ignore[reportAttributeAccessIssue]
    assert np.allclose(H.imag, 0.0)  # type: ignore[reportAttributeAccessIssue]


def test_transfer_function_z_sign_symmetry(grid):
    # Evanescent decay must use |z|: propagating the "wrong way" should
    # still attenuate evanescent content, not blow it up.
    H_pos, is_prop = transfer_function(grid.f, WAVELENGTH, z=10.0)
    H_neg, _ = transfer_function(grid.f, WAVELENGTH, z=-10.0)
    assert np.allclose(np.abs(H_pos[~is_prop]), np.abs(H_neg[~is_prop]))


# ---------------------------------------------------------------------
# propagate_angular_spectrum
# ---------------------------------------------------------------------

def test_propagate_z_zero_is_identity(grid, slit):
    out = propagate_angular_spectrum(slit, grid.dx, WAVELENGTH, z=0.0)
    assert np.max(np.abs(out - slit)) < 1e-10


def test_propagate_preserves_array_length(grid, slit):
    out = propagate_angular_spectrum(slit, grid.dx, WAVELENGTH, z=5.0)
    assert out.shape == slit.shape


def test_propagate_energy_loss_is_small_and_nonincreasing(grid, slit):
    # Stripping evanescent content can only remove energy, never add it,
    # and at z=0 energy must be exactly conserved.
    e0 = np.sum(np.abs(slit) ** 2)
    e_small_z = np.sum(
        np.abs(propagate_angular_spectrum(slit, grid.dx, WAVELENGTH, z=0.01)) ** 2
    )
    e_large_z = np.sum(
        np.abs(propagate_angular_spectrum(slit, grid.dx, WAVELENGTH, z=50.0)) ** 2
    )
    assert e_small_z <= e0 + 1e-9
    assert e_large_z <= e_small_z + 1e-9


def test_propagate_evanescent_field_vanishes_far_from_aperture():
    # A field with spectral content entirely above the propagating
    # cutoff is purely evanescent and must collapse to ~0 after a few
    # wavelengths -- this is the literal "evanescent waves carry no
    # energy away from the aperture" statement from Goodman 3.10.2.
    N, L = 256, 4.0
    grid_fine = Grid1D(L=L, N=N)
    # High spatial frequency carrier well above 1/wavelength
    carrier_f = 6.0  # cycles/um, with wavelength=0.5 -> cutoff=2.0
    field = np.exp(1j * 2 * np.pi * carrier_f * grid_fine.x)
    out = propagate_angular_spectrum(field, grid_fine.dx, WAVELENGTH, z=20.0)
    assert np.max(np.abs(out)) < 1e-6


# ---------------------------------------------------------------------
# propagated_spectrum
# ---------------------------------------------------------------------

def test_propagated_spectrum_matches_manual_fft_multiply(grid, slit):
    from fft_engine import fft1d, freq_axis

    f, spectrum_z, is_prop = propagated_spectrum(slit, grid.dx, WAVELENGTH, z=7.0)
    f_manual = freq_axis(len(slit), grid.dx)
    H_manual, is_prop_manual = transfer_function(f_manual, WAVELENGTH, 7.0)
    spectrum_manual = fft1d(slit, grid.dx) * H_manual

    assert np.array_equal(f, f_manual)
    assert np.array_equal(is_prop, is_prop_manual)
    assert np.allclose(spectrum_z, spectrum_manual)


def test_propagated_spectrum_propagating_count_matches_cutoff(grid):
    _, _, is_prop = propagated_spectrum(
        np.zeros(grid.N, dtype=complex), grid.dx, WAVELENGTH, z=1.0
    )
    cutoff = 1.0 / WAVELENGTH
    expected_count = np.sum(np.abs(grid.f) < cutoff)
    assert is_prop.sum() == expected_count