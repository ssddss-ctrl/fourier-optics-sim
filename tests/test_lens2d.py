"""
tests/test_lens2d.py
------------------------
Unit tests for physics/lens2d.py: circular pupil, 2D coherent aerial image,
and the closed-form Airy-disk verification that finally delivers on
docs/physics_assumptions.md's originally-planned-but-never-built Week 8
"Airy pattern" (a circular-aperture diffraction pattern).

Test organization mirrors tests/test_lens.py: plumbing checks first (pupil
cutoff, wide-open-pupil round trip), then a resolution-limit check, then
the closed-form analytic comparison.
"""

import numpy as np
import pytest
from scipy.special import j1

from grid2d import Grid2D
from masks2d import contact_hole_array
from constants import WAVELENGTH, NA_DEFAULT
from fft_engine import ifft2d
from lens import cutoff_frequency
from lens2d import pupil_function_freq_2d, coherent_aerial_image_2d


@pytest.fixture
def grid():
    return Grid2D(L=8.0, N=128)


# ── Pupil function ────────────────────────────────────────────────────────

def test_pupil_is_circular_not_square(grid):
    """A point at radius exactly between the cutoff and its diagonal
    (fx=fy=f_cutoff/sqrt(2)*1.2, safely outside the circle but which WOULD
    be inside a square pupil of half-width f_cutoff) must be excluded --
    confirms the pupil is a disk, not a square brick-wall extended to 2D."""
    P = pupil_function_freq_2d(grid, NA=NA_DEFAULT, wavelength=WAVELENGTH)
    f_cutoff = cutoff_frequency(NA_DEFAULT, WAVELENGTH)

    # Point just inside the circle, on-axis: must pass.
    on_axis_idx = np.argmin(np.abs(grid.fx - f_cutoff * 0.5))
    center_idx = np.argmin(np.abs(grid.fy))
    assert P[center_idx, on_axis_idx] == 1.0

    # Point at radius 1.3*f_cutoff along the diagonal: must be excluded even
    # though a NAIVE per-axis brick-wall check (|fx|<=f_cutoff AND
    # |fy|<=f_cutoff) could wrongly pass a diagonal point beyond the circle.
    diag_component = f_cutoff * 1.3 / np.sqrt(2)
    idx_x = np.argmin(np.abs(grid.fx - diag_component))
    idx_y = np.argmin(np.abs(grid.fy - diag_component))
    assert P[idx_y, idx_x] == 0.0


def test_pupil_is_binary(grid):
    P = pupil_function_freq_2d(grid, NA=NA_DEFAULT, wavelength=WAVELENGTH)
    assert set(np.unique(P)).issubset({0.0, 1.0})


# ── Coherent aerial image: plumbing + resolution limit ───────────────────

def test_wide_open_pupil_reproduces_mask(grid):
    """Same limiting-case check lens.coherent_aerial_image performs in 1D:
    with a cutoff far above grid.f_max, the 2D image must reduce exactly
    to the original binary mask."""
    mask = contact_hole_array(grid.X, grid.Y, hole_diameter=1.0, pitch=2.0)
    _, intensity, P = coherent_aerial_image_2d(mask, grid, wavelength=1e-6, NA=0.99)
    assert np.max(np.abs(intensity - mask)) < 1e-6


def test_resolution_limit_below_vs_above_cutoff(grid):
    """
    Hand-verified (NA=0.5, wavelength=0.365 um -> f_cutoff=1.37 cycles/um,
    on this L=8/N=128/dx=0.0625 grid): a contact-hole array with pitch=2.0
    (fundamental frequency 0.5 cycles/um, well below cutoff) shows strong
    intensity modulation (std=0.378); an otherwise-identical array with
    pitch=0.5 (fundamental frequency 2.0 cycles/um, above cutoff) shows
    EXACTLY zero modulation (std=0.0 to floating-point precision) -- only
    the DC term of its spectrum survives the pupil, exactly the coherent-
    imaging analogue of lens.py's own 1D grating resolution-limit check.
    Both pitches are grid-aligned (2.0/0.0625=32, 0.5/0.0625=8, both exact
    integers) per lens.py's own documented convention for avoiding sampling
    leakage in test gratings.
    """
    wavelength, NA = 0.365, 0.5

    mask_below = contact_hole_array(grid.X, grid.Y, hole_diameter=1.0, pitch=2.0)
    _, intensity_below, _ = coherent_aerial_image_2d(mask_below, grid, wavelength=wavelength, NA=NA)
    assert np.std(intensity_below) > 0.1

    mask_above = contact_hole_array(grid.X, grid.Y, hole_diameter=0.25, pitch=0.5)
    _, intensity_above, _ = coherent_aerial_image_2d(mask_above, grid, wavelength=wavelength, NA=NA)
    assert np.std(intensity_above) < 1e-9


# ── Airy-disk closed-form comparison (the never-delivered Week 8 pattern) ─

def _jinc(u: np.ndarray) -> np.ndarray:
    """2*J1(u)/u, with the u->0 limit (value 1.0) handled explicitly."""
    out = np.ones_like(u)
    nonzero = u != 0
    out[nonzero] = 2.0 * j1(u[nonzero]) / u[nonzero]
    return out


def test_coherent_psf_matches_analytic_airy_profile():
    """
    The coherent point-spread function of a circular pupil -- ifft2d of
    pupil_function_freq_2d -- must match Goodman's classic closed-form
    Airy/jinc profile, 2*J1(2*pi*f_cutoff*rho)/(2*pi*f_cutoff*rho), along a
    radial cut, both in normalized shape and in the location of the first
    dark ring.

    IMPORTANT INDEXING NOTE (matches a documented 1D precedent): ifft2d's
    output, like ifft1d's in imaging.amplitude_point_spread_function, comes
    out centered at array INDEX (0,0) (the DFT's natural seam point), not
    at the physical center index (N/2, N/2) where grid.X/grid.Y == 0. An
    explicit np.fft.fftshift is required to re-align it with grid.X/grid.Y's
    own indexing before taking a physical radial cut -- confirmed necessary
    by hand (without it, the profile's peak lands at index [0,0], not the
    grid's center, and comparing against grid.X-based rho values silently
    picks up the wrong sample entirely).
    """
    grid = Grid2D(L=8.0, N=256)
    wavelength, NA = 0.365, 0.5
    f_cutoff = cutoff_frequency(NA, wavelength)

    P = pupil_function_freq_2d(grid, NA=NA, wavelength=wavelength)
    h = np.fft.fftshift(ifft2d(P, grid.dx))
    assert np.max(np.abs(h.imag)) < 1e-9  # real pupil -> real PSF, to numerical precision
    h = h.real

    center = grid.N // 2
    peak = h[center, center]
    assert peak > 0  # sanity: the on-axis PSF value is the pupil's positive area, not near zero

    row = h[center, :] / peak
    rho = np.abs(grid.X[center, :])
    order = np.argsort(rho)
    rho_sorted, row_sorted = rho[order], row[order]

    analytic = _jinc(2.0 * np.pi * f_cutoff * rho_sorted)

    # Shape comparison over the region containing the central lobe and
    # first couple of side lobes (hand-verified max deviation ~0.02 here).
    within_range = rho_sorted < 2.0
    assert np.max(np.abs(row_sorted[within_range] - analytic[within_range])) < 0.05

    # First dark ring: J1's first zero is at u=3.8317, i.e.
    # rho_zero1 = 3.8317 / (2*pi*f_cutoff). The numerically computed profile
    # must be close to zero there too (hand-verified value ~0.0086).
    rho_zero1_theory = 3.8317 / (2.0 * np.pi * f_cutoff)
    nearest_idx = np.argmin(np.abs(rho_sorted - rho_zero1_theory))
    assert abs(row_sorted[nearest_idx]) < 0.05
