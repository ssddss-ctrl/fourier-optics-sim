"""
tests/test_lens.py
--------------------
Unit tests for physics/lens.py.

WHY THESE SPECIFIC TESTS
--------------------------
Three things could silently be wrong in a lens/pupil module and still
"run" without crashing: (1) the NA/wavelength -> cutoff-frequency
conversion could have the focal-length cancellation wrong, (2) the pupil
filter could be applied at the wrong scale or with the wrong sign
convention, and (3) the whole chain could subtly fail to reduce to the
original mask when nothing is actually filtered. None of these produce a
Python exception -- they produce a plausible-looking but physically wrong
image. So these tests target exactly those three failure modes, using the
same hand-verified limiting cases documented in lens.py's own docstrings
(see coherent_aerial_image's "VALIDATION PERFORMED BY HAND BEFORE
DELIVERY" section) rather than re-deriving new ones.

Test gratings in this file always use pitch = k * grid.dx for integer k
(an exact number of samples per period). A non-grid-aligned pitch causes
spectral leakage into neighboring frequency bins even for a mathematically
periodic grating (documented in the Week 8 build log and re-encountered
while validating lens.py) -- that leakage is a sampling artifact of the
test grating, not a property of coherent_aerial_image, and would make the
resolution-limit tests below fail for the wrong reason.
"""

import numpy as np
import pytest

from grid import Grid1D
from masks import single_line, line_space_grating
from lens import (
    thin_lens_phase,
    pupil_radius,
    cutoff_frequency,
    lens_focal_plane_field,
    pupil_function_freq,
    coherent_aerial_image,
)
from constants import WAVELENGTH, NA_DEFAULT


# ── Fixtures / shared setup ─────────────────────────────────────────────────

@pytest.fixture
def grid():
    return Grid1D(L=20.0, N=256)


# ── thin_lens_phase (Eq. 5-10) ───────────────────────────────────────────────

def test_thin_lens_phase_unit_magnitude(grid):
    """
    Eq. 5-10 is a pure phase transformation: |t_l(x)| = 1 everywhere,
    regardless of x, focal length, or wavelength. If this ever fails, the
    phase transform has picked up an amplitude term it shouldn't have.
    """
    phase = thin_lens_phase(grid.x, focal_length=50_000.0, wavelength=WAVELENGTH)
    assert np.allclose(np.abs(phase), 1.0)


def test_thin_lens_phase_on_axis_is_unity(grid):
    """At x=0 the quadratic phase term vanishes, so t_l(0) = 1 + 0j exactly."""
    phase_at_zero = thin_lens_phase(np.array([0.0]), focal_length=50_000.0, wavelength=WAVELENGTH)
    assert np.isclose(phase_at_zero[0], 1.0 + 0.0j)


# ── pupil_radius ──────────────────────────────────────────────────────────

def test_pupil_radius_formula():
    """r_pupil = focal_length * NA, directly per the paraxial NA definition."""
    assert pupil_radius(focal_length=50_000.0, NA=0.5) == pytest.approx(25_000.0)


@pytest.mark.parametrize("bad_NA", [0.0, -0.1, 1.0, 1.5])
def test_pupil_radius_rejects_nonparaxial_NA(bad_NA):
    """NA outside (0, 1) breaks the paraxial sin~=tan approximation this
    whole module relies on and should raise, not silently return a
    physically meaningless value."""
    with pytest.raises(ValueError):
        pupil_radius(focal_length=50_000.0, NA=bad_NA)


# ── cutoff_frequency (Eq. 6-20 + NA geometry) ────────────────────────────────

def test_cutoff_frequency_matches_geometric_derivation():
    """
    cutoff_frequency's shortcut (NA/wavelength) must agree with computing
    it the "long way" via r_pupil/(wavelength*focal_length) -- this is the
    focal-length-cancellation claim made explicitly in the docstring, and
    the whole reason cutoff_frequency doesn't take focal_length as an
    argument at all. Checked at two different focal lengths to confirm
    the result really is focal-length-independent, not just correct by
    coincidence at one value.
    """
    NA, wavelength = 0.5, WAVELENGTH
    direct = cutoff_frequency(NA, wavelength)
    for f in (10_000.0, 50_000.0, 200_000.0):
        via_geometry = pupil_radius(f, NA) / (wavelength * f)
        assert via_geometry == pytest.approx(direct)


# ── lens_focal_plane_field (Eq. 5-22) ───────────────────────────────────────

def test_lens_focal_plane_field_requires_focal_length(grid):
    """focal_length has no project-wide default (unlike wavelength/NA) --
    calling without it should raise, not silently substitute something."""
    mask = single_line(grid.x, width=1.0)
    with pytest.raises(ValueError):
        lens_focal_plane_field(mask, grid, wavelength=WAVELENGTH)


def test_lens_focal_plane_field_peak_at_center_for_symmetric_mask(grid):
    """
    A single line centered at x=0 is an even function, so its exact
    Fourier transform (Eq. 5-22) must be real-valued and peak at u=0 --
    the same real-and-centered property fraunhofer_pattern relies on in
    diffraction.py. If the field peaked off-center or had a large
    imaginary part, the FFT axis alignment (fftshift ordering between the
    mask and grid.f) would be the likely culprit.
    """
    mask = single_line(grid.x, width=1.0, center=0.0)
    u, field = lens_focal_plane_field(mask, grid, wavelength=WAVELENGTH, focal_length=50_000.0)
    center_idx = np.argmin(np.abs(u))
    assert np.abs(u[center_idx]) < 1e-6  # u=0 really is on the grid
    assert np.argmax(np.abs(field)) == center_idx
    # Real-valuedness: imaginary part should be negligible relative to the peak magnitude
    assert np.max(np.abs(np.imag(field))) < 1e-6 * np.max(np.abs(field))


def test_lens_focal_plane_field_u_axis_scaling(grid):
    """u = grid.f * wavelength * focal_length -- checked directly rather
    than trusting the implementation, since a missing or extra factor of
    wavelength or focal_length here would silently mislabel every pupil-
    plane plot without breaking anything numerically downstream."""
    mask = single_line(grid.x, width=1.0)
    f = 50_000.0
    u, _ = lens_focal_plane_field(mask, grid, wavelength=WAVELENGTH, focal_length=f)
    assert np.allclose(u, grid.f * WAVELENGTH * f)


# ── pupil_function_freq (Eq. 6-20) ──────────────────────────────────────────

def test_pupil_function_freq_hard_edge(grid):
    """Pupil must be exactly 1 inside the cutoff and exactly 0 outside it
    -- a brick-wall filter, per Goodman's diffraction-limited (unaberrated)
    pupil model."""
    NA, wavelength = 0.5, WAVELENGTH
    f_cutoff = cutoff_frequency(NA, wavelength)
    P = pupil_function_freq(grid, NA=NA, wavelength=wavelength)
    passed = np.abs(grid.f) <= f_cutoff
    assert np.array_equal(P == 1.0, passed)
    assert np.array_equal(P == 0.0, ~passed)


# ── coherent_aerial_image (Eq. 5-22 + Eq. 6-20 chained) ─────────────────────

def test_full_pupil_reproduces_mask_exactly(grid):
    """
    The central sanity check for the whole pipeline: if the pupil passes
    every frequency component the grid can represent, fft1d -> multiply
    by an all-ones pupil -> ifft1d must be an exact round trip (per
    ifft_engine's guaranteed round-trip property), so the aerial image
    should reproduce the original binary mask to floating-point precision.
    An unrealistically small wavelength is used here specifically to push
    f_cutoff = NA/wavelength comfortably above grid.f_max -- this test is
    about the round-trip mechanics, not physical realism (see the
    resolution-limit tests below for physically realistic NA/wavelength).
    """
    mask = single_line(grid.x, width=1.0)
    unrealistically_small_wavelength = 0.01
    NA = 0.9
    assert cutoff_frequency(NA, unrealistically_small_wavelength) > grid.f_max  # confirm the setup

    _, intensity, P = coherent_aerial_image(mask, grid, wavelength=unrealistically_small_wavelength, NA=NA)
    assert np.all(P == 1.0)
    assert np.max(np.abs(intensity - mask)) < 1e-10


def test_aerial_image_intensity_is_nonnegative(grid):
    """Intensity is |field|^2 -- must never be negative regardless of NA."""
    mask = single_line(grid.x, width=1.0)
    _, intensity, _ = coherent_aerial_image(mask, grid, wavelength=WAVELENGTH, NA=NA_DEFAULT)
    assert np.all(intensity >= 0.0)


def test_resolution_limit_grating_unresolved_below_cutoff(grid):
    """
    A grid-aligned grating whose fundamental frequency sits ABOVE
    f_cutoff should be completely unresolved: since a 50% duty-cycle
    square wave has only odd harmonics (all >= the fundamental), none of
    its non-DC content survives, and the image should be flat at exactly
    duty_cycle^2 = 0.25 (DC term only). This is the classic diffraction-
    limited resolution cutoff, and the single most important physical
    behavior this module needs to get right.
    """
    NA, wavelength = 0.5, WAVELENGTH
    f_cutoff = cutoff_frequency(NA, wavelength)

    samples_per_period = 4  # pitch=4*dx; fundamental = 1/pitch, chosen to exceed f_cutoff below
    pitch = samples_per_period * grid.dx
    fundamental = 1.0 / pitch
    assert fundamental > f_cutoff  # confirm the setup actually tests the unresolved regime

    grating = line_space_grating(grid.x, pitch=pitch, duty_cycle=0.5)
    _, intensity, _ = coherent_aerial_image(grating, grid, wavelength=wavelength, NA=NA)

    assert np.ptp(intensity) < 1e-10          # zero modulation depth
    assert np.allclose(intensity, 0.25, atol=1e-10)  # flat at duty_cycle^2


def test_resolution_limit_grating_resolved_above_cutoff(grid):
    """Same setup as above, but with the grating's fundamental frequency
    placed comfortably BELOW f_cutoff: the image should show strong
    intensity modulation (the grating is resolved)."""
    NA, wavelength = 0.5, WAVELENGTH
    f_cutoff = cutoff_frequency(NA, wavelength)

    samples_per_period = 16  # coarser grating -> lower fundamental frequency
    pitch = samples_per_period * grid.dx
    fundamental = 1.0 / pitch
    assert fundamental < f_cutoff  # confirm the setup actually tests the resolved regime

    grating = line_space_grating(grid.x, pitch=pitch, duty_cycle=0.5)
    _, intensity, _ = coherent_aerial_image(grating, grid, wavelength=wavelength, NA=NA)

    assert np.ptp(intensity) > 0.5  # substantial modulation depth (resolved)