"""
tests/test_imaging.py
------------------------
Unit tests for physics/imaging.py: ATF/OTF construction, coherent vs.
incoherent aerial imaging, thresholding, and print-error quantification
(edge placement error, linewidth error).

Test organization mirrors tests/test_lens.py: physical-invariant checks
first (properties the OTF must satisfy regardless of implementation
detail), then closed-form/analytic comparisons, then the engineering
(non-Goodman) thresholding/EPE/linewidth utilities, each validated with a
hand-traceable synthetic case.
"""

import numpy as np
import pytest

from grid import Grid1D
from masks import single_line
from constants import WAVELENGTH, NA_DEFAULT
from lens import coherent_aerial_image, cutoff_frequency, pupil_function_freq
from imaging import (
    amplitude_point_spread_function,
    optical_transfer_function,
    incoherent_aerial_image,
    apply_threshold,
    find_edges,
    edge_placement_error,
    linewidth_error,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def grid():
    # L=200 um, N=4096 matches the values used during hand-verification
    # of this module before delivery, so any regression here reproduces
    # the exact numbers already checked by hand.
    return Grid1D(L=200.0, N=4096)


# ── ATF / amplitude PSF ───────────────────────────────────────────────────────

def test_amplitude_psf_matches_pupil_function(grid):
    """amplitude_point_spread_function's H must be identical to
    lens.pupil_function_freq's own output -- it's supposed to be a direct
    reuse, not a re-derivation."""
    h, H = amplitude_point_spread_function(grid, wavelength=WAVELENGTH, NA=NA_DEFAULT)
    H_direct = pupil_function_freq(grid, NA=NA_DEFAULT, wavelength=WAVELENGTH)
    assert np.allclose(H, H_direct)


def test_amplitude_psf_shape(grid):
    h, H = amplitude_point_spread_function(grid)
    assert h.shape == (grid.N,)
    assert H.shape == (grid.N,)


# ── OTF: physical invariants (Goodman Sec. 6.3.2, "General Properties of the OTF") ─

def test_otf_dc_is_exactly_one(grid):
    """Property 1: OTF is always unity at zero frequency (Eq. 6-28 with
    fx=fy=0 substituted directly)."""
    OTF, _ = optical_transfer_function(grid, wavelength=WAVELENGTH, NA=NA_DEFAULT)
    dc_index = int(np.argmin(np.abs(grid.f)))
    assert OTF[dc_index] == pytest.approx(1.0 + 0.0j)


def test_otf_never_exceeds_one(grid):
    """Property 3 (proved via Schwarz's inequality, Eq. 6-29): |OTF(fx)|
    must never exceed its zero-frequency value of 1, for ANY frequency."""
    OTF, _ = optical_transfer_function(grid, wavelength=WAVELENGTH, NA=NA_DEFAULT)
    assert np.max(np.abs(OTF)) == pytest.approx(1.0, abs=1e-9)
    assert np.all(np.abs(OTF) <= 1.0 + 1e-9)


def test_otf_is_real_for_unaberrated_pupil(grid):
    """An unaberrated system's OTF must be real (Sec. 6.4.3: only
    aberrations, i.e. a nonzero wavefront error W, introduce an imaginary/
    negative-going part). This pupil carries no phase term, so any
    imaginary component should be floating-point noise only."""
    OTF, _ = optical_transfer_function(grid, wavelength=WAVELENGTH, NA=NA_DEFAULT)
    assert np.max(np.abs(np.imag(OTF))) < 1e-9


def test_otf_is_even_symmetric(grid):
    """The OTF of a real, symmetric (even) pupil must itself be even:
    OTF(-fx) == OTF(fx). Follows from H(fx) being real and even (a
    symmetric hard-edged pupil) combined with the autocorrelation
    relation, Eq. 6-28.

    Note: grid.f (np.fft.fftfreq convention, even N) has one unmatched
    bin at the negative-Nyquist end (verified directly: grid.f[0] has no
    positive-frequency counterpart, while grid.f[1:] is exactly
    symmetric) -- this is a standard FFT indexing artifact, not a
    physical asymmetry, so it's excluded from this comparison rather than
    loosening the tolerance and risking hiding a real bug elsewhere.
    """
    OTF, _ = optical_transfer_function(grid, wavelength=WAVELENGTH, NA=NA_DEFAULT)
    interior = OTF[1:]
    assert np.allclose(interior, interior[::-1], atol=1e-9)


# ── OTF: closed-form comparison (Goodman Eq. 6-31, square pupil) ─────────────

def test_otf_matches_analytic_triangle_function(grid):
    """For a square (1D) pupil, Goodman's closed-form result (Eq. 6-31) is
    OTF(fx) = Lambda(fx / 2*f0), a triangle function, where f0 is the
    COHERENT cutoff frequency. Verified by hand before delivery (see
    optical_transfer_function's docstring); re-checked here as a
    regression test with a looser tolerance to account for grid
    discretization."""
    OTF, _ = optical_transfer_function(grid, wavelength=WAVELENGTH, NA=NA_DEFAULT)
    f0 = cutoff_frequency(NA_DEFAULT, WAVELENGTH)

    fn = np.abs(grid.f) / (2.0 * f0)
    analytic = np.clip(1.0 - fn, 0.0, None)

    assert np.max(np.abs(np.real(OTF) - analytic)) < 5e-3


def test_incoherent_cutoff_is_twice_coherent_cutoff(grid):
    """Eq. 6-31's stated result: the incoherent (OTF) cutoff occurs at
    2*f0, twice the coherent (ATF) cutoff f0 -- NOT because incoherent
    imaging resolves finer features (Goodman's own footnote 8 warns
    against that reading), just because the OTF's support is twice as
    wide in frequency."""
    OTF, _ = optical_transfer_function(grid, wavelength=WAVELENGTH, NA=NA_DEFAULT)
    f0 = cutoff_frequency(NA_DEFAULT, WAVELENGTH)

    passband = grid.f[np.abs(OTF) > 1e-3]
    numeric_cutoff = np.max(np.abs(passband))

    assert numeric_cutoff == pytest.approx(2 * f0, rel=0.02)


# ── Incoherent aerial image ───────────────────────────────────────────────────

def test_incoherent_wide_open_pupil_reproduces_mask(grid):
    """Same limiting-case check lens.coherent_aerial_image already
    performs for the coherent path: with no frequency content removed,
    the incoherent image must reduce exactly to the original mask."""
    mask = single_line(grid.x, width=2.0, center=0.0)

    # Monkeypatch-free wide-open check: use an NA/wavelength combination
    # whose cutoff safely exceeds grid.f_max, matching lens.py's own
    # "pupil forced wide open" convention.
    wide_wavelength = 1e-6
    intensity, OTF, H = incoherent_aerial_image(mask, grid, wavelength=wide_wavelength, NA=0.99)
    assert np.max(np.abs(intensity - mask)) < 1e-6


def test_incoherent_image_preserves_object_mean(grid):
    """DC/energy conservation: the OTF's zero-frequency normalization
    (OTF(0)=1) must preserve the object's average intensity exactly
    (Eq. 6-24's G_i(0)=G_o(0) normalization), regardless of NA."""
    mask = single_line(grid.x, width=2.0, center=0.0)
    intensity, _, _ = incoherent_aerial_image(mask, grid, wavelength=WAVELENGTH, NA=NA_DEFAULT)
    assert np.mean(intensity) == pytest.approx(np.mean(mask), abs=1e-9)


def test_incoherent_image_is_nonnegative(grid):
    """Incoherent image intensity, like the coherent path's, must never
    go negative -- it's a physical intensity."""
    mask = single_line(grid.x, width=2.0, center=0.0)
    intensity, _, _ = incoherent_aerial_image(mask, grid, wavelength=WAVELENGTH, NA=NA_DEFAULT)
    assert np.all(intensity >= -1e-9)


def test_incoherent_differs_from_coherent(grid):
    """Sanity check that the two imaging paths are not accidentally
    identical (e.g. from a copy-paste bug reusing the ATF instead of the
    OTF) -- they should differ noticeably for a feature near the
    resolution limit, where the coherent path's ringing and the
    incoherent path's smooth rolloff diverge most."""
    mask = single_line(grid.x, width=0.8, center=0.0)
    _, intensity_c, _ = coherent_aerial_image(mask, grid, wavelength=WAVELENGTH, NA=NA_DEFAULT)
    intensity_i, _, _ = incoherent_aerial_image(mask, grid, wavelength=WAVELENGTH, NA=NA_DEFAULT)
    assert np.max(np.abs(intensity_c - intensity_i)) > 0.05


# ── Thresholding ──────────────────────────────────────────────────────────────

def test_apply_threshold_basic():
    """Hand-traceable case: values >= threshold print (1.0), values below
    do not (0.0)."""
    intensity = np.array([0.0, 0.1, 0.29, 0.3, 0.31, 0.9, 1.2])
    printed = apply_threshold(intensity, threshold=0.3)
    assert np.array_equal(printed, np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]))


def test_apply_threshold_shape_and_dtype(grid):
    mask = single_line(grid.x, width=2.0, center=0.0)
    intensity, _, _ = incoherent_aerial_image(mask, grid, wavelength=WAVELENGTH, NA=NA_DEFAULT)
    printed = apply_threshold(intensity, threshold=0.3)
    assert printed.shape == intensity.shape
    assert set(np.unique(printed)).issubset({0.0, 1.0})


# ── Edge finding ──────────────────────────────────────────────────────────────

def test_find_edges_exact_grid_aligned_transition():
    """Hand-traceable case: a step at exactly the midpoint between two
    samples should be located there by linear interpolation, to
    floating-point precision."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    pattern = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    edges = find_edges(pattern, x)
    # transition between x=1.0 (value 0) and x=2.0 (value 1): crosses 0.5
    # exactly halfway, at x=1.5
    assert len(edges) == 1
    assert edges[0] == pytest.approx(1.5)


def test_find_edges_subpixel_interpolation_off_center():
    """Hand-traceable case with an asymmetric crossing: values 0.2 -> 0.8
    between x=0 and x=1 should place the 0.5-crossing at frac=(0.5-0.2)/
    (0.8-0.2)=0.5 -- i.e. still at the midpoint here by construction, so
    additionally check a genuinely off-center case."""
    x = np.array([0.0, 1.0])
    pattern = np.array([0.0, 1.0])
    edges = find_edges(pattern, x)
    assert edges[0] == pytest.approx(0.5)

    x2 = np.array([0.0, 1.0])
    pattern2 = np.array([0.4, 1.0])
    # frac = (0.5-0.4)/(1.0-0.4) = 0.1/0.6 = 0.1667
    edges2 = find_edges(pattern2, x2)
    assert edges2[0] == pytest.approx(0.1 / 0.6)


def test_find_edges_returns_none_for_flat_pattern():
    x = np.linspace(0, 10, 11)
    pattern = np.zeros_like(x)
    edges = find_edges(pattern, x)
    assert len(edges) == 0


def test_find_edges_isolated_line_gives_two_edges(grid):
    mask = single_line(grid.x, width=2.0, center=0.0)
    edges = find_edges(mask, grid.x)
    assert len(edges) == 2
    assert edges[0] == pytest.approx(-1.0, abs=grid.dx)
    assert edges[1] == pytest.approx(1.0, abs=grid.dx)


# ── Edge placement error ──────────────────────────────────────────────────────

def test_epe_zero_when_printed_matches_target_exactly(grid):
    """Hand-traceable case: if the 'printed' pattern IS the target
    pattern, every EPE must be exactly zero."""
    mask = single_line(grid.x, width=2.0, center=0.0)
    epe, target_edges, printed_edges = edge_placement_error(mask, mask, grid.x)
    assert np.allclose(epe, 0.0)
    assert len(target_edges) == len(printed_edges) == 2


def test_epe_sign_convention_outward_shift():
    """Hand-traceable case: printed edges shifted outward (line printed
    wider) should give a NEGATIVE EPE at the left edge and POSITIVE EPE
    at the right edge, matching printed_edge - target_edge."""
    x = np.linspace(-5, 5, 2001)
    target = ((x >= -1.0) & (x <= 1.0)).astype(float)
    printed = ((x >= -1.2) & (x <= 1.2)).astype(float)
    epe, target_edges, printed_edges = edge_placement_error(target, printed, x)
    assert epe[0] == pytest.approx(-0.2, abs=0.01)   # left edge moved to -1.2
    assert epe[1] == pytest.approx(0.2, abs=0.01)    # right edge moved to +1.2


def test_epe_nan_when_feature_fails_to_print():
    """If the printed pattern has no edges at all (feature never crossed
    threshold), EPE must be NaN rather than raising or silently returning
    a misleading number."""
    x = np.linspace(-5, 5, 501)
    target = ((x >= -1.0) & (x <= 1.0)).astype(float)
    printed = np.zeros_like(x)
    epe, target_edges, printed_edges = edge_placement_error(target, printed, x)
    assert len(printed_edges) == 0
    assert np.all(np.isnan(epe))


# ── Linewidth error ────────────────────────────────────────────────────────────

def test_linewidth_error_isolated_line_hand_traced():
    """Hand-traceable case: target width 2.0 um, printed width 2.4 um ->
    linewidth error should be exactly +0.4 um."""
    x = np.linspace(-5, 5, 2001)
    target = ((x >= -1.0) & (x <= 1.0)).astype(float)
    printed = ((x >= -1.2) & (x <= 1.2)).astype(float)
    printed_width, target_width, width_error = linewidth_error(target, printed, x)
    assert target_width == pytest.approx(2.0, abs=0.02)
    assert printed_width == pytest.approx(2.4, abs=0.02)
    assert width_error == pytest.approx(0.4, abs=0.02)


def test_linewidth_error_nan_on_edge_count_mismatch():
    """If the printed pattern breaks up (e.g. 4 edges from ringing near
    the resolution limit) rather than staying a single isolated line,
    linewidth_error must return NaN rather than guessing which pair of
    edges to use."""
    x = np.linspace(-5, 5, 2001)
    target = ((x >= -1.0) & (x <= 1.0)).astype(float)
    # two disjoint printed segments -> 4 edges
    printed = (((x >= -1.2) & (x <= -0.5)) | ((x >= 0.5) & (x <= 1.2))).astype(float)
    printed_width, target_width, width_error = linewidth_error(target, printed, x)
    assert np.isnan(printed_width)
    assert np.isnan(width_error)


def test_linewidth_error_end_to_end_two_feature_sizes(grid):
    """Definition of Done check: EPE and linewidth error must be
    computable for at least two feature sizes, end to end through the
    real aerial-image + threshold pipeline (not just synthetic patterns)."""
    for width in (2.0, 0.8):
        mask = single_line(grid.x, width=width, center=0.0)
        intensity, _, _ = incoherent_aerial_image(mask, grid, wavelength=WAVELENGTH, NA=NA_DEFAULT)
        printed = apply_threshold(intensity, threshold=0.3)
        epe, target_edges, printed_edges = edge_placement_error(mask, printed, grid.x)
        printed_width, target_width, width_error = linewidth_error(mask, printed, grid.x)
        assert len(epe) == 2
        assert not np.any(np.isnan(epe))
        assert not np.isnan(width_error)