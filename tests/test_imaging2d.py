"""
tests/test_imaging2d.py
---------------------------
Unit tests for physics/imaging2d.py: the 2D ATF/OTF/incoherent-imaging
path, iou_score, plus a passthrough confirmation that imaging.apply_threshold
(reused unchanged) works correctly on a 2D array.

Test organization mirrors tests/test_imaging.py: OTF physical-invariant
checks first, then the incoherent aerial image's limiting cases, then
iou_score.
"""

import numpy as np
import pytest

from grid2d import Grid2D
from masks2d import contact_hole_array
from constants import WAVELENGTH, NA_DEFAULT
from lens2d import coherent_aerial_image_2d, pupil_function_freq_2d
from imaging import apply_threshold
from imaging2d import (
    amplitude_point_spread_function_2d,
    optical_transfer_function_2d,
    incoherent_aerial_image_2d,
    iou_score,
)


@pytest.fixture
def grid():
    return Grid2D(L=8.0, N=128)


# ── ATF / amplitude PSF (2D) ─────────────────────────────────────────────────

def test_amplitude_psf_2d_matches_pupil_function(grid):
    """amplitude_point_spread_function_2d's H must be identical to
    lens2d.pupil_function_freq_2d's own output -- reuse, not re-derivation."""
    h, H = amplitude_point_spread_function_2d(grid, wavelength=WAVELENGTH, NA=NA_DEFAULT)
    H_direct = pupil_function_freq_2d(grid, NA=NA_DEFAULT, wavelength=WAVELENGTH)
    assert np.allclose(H, H_direct)
    assert h.shape == (grid.N, grid.N)


# ── OTF: physical invariants (2D generalization of Goodman Sec. 6.3.2) ──────

def test_otf_2d_dc_is_exactly_one(grid):
    OTF, _ = optical_transfer_function_2d(grid, wavelength=WAVELENGTH, NA=NA_DEFAULT)
    dc_index = int(np.argmin(np.abs(grid.fx)))
    assert OTF[dc_index, dc_index] == pytest.approx(1.0 + 0.0j)


def test_otf_2d_never_exceeds_one(grid):
    OTF, _ = optical_transfer_function_2d(grid, wavelength=WAVELENGTH, NA=NA_DEFAULT)
    assert np.max(np.abs(OTF)) == pytest.approx(1.0, abs=1e-9)
    assert np.all(np.abs(OTF) <= 1.0 + 1e-9)


def test_otf_2d_is_real_no_aberration_path_exists(grid):
    """This 2D extension has no aberration/defocus path at all -- the
    pupil is always real, so the OTF must be real to numerical precision
    on every call, not just for a specific unaberrated case."""
    OTF, _ = optical_transfer_function_2d(grid, wavelength=WAVELENGTH, NA=NA_DEFAULT)
    assert np.max(np.abs(np.imag(OTF))) < 1e-9


# ── Incoherent 2D aerial image ───────────────────────────────────────────────

def test_incoherent_2d_wide_open_pupil_reproduces_mask(grid):
    mask = contact_hole_array(grid.X, grid.Y, hole_diameter=1.0, pitch=2.0)
    intensity, OTF, H = incoherent_aerial_image_2d(mask, grid, wavelength=1e-6, NA=0.99)
    assert np.max(np.abs(intensity - mask)) < 1e-6


def test_incoherent_2d_preserves_object_mean(grid):
    mask = contact_hole_array(grid.X, grid.Y, hole_diameter=1.0, pitch=2.0)
    intensity, _, _ = incoherent_aerial_image_2d(mask, grid, wavelength=WAVELENGTH, NA=NA_DEFAULT)
    assert np.mean(intensity) == pytest.approx(np.mean(mask), abs=1e-9)


def test_incoherent_2d_is_nonnegative(grid):
    mask = contact_hole_array(grid.X, grid.Y, hole_diameter=1.0, pitch=2.0)
    intensity, _, _ = incoherent_aerial_image_2d(mask, grid, wavelength=WAVELENGTH, NA=NA_DEFAULT)
    assert np.all(intensity >= -1e-9)


def test_incoherent_2d_differs_from_coherent(grid):
    """Sanity check the two 2D imaging paths aren't accidentally identical
    (e.g. a copy-paste bug reusing the ATF instead of the OTF) -- hand-
    verified max abs difference ~1.16 for this hole/pitch/NA combination,
    well above a noise-level threshold."""
    mask = contact_hole_array(grid.X, grid.Y, hole_diameter=1.0, pitch=2.0)
    wavelength, NA = 0.365, 0.5
    _, intensity_c, _ = coherent_aerial_image_2d(mask, grid, wavelength=wavelength, NA=NA)
    intensity_i, _, _ = incoherent_aerial_image_2d(mask, grid, wavelength=wavelength, NA=NA)
    assert np.max(np.abs(intensity_c - intensity_i)) > 0.1


# ── apply_threshold on 2D arrays (reused unchanged -- plumbing check only) ──

def test_apply_threshold_works_on_2d_array():
    """imaging.apply_threshold is not reimplemented here -- this just
    confirms, per this file's own docstring claim, that the existing
    elementwise function works correctly given a 2D array without any
    modification."""
    intensity = np.array([[0.0, 0.2, 0.35], [0.29, 0.3, 0.9]])
    printed = apply_threshold(intensity, threshold=0.3)
    expected = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    assert np.array_equal(printed, expected)
    assert printed.shape == intensity.shape


# ── iou_score ─────────────────────────────────────────────────────────────

def test_iou_identical_patterns_is_one():
    pattern = np.array([[1.0, 0.0], [0.0, 1.0]])
    score, warning = iou_score(pattern, pattern)
    assert score == pytest.approx(1.0)
    assert warning is None


def test_iou_disjoint_patterns_is_zero():
    target = np.array([[1.0, 0.0], [0.0, 0.0]])
    printed = np.array([[0.0, 1.0], [0.0, 0.0]])
    score, warning = iou_score(target, printed)
    assert score == pytest.approx(0.0)
    assert warning is None


def test_iou_partial_overlap_hand_computed():
    """
    Hand-traceable 4x4 case:
      target:  a 2x2 block (top-left quadrant) -> 4 "on" pixels
      printed: a 2x2 block shifted by 1 pixel right and down, overlapping
               the target in exactly 1 pixel (bottom-right of the target
               block == top-left of the printed block)
      intersection = 1, union = 4 + 4 - 1 = 7 -> IoU = 1/7
    """
    target = np.zeros((4, 4))
    target[0:2, 0:2] = 1.0  # rows 0-1, cols 0-1

    printed = np.zeros((4, 4))
    printed[1:3, 1:3] = 1.0  # rows 1-2, cols 1-2 (overlaps target at [1,1] only)

    score, warning = iou_score(target, printed)
    assert score == pytest.approx(1.0 / 7.0)
    assert warning is None


def test_iou_over_exposure_is_penalized_not_gamed():
    """An all-white 'printed' pattern must NOT score 1.0 against a
    non-empty target -- IoU's union term must penalize over-exposure,
    unlike a plain intersection/target_area fraction would."""
    target = np.zeros((4, 4))
    target[0, 0] = 1.0  # a single "on" pixel
    printed = np.ones((4, 4))  # everything printed

    score, warning = iou_score(target, printed)
    assert score == pytest.approx(1.0 / 16.0)  # intersection=1, union=16
    assert score < 1.0
    assert warning is None


def test_iou_both_empty_returns_nan_and_warning():
    target = np.zeros((3, 3))
    printed = np.zeros((3, 3))
    score, warning = iou_score(target, printed)
    assert np.isnan(score)
    assert warning is not None
