"""
tests/test_imaging2d.py
---------------------------
Unit tests for physics/imaging2d.py: iou_score, plus a passthrough
confirmation that imaging.apply_threshold (reused unchanged) works
correctly on a 2D array.
"""

import numpy as np
import pytest

from imaging import apply_threshold
from imaging2d import iou_score


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
