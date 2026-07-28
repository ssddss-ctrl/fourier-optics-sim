"""
tests/test_opc.py
---------------------
Unit tests for physics/opc.py's edge-bias OPC loop.

Values below (grid, feature width, converged EPE numbers) were verified by
hand before writing these tests -- see the build log for the interactive
session that traced through iteration-by-iteration current_edges/epe/
printed_edges for this exact grid and mask before delivery.
"""

import numpy as np
import pytest

from grid import Grid1D
from masks import single_line
from imaging import edge_placement_error
from opc import edge_bias_opc, _forward_print


@pytest.fixture
def grid():
    # Matches tests/test_imaging.py's grid so any regression here reproduces
    # numbers already checked by hand in that module too.
    return Grid1D(L=200.0, N=4096)


WAVELENGTH = 0.365
NA = 0.5


# ── Convergence on a simple case ─────────────────────────────────────────────

def test_converges_on_isolated_line(grid):
    """A single isolated line, gain=0.5, is the textbook-stable edge-bias
    case: hand-verified to converge within 3 forward-model passes (iteration
    2's max_abs_epe hits exactly 0.0, well under convergence_tol=0.01)."""
    target = single_line(grid.x, width=1.0, center=0.0)
    result = edge_bias_opc(target, grid, wavelength=WAVELENGTH, NA=NA,
                            coherence="Incoherent", threshold=0.3,
                            gain=0.5, convergence_tol=0.01, max_iterations=20)

    assert result["converged"] is True
    assert result["n_iterations"] <= 20
    last = result["history"][-1]
    assert last["max_abs_epe"] < 0.01

    # The corrected mask's own forward-model print must independently
    # reproduce that same converged EPE -- corrected_printed isn't stale.
    printed = _forward_print(result["corrected_mask"], grid, WAVELENGTH, NA,
                              0.0, "Incoherent", 0.3)
    assert np.array_equal(printed, result["corrected_printed"])
    epe, _, _ = edge_placement_error(target, printed, grid.x)
    assert np.nanmax(np.abs(epe)) < 0.01


def test_corrected_mask_differs_from_target_when_correction_applied(grid):
    """If the naive print already had nonzero EPE, the corrected mask must
    actually have moved edges relative to the target -- otherwise no
    correction happened at all."""
    target = single_line(grid.x, width=1.0, center=0.0)
    result = edge_bias_opc(target, grid, wavelength=WAVELENGTH, NA=NA,
                            coherence="Incoherent", threshold=0.3,
                            gain=0.5, convergence_tol=0.01, max_iterations=20)
    assert np.nanmax(np.abs(result["naive_epe"])) > 0.0
    assert not np.array_equal(result["corrected_mask"], target)


# ── EPE decreasing across iterations for a stable gain ──────────────────────

def test_epe_bounded_and_decreasing_for_stable_gain(grid):
    """For a damped (gain<1) correction, mean |EPE| across the recorded
    history must never increase iteration-to-iteration, and must strictly
    improve from the naive (iteration 0) value by the time the loop stops."""
    target = single_line(grid.x, width=0.8, center=0.0)
    result = edge_bias_opc(target, grid, wavelength=WAVELENGTH, NA=NA,
                            coherence="Incoherent", threshold=0.3,
                            gain=0.5, convergence_tol=0.01, max_iterations=20)

    mean_abs_epe_series = [h["mean_abs_epe"] for h in result["history"]]
    assert len(mean_abs_epe_series) >= 2

    # Bounded: never gets worse than the previous iteration.
    for prev, cur in zip(mean_abs_epe_series, mean_abs_epe_series[1:]):
        assert cur <= prev + 1e-12

    # Strictly improved overall (naive -> final).
    assert mean_abs_epe_series[-1] < mean_abs_epe_series[0]


# ── Non-convergent / max-iterations-hit behaves sanely ───────────────────────

def test_max_iterations_hit_returns_best_effort_without_crashing(grid):
    """An unsatisfiable convergence_tol (0.0 -- grid quantization alone
    prevents EPE from ever being exactly recorded as < 0.0) must hit
    max_iterations, report converged=False, and still return a
    self-consistent, correctly-shaped result rather than raising or
    returning NaN-filled/garbage arrays."""
    target = single_line(grid.x, width=1.0, center=0.0)
    result = edge_bias_opc(target, grid, wavelength=WAVELENGTH, NA=NA,
                            coherence="Incoherent", threshold=0.3,
                            gain=0.5, convergence_tol=0.0, max_iterations=5)

    assert result["converged"] is False
    assert result["n_iterations"] == 5
    assert len(result["history"]) == 5
    assert result["corrected_mask"].shape == target.shape
    assert result["corrected_printed"].shape == target.shape
    assert set(np.unique(result["corrected_mask"])).issubset({0.0, 1.0})
    assert not np.any(np.isnan(result["corrected_mask"]))


def test_unstable_gain_does_not_converge_but_stays_well_behaved(grid):
    """A gain far outside the stable (<1) range should fail to converge
    (oscillates/overcorrects rather than settling) but must still return a
    finite, correctly-shaped result -- 'doesn't crash', not 'doesn't ever
    fail to converge'."""
    target = single_line(grid.x, width=1.0, center=0.0)
    result = edge_bias_opc(target, grid, wavelength=WAVELENGTH, NA=NA,
                            coherence="Incoherent", threshold=0.3,
                            gain=2.0, convergence_tol=0.001, max_iterations=10)

    assert result["converged"] is False
    assert result["n_iterations"] == 10
    assert result["corrected_mask"].shape == target.shape
    assert np.all(np.isfinite(result["corrected_mask"]))


# ── Regression: naive (0th-iteration) EPE matches uncorrected print error ───

def test_naive_epe_matches_direct_uncorrected_forward_model(grid):
    """history[0]/naive_epe must be identical to computing the print error
    directly from `target` with no OPC involved at all -- iteration 0 is by
    construction the uncorrected case, and this must not silently drift if
    the loop's internal bookkeeping changes."""
    target = single_line(grid.x, width=1.0, center=0.0)
    result = edge_bias_opc(target, grid, wavelength=WAVELENGTH, NA=NA,
                            coherence="Incoherent", threshold=0.3,
                            gain=0.5, convergence_tol=0.01, max_iterations=20)

    printed_direct = _forward_print(target, grid, WAVELENGTH, NA, 0.0, "Incoherent", 0.3)
    epe_direct, _, _ = edge_placement_error(target, printed_direct, grid.x)

    assert np.allclose(result["naive_epe"], epe_direct, equal_nan=True)
    assert np.allclose(result["history"][0]["epe"], epe_direct, equal_nan=True)
    assert np.array_equal(result["naive_printed"], printed_direct)
