"""
tests/test_api.py
--------------------
Backend API tests: hit each of the four simulator endpoints (via FastAPI's
TestClient, no live server needed) with known inputs, and check the
response against the equivalent DIRECT physics/ call -- confirming
backend/simulator.py's wrapping (Grid1D/mask construction, numpy -> JSON
conversion) didn't lose or distort anything relative to calling physics/
directly, the same way app/main_streamlit_archived.py used to.

Not a physics test suite (that's tests/test_*.py for physics/ itself) --
this only checks the translation layer.
"""

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from grid import Grid1D
from masks import single_line, line_space_grating
from lens import coherent_aerial_image, cutoff_frequency
from imaging import (
    optical_transfer_function,
    incoherent_aerial_image,
    apply_threshold,
    edge_placement_error,
    linewidth_error,
)
from opc import edge_bias_opc
from grid2d import Grid2D
from masks2d import contact_hole_array, chip_block_layout
from lens2d import coherent_aerial_image_2d
from imaging2d import incoherent_aerial_image_2d, iou_score

from backend.main import app

client = TestClient(app)

# Shared inputs for most tests -- deliberately not the API's own defaults,
# so a bug that only shows up away from default values would be caught.
L, N = 200.0, 4096
WAVELENGTH_NM = 365.0
WAVELENGTH_UM = WAVELENGTH_NM / 1000.0
NA = 0.5
FEATURE_WIDTH = 2.0


def _assert_close_lists(actual, expected, atol=1e-9):
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        if e is None or (isinstance(e, float) and math.isnan(e)):
            assert a is None
        else:
            assert a == pytest.approx(e, abs=atol)


# ── /api/mask ────────────────────────────────────────────────────────────────

def test_mask_isolated_line_matches_direct_call():
    resp = client.post("/api/mask", json={
        "pattern_type": "Isolated Line", "feature_width": FEATURE_WIDTH, "L": L, "N": N,
    })
    assert resp.status_code == 200
    body = resp.json()

    grid = Grid1D(L=L, N=N)
    expected_mask = single_line(grid.x, width=FEATURE_WIDTH)

    _assert_close_lists(body["x"], grid.x.tolist())
    _assert_close_lists(body["mask"], expected_mask.tolist())
    _assert_close_lists(body["target"], expected_mask.tolist())


def test_mask_grating_matches_direct_call():
    resp = client.post("/api/mask", json={
        "pattern_type": "Line-Space Grating", "pitch": 2.0, "duty_cycle": 0.5, "L": L, "N": N,
    })
    assert resp.status_code == 200
    body = resp.json()

    grid = Grid1D(L=L, N=N)
    expected_mask = line_space_grating(grid.x, pitch=2.0, duty_cycle=0.5)

    _assert_close_lists(body["mask"], expected_mask.tolist())


# ── /api/aerial-image ─────────────────────────────────────────────────────────

def test_aerial_image_coherent_matches_direct_call():
    resp = client.post("/api/aerial-image", json={
        "feature_width": FEATURE_WIDTH, "L": L, "N": N,
        "wavelength_nm": WAVELENGTH_NM, "NA": NA, "coherence": "Coherent", "defocus_waves": 0.0,
    })
    assert resp.status_code == 200
    body = resp.json()

    grid = Grid1D(L=L, N=N)
    mask = single_line(grid.x, width=FEATURE_WIDTH)
    _, expected_intensity, _ = coherent_aerial_image(mask, grid, wavelength=WAVELENGTH_UM, NA=NA)

    _assert_close_lists(body["intensity"], expected_intensity.tolist())


def test_aerial_image_incoherent_with_defocus_matches_direct_call():
    defocus_waves = 1.0
    resp = client.post("/api/aerial-image", json={
        "feature_width": FEATURE_WIDTH, "L": L, "N": N,
        "wavelength_nm": WAVELENGTH_NM, "NA": NA, "coherence": "Incoherent",
        "defocus_waves": defocus_waves,
    })
    assert resp.status_code == 200
    body = resp.json()

    grid = Grid1D(L=L, N=N)
    mask = single_line(grid.x, width=FEATURE_WIDTH)
    expected_intensity, _, _ = incoherent_aerial_image(mask, grid, wavelength=WAVELENGTH_UM, NA=NA,
                                                        defocus_waves=defocus_waves)

    _assert_close_lists(body["intensity"], expected_intensity.tolist())


# ── /api/atf-otf ─────────────────────────────────────────────────────────────

def test_atf_otf_unaberrated_matches_direct_call():
    resp = client.post("/api/atf-otf", json={
        "L": L, "N": N, "wavelength_nm": WAVELENGTH_NM, "NA": NA, "defocus_waves": 0.0,
    })
    assert resp.status_code == 200
    body = resp.json()

    grid = Grid1D(L=L, N=N)
    OTF, H = optical_transfer_function(grid, wavelength=WAVELENGTH_UM, NA=NA)
    f0 = cutoff_frequency(NA, WAVELENGTH_UM)

    _assert_close_lists(body["fx"], grid.f.tolist())
    _assert_close_lists(body["atf_magnitude"], np.abs(H).tolist())
    _assert_close_lists(body["otf_magnitude"], np.abs(OTF).tolist())
    assert body["cutoff_frequency"] == pytest.approx(f0)
    assert body["contrast_reversal"] is False

    # Phase must be null exactly where the direct-call pupil is zero, and
    # equal to np.angle(H) everywhere the pupil is nonzero.
    support = np.abs(H) > 0
    for phase_val, is_supported, h_val in zip(body["atf_phase"], support, H):
        if is_supported:
            assert phase_val == pytest.approx(np.angle(h_val))
        else:
            assert phase_val is None


def test_atf_otf_large_defocus_flags_contrast_reversal():
    resp = client.post("/api/atf-otf", json={
        "L": L, "N": N, "wavelength_nm": WAVELENGTH_NM, "NA": NA, "defocus_waves": 1.5,
    })
    assert resp.status_code == 200
    body = resp.json()

    grid = Grid1D(L=L, N=N)
    OTF, _ = optical_transfer_function(grid, wavelength=WAVELENGTH_UM, NA=NA, defocus_waves=1.5)
    expected_reversal = bool(np.any(OTF.real < -1e-6))

    assert expected_reversal is True  # sanity: this defocus really does reverse contrast
    assert body["contrast_reversal"] == expected_reversal


# ── /api/printed-feature ──────────────────────────────────────────────────────

def test_printed_feature_isolated_line_matches_direct_call():
    threshold = 0.3
    resp = client.post("/api/printed-feature", json={
        "feature_width": FEATURE_WIDTH, "L": L, "N": N,
        "wavelength_nm": WAVELENGTH_NM, "NA": NA, "coherence": "Incoherent",
        "defocus_waves": 0.0, "threshold": threshold,
    })
    assert resp.status_code == 200
    body = resp.json()

    grid = Grid1D(L=L, N=N)
    mask = single_line(grid.x, width=FEATURE_WIDTH)
    intensity, _, _ = incoherent_aerial_image(mask, grid, wavelength=WAVELENGTH_UM, NA=NA)
    printed = apply_threshold(intensity, threshold=threshold)
    epe, target_edges, printed_edges = edge_placement_error(mask, printed, grid.x)
    printed_w, target_w, width_err = linewidth_error(mask, printed, grid.x)

    _assert_close_lists(body["printed"], printed.tolist())
    _assert_close_lists(body["epe"], epe.tolist())
    assert body["max_abs_epe"] == pytest.approx(np.nanmax(np.abs(epe)))
    assert body["mean_abs_epe"] == pytest.approx(np.nanmean(np.abs(epe)))
    assert body["target_linewidth"] == pytest.approx(target_w)
    assert body["printed_linewidth"] == pytest.approx(printed_w)
    assert body["linewidth_error"] == pytest.approx(width_err)
    assert body["epe_warning"] is None
    assert body["linewidth_warning"] is None


def test_printed_feature_grating_reports_linewidth_warning_not_value():
    resp = client.post("/api/printed-feature", json={
        "pattern_type": "Line-Space Grating", "pitch": 2.0, "duty_cycle": 0.5,
        "L": L, "N": N, "wavelength_nm": WAVELENGTH_NM, "NA": NA,
        "coherence": "Incoherent", "defocus_waves": 0.0, "threshold": 0.3,
    })
    assert resp.status_code == 200
    body = resp.json()

    assert body["target_linewidth"] is None
    assert body["printed_linewidth"] is None
    assert body["linewidth_warning"] is not None
    assert body["max_abs_epe"] is not None  # EPE is still reported for gratings


def test_printed_feature_severe_defocus_fails_to_print():
    # Confirmed by hand in the Week 11 build: width=1.5, NA=0.5, wavelength
    # 0.365 um, defocus_waves=2.0 pushes peak intensity below threshold=0.3
    # entirely (peak ~0.285) -- the feature genuinely does not print.
    resp = client.post("/api/printed-feature", json={
        "feature_width": 1.5, "L": L, "N": N,
        "wavelength_nm": WAVELENGTH_NM, "NA": NA, "coherence": "Incoherent",
        "defocus_waves": 2.0, "threshold": 0.3,
    })
    assert resp.status_code == 200
    body = resp.json()

    assert body["epe_warning"] == (
        "Feature did not print at this threshold (no printed edges found)."
    )
    assert body["max_abs_epe"] is None
    assert all(v is None for v in body["epe"])


# ── /api/opc ───────────────────────────────────────────────────────────────

def test_opc_matches_direct_call():
    resp = client.post("/api/opc", json={
        "feature_width": FEATURE_WIDTH, "L": L, "N": N,
        "wavelength_nm": WAVELENGTH_NM, "NA": NA, "coherence": "Incoherent",
        "defocus_waves": 0.0, "threshold": 0.3,
        "gain": 0.5, "convergence_tol": 0.01, "max_iterations": 20,
    })
    assert resp.status_code == 200
    body = resp.json()

    grid = Grid1D(L=L, N=N)
    target = single_line(grid.x, width=FEATURE_WIDTH)
    expected = edge_bias_opc(target, grid, wavelength=WAVELENGTH_UM, NA=NA,
                              coherence="Incoherent", threshold=0.3,
                              gain=0.5, convergence_tol=0.01, max_iterations=20)

    _assert_close_lists(body["target"], target.tolist())
    _assert_close_lists(body["naive_printed"], expected["naive_printed"].tolist())
    _assert_close_lists(body["corrected_mask"], expected["corrected_mask"].tolist())
    _assert_close_lists(body["corrected_printed"], expected["corrected_printed"].tolist())
    _assert_close_lists(body["naive_epe"], expected["naive_epe"].tolist())
    _assert_close_lists(body["corrected_epe"], expected["history"][-1]["epe"].tolist())
    assert body["n_iterations"] == expected["n_iterations"]
    assert body["converged"] == expected["converged"]
    assert len(body["history"]) == len(expected["history"])
    for entry, expected_entry in zip(body["history"], expected["history"]):
        assert entry["iteration"] == expected_entry["iteration"]
        assert entry["max_abs_epe"] == pytest.approx(expected_entry["max_abs_epe"], nan_ok=True)
        assert entry["mean_abs_epe"] == pytest.approx(expected_entry["mean_abs_epe"], nan_ok=True)


def test_opc_reports_improvement_for_correctable_feature():
    resp = client.post("/api/opc", json={
        "feature_width": 1.0, "L": L, "N": N,
        "wavelength_nm": WAVELENGTH_NM, "NA": NA, "coherence": "Incoherent",
        "defocus_waves": 0.0, "threshold": 0.3,
        "gain": 0.5, "convergence_tol": 0.01, "max_iterations": 20,
    })
    assert resp.status_code == 200
    body = resp.json()

    assert body["converged"] is True
    assert body["corrected_max_abs_epe"] <= body["naive_max_abs_epe"]


# ── /api/2d/mask ──────────────────────────────────────────────────────────

L2D, N2D = 8.0, 64
HOLE_DIAMETER, PITCH = 0.6, 2.0


def test_mask2d_contact_hole_array_matches_direct_call():
    resp = client.post("/api/2d/mask", json={
        "pattern_type": "Contact Hole Array", "hole_diameter": HOLE_DIAMETER, "pitch": PITCH,
        "L": L2D, "N": N2D,
    })
    assert resp.status_code == 200
    body = resp.json()

    grid = Grid2D(L=L2D, N=N2D)
    expected_mask = contact_hole_array(grid.X, grid.Y, hole_diameter=HOLE_DIAMETER, pitch=PITCH)

    assert body["x"] == pytest.approx(grid.x.tolist())
    assert body["y"] == pytest.approx(grid.y.tolist())
    for row_body, row_expected in zip(body["mask"], expected_mask.tolist()):
        assert row_body == pytest.approx(row_expected)
    for row_body, row_expected in zip(body["target"], expected_mask.tolist()):
        assert row_body == pytest.approx(row_expected)


def test_mask2d_chip_block_layout_matches_direct_call():
    resp = client.post("/api/2d/mask", json={
        "pattern_type": "Chip Block Layout", "L": L2D, "N": N2D,
    })
    assert resp.status_code == 200
    body = resp.json()

    grid = Grid2D(L=L2D, N=N2D)
    expected_mask = chip_block_layout(grid.X, grid.Y)
    for row_body, row_expected in zip(body["mask"], expected_mask.tolist()):
        assert row_body == pytest.approx(row_expected)


# ── /api/2d/simulate ──────────────────────────────────────────────────────

def test_simulate2d_contact_hole_array_matches_direct_call():
    resp = client.post("/api/2d/simulate", json={
        "pattern_type": "Contact Hole Array", "hole_diameter": HOLE_DIAMETER, "pitch": PITCH,
        "L": L2D, "N": N2D, "wavelength_nm": WAVELENGTH_NM, "NA": NA, "threshold": 0.3,
    })
    assert resp.status_code == 200
    body = resp.json()

    grid = Grid2D(L=L2D, N=N2D)
    mask = contact_hole_array(grid.X, grid.Y, hole_diameter=HOLE_DIAMETER, pitch=PITCH)
    _, expected_intensity, _ = coherent_aerial_image_2d(mask, grid, wavelength=WAVELENGTH_UM, NA=NA)
    expected_score, _ = iou_score(mask, mask)  # target==mask for this pattern_type (no OPC yet)

    assert body["x"] == pytest.approx(grid.x.tolist())
    assert body["y"] == pytest.approx(grid.y.tolist())
    for row_body, row_expected in zip(body["mask"], mask.tolist()):
        assert row_body == pytest.approx(row_expected)
    for row_body, row_expected in zip(body["aerial_intensity"], expected_intensity.tolist()):
        assert row_body == pytest.approx(row_expected, abs=1e-9)
    assert body["cutoff_frequency"] == pytest.approx(cutoff_frequency(NA, WAVELENGTH_UM))


def test_simulate2d_incoherent_matches_direct_call():
    resp = client.post("/api/2d/simulate", json={
        "pattern_type": "Contact Hole Array", "hole_diameter": HOLE_DIAMETER, "pitch": PITCH,
        "L": L2D, "N": N2D, "wavelength_nm": WAVELENGTH_NM, "NA": NA,
        "coherence": "Incoherent", "threshold": 0.3,
    })
    assert resp.status_code == 200
    body = resp.json()

    grid = Grid2D(L=L2D, N=N2D)
    mask = contact_hole_array(grid.X, grid.Y, hole_diameter=HOLE_DIAMETER, pitch=PITCH)
    expected_intensity, _, _ = incoherent_aerial_image_2d(mask, grid, wavelength=WAVELENGTH_UM, NA=NA)

    for row_body, row_expected in zip(body["aerial_intensity"], expected_intensity.tolist()):
        assert row_body == pytest.approx(row_expected, abs=1e-9)


def test_simulate2d_chip_block_layout_pattern_runs_end_to_end():
    resp = client.post("/api/2d/simulate", json={
        "pattern_type": "Chip Block Layout",
        "L": L2D, "N": N2D, "wavelength_nm": WAVELENGTH_NM, "NA": NA, "threshold": 0.3,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["fidelity_score"] is not None
    assert body["fidelity_warning"] is None


def test_simulate2d_wide_open_pupil_prints_target_exactly():
    """Same wide-open-pupil limiting case the 1D endpoints/physics tests
    use: with a cutoff far above the grid's Nyquist frequency, the printed
    pattern must equal the target exactly and fidelity_score must be
    exactly 1.0 -- a real hand-verified end-to-end check, not a shape/type
    check."""
    resp = client.post("/api/2d/simulate", json={
        "pattern_type": "Contact Hole Array", "hole_diameter": 1.0, "pitch": 2.0,
        "L": L2D, "N": N2D, "wavelength_nm": 0.001, "NA": 0.99, "threshold": 0.3,
    })
    assert resp.status_code == 200
    body = resp.json()

    for row_printed, row_target in zip(body["printed"], body["target"]):
        assert row_printed == row_target
    assert body["fidelity_score"] == pytest.approx(1.0)
    assert body["fidelity_warning"] is None


# ── Response validation (Pydantic / numpy-to-JSON conversion) ────────────────

def test_responses_contain_no_raw_numpy_types():
    """Every numeric leaf in each response must be a plain Python float/int/
    bool/None -- not a numpy scalar (which FastAPI's default JSON encoder
    cannot serialize, per the task's own note to handle this explicitly)."""
    resp = client.post("/api/atf-otf", json={"L": L, "N": 64, "wavelength_nm": WAVELENGTH_NM, "NA": NA})
    body = resp.json()
    for value in body["atf_magnitude"] + body["otf_magnitude"]:
        assert type(value) is float
    for value in body["fx"]:
        assert type(value) is float
