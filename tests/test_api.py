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
