"""
backend/simulator.py
-----------------------
Thin wrapping layer between backend/schemas.py's request models and
physics/'s actual functions. No physics/ code is modified or reimplemented
here -- every function below just (1) builds the Grid1D/mask physics/
needs from a request model's fields, (2) calls straight through to the
existing physics/ function, and (3) converts the numpy result into plain
Python types schemas.py's response models can serialize.

Kept separate from backend/main.py (which only wires these into HTTP
routes) so this layer can be imported and unit-tested directly, without
going through FastAPI/uvicorn at all -- see tests/test_api.py, which does
exactly that alongside the HTTP-level checks.

IMPORT MECHANISM -- same fix as backend/main.py, applied here directly
------------------------------------------------------------------------
physics/ modules import each other flatly (`from masks import make_grid`),
so physics/ itself must be on sys.path, not just the repo root -- this is
done HERE (not only in backend/main.py) so backend/simulator.py works
whether it's imported via backend.main or directly (e.g. from
tests/test_api.py), matching CLAUDE.md's "any new entry point needs the
same sys.path setup" note.
"""

import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "physics"))

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

from .schemas import MaskRequest, AtfOtfRequest, AerialImageRequest, PrintedFeatureRequest


# ── numpy -> plain-Python conversion (FastAPI/Pydantic can't serialize
#    ndarrays or numpy scalar dtypes directly, and NaN isn't valid JSON) ─────

def to_json_list(arr: np.ndarray) -> List[Optional[float]]:
    """
    Convert a 1D numpy array to a plain list of Python floats, mapping NaN
    to None (JSON has no NaN literal -- Python's json module will emit the
    non-standard token `NaN` unless this conversion happens explicitly, per
    the task's own note to handle numpy-to-JSON conversion deliberately
    rather than let FastAPI's default encoder paper over it).
    """
    return [None if math.isnan(v) else float(v) for v in np.asarray(arr, dtype=float).tolist()]


def to_json_float(value: float) -> Optional[float]:
    """Same NaN -> None conversion, for a single scalar."""
    value = float(value)
    return None if math.isnan(value) else value


# ── Shared mask-building step (used by /mask, /aerial-image, /printed-feature) ─

def build_mask(req: MaskRequest) -> Tuple[Grid1D, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the Grid1D + mask/target pair for a mask request, exactly
    mirroring app/main_streamlit_archived.py's single branch on
    pattern_type (mask and target are identical arrays for both pattern
    types in the archived app -- there is no separate "target ≠ mask"
    case yet, since OPC/Week 12 is what would eventually make them differ).

    Returns
    -------
    grid, x, mask, target
    """
    grid = Grid1D(L=req.L, N=req.N)
    x = grid.x

    if req.pattern_type == "Isolated Line":
        mask = single_line(x, width=req.feature_width)
        target = single_line(x, width=req.feature_width)
    else:
        mask = line_space_grating(x, pitch=req.pitch, duty_cycle=req.duty_cycle)
        target = line_space_grating(x, pitch=req.pitch, duty_cycle=req.duty_cycle)

    return grid, x, mask, target


# ── Endpoint logic ────────────────────────────────────────────────────────────

def compute_mask(req: MaskRequest) -> dict:
    _, x, mask, target = build_mask(req)
    return {
        "x": to_json_list(x),
        "mask": to_json_list(mask),
        "target": to_json_list(target),
    }


def compute_atf_otf(req: AtfOtfRequest) -> dict:
    grid = Grid1D(L=req.L, N=req.N)
    wavelength = req.wavelength_nm / 1000.0  # physics/ works in µm throughout

    OTF, H = optical_transfer_function(grid, wavelength=wavelength, NA=req.NA,
                                        defocus_waves=req.defocus_waves)
    f0 = cutoff_frequency(req.NA, wavelength)

    # Phase of an exactly-zero pupil is undefined, not physically zero --
    # masked to NaN (-> null in the JSON response) outside the pupil
    # support, same convention as the archived Streamlit app's ATF panel.
    support = np.abs(H) > 0
    phase = np.full_like(grid.f, np.nan)
    phase[support] = np.angle(H[support])

    contrast_reversal = bool(req.defocus_waves > 0.0 and np.any(OTF.real < -1e-6))

    return {
        "fx": to_json_list(grid.f),
        "atf_magnitude": to_json_list(np.abs(H)),
        "atf_phase": to_json_list(phase),
        "otf_magnitude": to_json_list(np.abs(OTF)),
        "cutoff_frequency": float(f0),
        "contrast_reversal": contrast_reversal,
    }


def _aerial_image_intensity(req: AerialImageRequest, grid: Grid1D, mask: np.ndarray) -> np.ndarray:
    wavelength = req.wavelength_nm / 1000.0
    if req.coherence == "Coherent":
        _, intensity, _ = coherent_aerial_image(mask, grid, wavelength=wavelength, NA=req.NA,
                                                 defocus_waves=req.defocus_waves)
    else:
        intensity, _, _ = incoherent_aerial_image(mask, grid, wavelength=wavelength, NA=req.NA,
                                                   defocus_waves=req.defocus_waves)
    return intensity


def compute_aerial_image(req: AerialImageRequest) -> dict:
    grid, x, mask, _ = build_mask(req)
    intensity = _aerial_image_intensity(req, grid, mask)
    return {
        "x": to_json_list(x),
        "intensity": to_json_list(intensity),
    }


def compute_printed_feature(req: PrintedFeatureRequest) -> dict:
    grid, x, mask, target = build_mask(req)
    intensity = _aerial_image_intensity(req, grid, mask)
    printed = apply_threshold(intensity, threshold=req.threshold)

    epe, target_edges, printed_edges = edge_placement_error(target, printed, x)

    result = {
        "x": to_json_list(x),
        "target": to_json_list(target),
        "printed": to_json_list(printed),
        "epe": to_json_list(epe),
        "target_edges": to_json_list(target_edges),
        "printed_edges": to_json_list(printed_edges),
        "max_abs_epe": None,
        "mean_abs_epe": None,
        "target_linewidth": None,
        "printed_linewidth": None,
        "linewidth_error": None,
        "epe_warning": None,
        "linewidth_warning": None,
    }

    # Mirrors app/main_streamlit_archived.py's exact three-way branch:
    # no target edges at all, feature failed to print entirely, or a
    # normal case where EPE (and possibly linewidth) can be computed.
    if len(target_edges) == 0:
        result["epe_warning"] = "No edges found in target pattern."
        return result
    if np.all(np.isnan(epe)):
        result["epe_warning"] = "Feature did not print at this threshold (no printed edges found)."
        return result

    result["max_abs_epe"] = to_json_float(np.nanmax(np.abs(epe)))
    result["mean_abs_epe"] = to_json_float(np.nanmean(np.abs(epe)))

    if req.pattern_type == "Isolated Line":
        printed_w, target_w, width_err = linewidth_error(target, printed, x)
        if not math.isnan(width_err):
            result["target_linewidth"] = to_json_float(target_w)
            result["printed_linewidth"] = to_json_float(printed_w)
            result["linewidth_error"] = to_json_float(width_err)
        else:
            result["linewidth_warning"] = (
                "Printed pattern doesn't have exactly 2 edges (feature failed to "
                "resolve cleanly at this threshold/NA) -- linewidth error not "
                "well-defined here."
            )
    else:
        result["linewidth_warning"] = (
            "Linewidth error is only reported for an isolated single line (a "
            "grating has multiple ambiguous 'widths'); EPE above covers every "
            "edge in the grating."
        )

    return result
