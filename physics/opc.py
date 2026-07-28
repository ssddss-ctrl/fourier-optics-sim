"""
physics/opc.py
-----------------
Optical proximity correction (OPC): an iterative edge-bias loop that
pre-distorts a mask so that its printed image matches the intended target,
compensating for the forward model's own diffraction-induced distortion.

WHY THIS MODULE EXISTS IN THE PIPELINE
-----------------------------------------
Every module through Week 11 answers "given a mask, what actually prints on
the wafer?" -- the FORWARD direction of the pipeline:

    mask --[lens.py/imaging.py]--> aerial image --[imaging.py threshold]-->
        printed feature --[imaging.py EPE]--> how far off is it?

That forward chain is a lossy, band-limited system (Goodman Ch. 6's pupil
cutoff): small mask features never survive it unchanged, and Week 10's EPE
metric already quantifies exactly how much a printed edge misses its target.
This module closes the loop by running the forward chain repeatedly and
using its own EPE output as an error signal to correct the mask -- the
INVERSE direction:

    target mask --[this module]--> corrected mask --[forward chain again]-->
        printed feature that (ideally) matches the ORIGINAL target

Physically, this is proximity effect correction: the printed position of a
mask edge depends on the diffraction/threshold response of nearby mask
features (not just that edge in isolation), so mask designers bias edges
outward or inward, by trial and error or simulation, to compensate. This is
the last stage of the pipeline precisely because it is the first stage that
needs every other stage already built and validated: it forward-models the
SAME mask through lens.py/imaging.py's existing coherent/incoherent path
many times per correction, and measures success with imaging.py's own EPE
metric -- reusing, not reimplementing, both.

NOT A GOODMAN EQUATION -- LITHOGRAPHY ENGINEERING PRACTICE
------------------------------------------------------------------
Goodman's Ch. 6 (Sec. 6.6.1 discusses resolution and the practical
limitations a finite-NA imaging system imposes on faithfully reproducing an
object) explains WHY a printed pattern deviates from its mask -- the pupil's
band-limiting behavior -- but says nothing about correcting for it: that is
a lithography industry practice (mask biasing / OPC), not a textbook
optics result. Section 6.6.1's discussion of resolution limits is the
physical reason this module needs to exist at all (a diffraction-limited
system's threshold response is NOT a rigid shift of the mask, so simple
edge-biasing works well for well-behaved isolated features and is only an
approximate, iterative fix for the same reason resolution enhancement in
real lithography is itself iterative and heuristic, not a closed-form
inverse of Eq. 6-20).

THE EDGE-BIAS ALGORITHM
---------------------------
1. Start from the target binary mask (the intended pattern).
2. Forward model: mask -> aerial image (lens.coherent_aerial_image or
   imaging.incoherent_aerial_image, whichever the caller selects) ->
   threshold (imaging.apply_threshold) -> printed feature.
3. Measure per-edge EPE (imaging.edge_placement_error) between the FIXED
   target and this iteration's printed feature -- not between the current
   (possibly already-biased) mask and the printed feature, since the goal
   is always to match the original design intent, not whatever the mask
   happens to look like mid-correction.
4. Bias each mask edge opposite the measured error, damped by `gain`:
   new_edge = old_edge - gain * EPE. gain < 1 damps the correction so it
   converges instead of overshooting/oscillating (a direct proportional
   step, gain=1, would in general overcorrect past the target on the very
   next iteration for a system with any residual nonlinearity between edge
   position and printed position).
5. Repeat from step 2 until every edge's |EPE| < convergence_tol or
   max_iterations is reached.

WHY EDGE POSITIONS ARE TRACKED AS A SEPARATE CONTINUOUS ARRAY, NOT
RE-DERIVED FROM THE BINARY MASK EACH ITERATION
------------------------------------------------------------------------
imaging.find_edges already locates sub-pixel edge positions in a binary
array by linear interpolation -- but the *rebuilt* mask array itself only
has grid resolution (each sample is exactly 0.0 or 1.0). Re-deriving "the
current edge position" by calling find_edges on that rebuilt array on every
iteration would round-trip the position through the grid spacing dx once
per iteration, silently discarding sub-pixel correction progress a bit at a
time. Instead, `_mask_from_edges` below builds the binary array directly
from a continuously-tracked float edge-position array on every iteration
(exactly how masks.single_line itself builds a mask from a continuous
width/center against the grid), so the only discretization loss is the one
grid spacing dx already imposes on any mask -- once, not compounded across
iterations.

Because the forward model is only ever evaluated on masks with the SAME
edge count and ordering as the target (edges are moved, never added or
removed), each target edge's EPE lines up positionally with exactly one
tracked edge to correct -- no edge-matching ambiguity beyond what
imaging.edge_placement_error already resolves via nearest-neighbor matching
against the printed pattern.

All spatial coordinates: µm. Wavelength: µm. NA: dimensionless.
"""

from typing import List, Optional

import numpy as np

from constants import WAVELENGTH, NA_DEFAULT
from lens import coherent_aerial_image
from imaging import (
    incoherent_aerial_image,
    apply_threshold,
    find_edges,
    edge_placement_error,
)


def _mask_from_edges(edges: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Rebuild a binary mask from a sorted array of alternating rising/falling
    edge positions (the inverse of imaging.find_edges).

    Parameters
    ----------
    edges : ndarray — sorted edge positions (µm): edges[0]/edges[1] is the
             first rising/falling pair, edges[2]/edges[3] the second, etc.
             (matches the alternating transition order imaging.find_edges
             returns for the dark-field masks physics/masks.py produces).
             A trailing unmatched edge (odd length) is dropped rather than
             guessed at.
    x     : ndarray — spatial grid, µm

    Returns
    -------
    mask : ndarray of 0.0/1.0, same shape as x
    """
    mask = np.zeros_like(x)
    n_pairs = len(edges) // 2
    for i in range(n_pairs):
        rising, falling = edges[2 * i], edges[2 * i + 1]
        mask[(x >= rising) & (x < falling)] = 1.0
    return mask


def _forward_print(mask: np.ndarray, grid, wavelength: float, NA: float,
                    defocus_waves: float, coherence: str, threshold: float) -> np.ndarray:
    """One forward-model pass: mask -> aerial image -> printed feature.
    Reuses lens.coherent_aerial_image / imaging.incoherent_aerial_image and
    imaging.apply_threshold exactly as backend/simulator.py's
    _aerial_image_intensity does -- no new forward physics here."""
    if coherence == "Coherent":
        _, intensity, _ = coherent_aerial_image(mask, grid, wavelength=wavelength, NA=NA,
                                                  defocus_waves=defocus_waves)
    else:
        intensity, _, _ = incoherent_aerial_image(mask, grid, wavelength=wavelength, NA=NA,
                                                    defocus_waves=defocus_waves)
    return apply_threshold(intensity, threshold=threshold)


def edge_bias_opc(target: np.ndarray, grid, wavelength: float = WAVELENGTH,
                   NA: float = NA_DEFAULT, defocus_waves: float = 0.0,
                   coherence: str = "Incoherent", threshold: float = 0.3,
                   gain: float = 0.5, convergence_tol: float = 0.01,
                   max_iterations: int = 20) -> dict:
    """
    Iterative edge-bias OPC: bias `target`'s edges so the forward-modeled
    print matches `target` as closely as possible, within `max_iterations`.

    See module docstring for the full algorithm and the Goodman 6.6.1
    connection. Every forward-model call here goes through the existing
    lens.py/imaging.py pipeline; only the edge-tracking and biasing logic
    is new.

    WHY EACH HISTORY ENTRY'S EPE DOUBLES AS BOTH "AFTER THE PREVIOUS
    ITERATION'S BIAS" AND "BEFORE THIS ITERATION'S BIAS"
    ------------------------------------------------------------------------
    history[i]["epe"] is measured on the mask as it exists at the START of
    iteration i -- i.e. AFTER iteration i-1's edge bias was applied (for
    i>0), and BEFORE iteration i's own bias is computed and applied. This
    is deliberately a single number per iteration, not a separately
    forward-modeled "before" and "after" pair: forward-modeling the
    just-biased mask a second time inside the same iteration would be
    numerically identical to simply evaluating it at the start of the next
    iteration, so doing so would double the forward-model cost for no new
    information. history[0]["epe"] is exactly the naive (uncorrected)
    EPE -- the regression case tests/test_opc.py checks directly against
    calling the forward model on `target` with no correction at all.

    Parameters
    ----------
    target           : ndarray — intended binary mask (0/1), on grid.x
    grid             : Grid1D — target's spatial/frequency grid
    wavelength       : float — wavelength, µm (defaults to constants.WAVELENGTH)
    NA               : float — numerical aperture (defaults to constants.NA_DEFAULT)
    defocus_waves    : float — peak defocus wavefront error, in waves (0.0 = none)
    coherence        : "Coherent" | "Incoherent" — which forward-imaging path
                        to run each iteration through
    threshold        : float — resist threshold (imaging.apply_threshold convention)
    gain             : float — damping factor on the edge bias, new_edge =
                        old_edge - gain * EPE. gain < 1 for stability (default
                        0.5); gain >= 1 is accepted (not clamped) so a caller
                        can deliberately explore/demonstrate non-convergent
                        behavior, but is not the recommended operating range.
    convergence_tol  : float — µm; converged once every edge's EPE is
                        non-NaN (the feature actually printed) and
                        max(|EPE|) < convergence_tol
    max_iterations   : int — hard cap on forward-model passes

    Returns
    -------
    result : dict with keys —
        target            : ndarray — the original target mask (unmodified)
        corrected_mask    : ndarray — final (possibly still-correcting) mask
        naive_printed     : ndarray — printed feature from `target` directly,
                             i.e. iteration 0, no correction applied
        corrected_printed : ndarray — printed feature from `corrected_mask`,
                             consistent with it (same forward-model call that
                             produced the last history entry -- no stale
                             pairing between mask and printed feature)
        naive_epe         : ndarray — per-edge EPE at iteration 0 (identical
                             to history[0]["epe"]; exposed directly since
                             it's the single-number-per-run summary a caller
                             most often wants for a "before" comparison)
        history           : list of dicts, one per forward-model pass, each
                             {"iteration": int, "epe": ndarray,
                              "max_abs_epe": float, "mean_abs_epe": float}
                             (max_abs_epe/mean_abs_epe are NaN if every edge
                             in that pass failed to print at all)
        n_iterations      : int — number of forward-model passes actually run
                             (== len(history))
        converged          : bool — True iff the loop stopped because every
                             edge met convergence_tol, False if max_iterations
                             was hit first
    """
    target_edges = find_edges(target, grid.x)
    current_edges = target_edges.copy()
    mask = target.copy()

    history: List[dict] = []
    naive_printed: Optional[np.ndarray] = None
    naive_epe: Optional[np.ndarray] = None
    printed = None
    converged = False

    for iteration in range(max_iterations):
        printed = _forward_print(mask, grid, wavelength, NA, defocus_waves, coherence, threshold)
        epe, _, _ = edge_placement_error(target, printed, grid.x)

        any_nan = bool(np.any(np.isnan(epe))) if epe.size else True
        max_abs_epe = float("nan") if any_nan else float(np.max(np.abs(epe)))
        mean_abs_epe = float("nan") if any_nan else float(np.mean(np.abs(epe)))
        history.append({
            "iteration": iteration,
            "epe": epe.copy(),
            "max_abs_epe": max_abs_epe,
            "mean_abs_epe": mean_abs_epe,
        })

        if iteration == 0:
            naive_printed = printed.copy()
            naive_epe = epe.copy()

        if (not any_nan) and max_abs_epe < convergence_tol:
            converged = True
            break

        if iteration == max_iterations - 1:
            # Out of iterations -- stop without applying a bias that would
            # never get forward-modeled/reported, so corrected_mask stays
            # consistent with corrected_printed/the last history entry.
            break

        bias = np.nan_to_num(epe, nan=0.0)
        current_edges = current_edges - gain * bias
        mask = _mask_from_edges(current_edges, grid.x)

    return {
        "target": target,
        "corrected_mask": mask,
        "naive_printed": naive_printed,
        "corrected_printed": printed,
        "naive_epe": naive_epe,
        "history": history,
        "n_iterations": len(history),
        "converged": converged,
    }
