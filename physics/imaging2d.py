"""
physics/imaging2d.py
------------------------
2D print-fidelity quantification -- the 2D counterpart to imaging.py's
thresholding/EPE/linewidth-error section, for the 2D mask -> aerial image
-> printed-feature chain (lens2d.coherent_aerial_image_2d).

WHY THIS MODULE EXISTS IN THE PIPELINE
-----------------------------------------
imaging.py closes the 1D pipeline's loop by turning an aerial image into a
printed-feature estimate (apply_threshold) and quantifying how well it
matches the target (edge_placement_error, linewidth_error). This module
does the analogous job for 2D:

    2D mask --[lens2d.py]--> 2D aerial image --[threshold]--> printed feature
    { 2D target, 2D printed feature } --[this module]--> fidelity score

`imaging.apply_threshold` IS REUSED HERE UNCHANGED -- NOT REIMPLEMENTED
------------------------------------------------------------------------
apply_threshold(intensity, threshold) is already a pure elementwise
comparison (`intensity >= threshold`) with no assumption about array
dimensionality baked in. It works correctly, with no modification at all,
on a 2D intensity array exactly as it does on a 1D one. Callers of this
2D pipeline should import apply_threshold directly from imaging.py; it is
not re-exported or wrapped here, to keep it unambiguous that this is the
exact same function, not a look-alike 2D copy.

WHY THIS MODULE DOES NOT PROVIDE A 2D EPE/LINEWIDTH-ERROR EQUIVALENT
------------------------------------------------------------------------
imaging.find_edges/edge_placement_error/linewidth_error are fundamentally
1D algorithms: they scan a single array for 0<->1 transitions along one
axis. A 2D binary pattern's "edges" are CONTOURS -- the boundaries of 2D
regions -- and measuring how far a printed contour deviates from a target
contour requires choosing gauge points along the boundary and biasing each
perpendicular to the local edge direction (this is, not coincidentally,
also exactly the missing piece for a hypothetical 2D OPC loop). That is a
genuinely different, more open-ended algorithm than a 1D array scan, not a
mechanical dimensional extension -- explicitly out of scope for this
project's 2D extension (see docs/physics_assumptions.md's "2D Extension
Assumptions" section). iou_score below is a deliberately SIMPLER stand-in
metric that answers the same practical question ("how well did it print?")
without needing any contour/edge detection at all.

All spatial coordinates: µm
"""

from typing import Optional, Tuple

import numpy as np


def iou_score(target: np.ndarray, printed: np.ndarray) -> Tuple[float, Optional[str]]:
    """
    Intersection-over-Union (Jaccard index) between a 2D printed pattern
    and its target -- this project's 2D stand-in for the 1D pipeline's
    edge-placement-error/linewidth-error metrics.

    NOT A GOODMAN EQUATION -- STANDARD IMAGE-SEGMENTATION METRIC
    ------------------------------------------------------------------
    IoU is a standard overlap metric from image segmentation/object
    detection, not a result from Fourier optics -- used here purely as an
    engineering convenience, exactly the same spirit in which imaging.py
    flags apply_threshold/edge_placement_error/linewidth_error as
    lithography-engineering conventions rather than textbook physics.

    WHY IoU, NOT A PLAIN OVERLAP FRACTION
    ------------------------------------------------------------------
    A simpler "intersection / target_area" fraction is gameable: an
    all-white, badly over-exposed `printed` pattern (printed=1 everywhere)
    would score a false 1.0 against ANY non-empty target, since every
    target pixel is trivially "covered." IoU's union term in the
    denominator correctly penalizes exactly that failure mode -- an
    over-printed pattern also enlarges the union, pulling the score down,
    not just the intersection up.

    Degenerate case (both target and printed are entirely empty, so
    union == 0): returns (NaN, warning string), mirroring imaging.py's own
    edge_placement_error/linewidth_error convention of returning NaN plus
    an explanatory string on an ill-defined case rather than silently
    guessing a value (e.g. returning 1.0 for "nothing was supposed to
    print and nothing printed" would hide a threshold/NA combination that
    is probably wrong, not a genuinely perfect match).

    Parameters
    ----------
    target  : ndarray — intended binary pattern (0/1), any shape
    printed : ndarray — thresholded printed-feature estimate (0/1), same
               shape as target

    Returns
    -------
    score   : float — intersection/union in [0.0, 1.0], or NaN if both
               patterns are entirely empty
    warning : str or None — explanatory message when score is NaN, else None
    """
    target_on = target > 0.5
    printed_on = printed > 0.5

    intersection = np.logical_and(target_on, printed_on).sum()
    union = np.logical_or(target_on, printed_on).sum()

    if union == 0:
        return float("nan"), "Target and printed pattern are both empty -- IoU is undefined."

    return float(intersection) / float(union), None
