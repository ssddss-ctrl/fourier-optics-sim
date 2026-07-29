"""
backend/schemas.py
---------------------
Pydantic request/response models for the FastAPI endpoints in backend/main.py.

These describe the HTTP contract only -- no physics/ imports here, and no
computation. Most field defaults mirror app/main_streamlit_archived.py's
widget defaults exactly (wavelength_nm=193.0, NA=0.75, threshold=0.3,
L=10.0, N=1024, feature_width=1.0, pitch=2.0, duty_cycle=0.5) so a client
that omits a field gets the same starting point the old Streamlit sidebar
did. `defocus_waves` is the one deliberate exception: 0.0 (the Streamlit
app's actual default) gives a "Good" match out of the box across every
reasonable NA/wavelength/width combination on the frontend's own
classifyMatch thresholds, leaving no room for a first-time user to learn
anything by tuning. 0.8 waves instead gives a clean, recoverable "Decent"
result -- fixable via the frontend's Advanced options panel -- so this
field's default is mirrored to the frontend's own initial state
(frontend/src/pages/Simulator.tsx) rather than the archived app's widget.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

PatternType = Literal["Isolated Line", "Line-Space Grating"]
CoherenceMode = Literal["Coherent", "Incoherent"]


# ── Shared parameter groups (mixed into request models below) ───────────────

class GridParams(BaseModel):
    L: float = Field(10.0, gt=0, description="Field width, µm")
    N: int = Field(1024, description="Grid points (matches Grid1D's N)")


class MaskParams(GridParams):
    pattern_type: PatternType = "Isolated Line"
    feature_width: float = Field(1.0, gt=0, description="Isolated-line width, µm (pattern_type='Isolated Line')")
    pitch: float = Field(2.0, gt=0, description="Grating pitch, µm (pattern_type='Line-Space Grating')")
    duty_cycle: float = Field(0.5, gt=0, lt=1, description="Grating duty cycle (pattern_type='Line-Space Grating')")


class OpticalParams(BaseModel):
    wavelength_nm: float = Field(193.0, gt=0, description="Wavelength, nm (converted to µm before calling physics/)")
    NA: float = Field(0.75, gt=0, description="Numerical aperture")
    defocus_waves: float = Field(0.8, description="Peak defocus wavefront error, in units of wavelength")


# ── Request models ───────────────────────────────────────────────────────────

class MaskRequest(MaskParams):
    pass


class AtfOtfRequest(GridParams, OpticalParams):
    pass


class AerialImageRequest(MaskParams, OpticalParams):
    coherence: CoherenceMode = "Coherent"


class PrintedFeatureRequest(AerialImageRequest):
    threshold: float = Field(0.3, gt=0, lt=1, description="Resist threshold, fraction of clear-field intensity")


class SpectrumPipelineRequest(AerialImageRequest):
    pass


class OpcRequest(PrintedFeatureRequest):
    gain: float = Field(0.5, gt=0, description="Edge-bias damping factor per iteration (gain<1 for stability)")
    convergence_tol: float = Field(0.01, gt=0, description="Convergence threshold on |EPE|, µm")
    max_iterations: int = Field(20, gt=0, description="Hard cap on forward-model passes")


# ── 2D extension (Week 12 addendum -- see physics/grid2d.py, masks2d.py, ────
#    lens2d.py, imaging2d.py). Coherent-only, no defocus/aberrations, no
#    OPC -- see docs/physics_assumptions.md's "2D Extension Assumptions"
#    section for the full list of deliberately scoped-out boundaries.

Pattern2DType = Literal["Contact Hole Array", "Chip Block Layout"]


class GridParams2D(BaseModel):
    L: float = Field(10.0, gt=0, description="Square field width, µm (both axes)")
    N: int = Field(128, gt=0, description="Grid points per axis (square grid, N*N total points)")


class Mask2DParams(GridParams2D):
    pattern_type: Pattern2DType = "Contact Hole Array"
    hole_diameter: float = Field(0.6, gt=0, description="Hole diameter, µm (pattern_type='Contact Hole Array')")
    pitch: float = Field(1.5, gt=0, description="Hole pitch, µm (pattern_type='Contact Hole Array')")


class Optical2DParams(BaseModel):
    wavelength_nm: float = Field(193.0, gt=0, description="Wavelength, nm (converted to µm before calling physics/)")
    NA: float = Field(0.75, gt=0, description="Numerical aperture")


class Simulate2DRequest(Mask2DParams, Optical2DParams):
    threshold: float = Field(0.3, gt=0, lt=1, description="Resist threshold, fraction of clear-field intensity")


class Simulate2DResponse(BaseModel):
    x: List[float]
    y: List[float]
    mask: List[List[float]]
    target: List[List[float]]
    aerial_intensity: List[List[float]]
    printed: List[List[float]]
    cutoff_frequency: float
    fidelity_score: Optional[float] = None
    fidelity_warning: Optional[str] = None


# ── Response models ──────────────────────────────────────────────────────────

class MaskResponse(BaseModel):
    x: List[float]
    mask: List[float]
    target: List[float]


class AerialImageResponse(BaseModel):
    x: List[float]
    intensity: List[float]


class AtfOtfResponse(BaseModel):
    fx: List[float]
    atf_magnitude: List[float]
    atf_phase: List[Optional[float]]  # null outside the pupil support (phase undefined there, not zero)
    otf_magnitude: List[float]
    cutoff_frequency: float
    contrast_reversal: bool


class PrintedFeatureResponse(BaseModel):
    x: List[float]
    target: List[float]
    printed: List[float]
    epe: List[Optional[float]]  # null for any target edge with no printed edge to match
    target_edges: List[float]
    printed_edges: List[float]
    max_abs_epe: Optional[float] = None
    mean_abs_epe: Optional[float] = None
    target_linewidth: Optional[float] = None
    printed_linewidth: Optional[float] = None
    linewidth_error: Optional[float] = None
    epe_warning: Optional[str] = None
    linewidth_warning: Optional[str] = None


class SpectrumPipelineResponse(BaseModel):
    x: List[float]
    mask: List[float]
    fx: List[float]
    mask_spectrum_magnitude: List[float]
    filtered_spectrum_magnitude: List[float]
    aerial_intensity: List[float]


class OpcIterationSummary(BaseModel):
    iteration: int
    max_abs_epe: Optional[float]  # null if every edge in this pass failed to print
    mean_abs_epe: Optional[float]


class OpcResponse(BaseModel):
    x: List[float]
    target: List[float]
    naive_printed: List[float]
    corrected_mask: List[float]
    corrected_printed: List[float]
    naive_epe: List[Optional[float]]
    corrected_epe: List[Optional[float]]
    naive_max_abs_epe: Optional[float]
    naive_mean_abs_epe: Optional[float]
    corrected_max_abs_epe: Optional[float]
    corrected_mean_abs_epe: Optional[float]
    history: List[OpcIterationSummary]
    n_iterations: int
    converged: bool
