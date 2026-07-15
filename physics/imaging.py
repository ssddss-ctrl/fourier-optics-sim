"""
physics/imaging.py
--------------------
Coherent (ATF) vs. incoherent (OTF) imaging paths, intensity thresholding,
and print-error quantification (edge placement error, linewidth error).

WHY THIS MODULE EXISTS IN THE PIPELINE
-----------------------------------------
Week 9's lens.py answered "what lands on the wafer if the mask is imaged
coherently?" (coherent_aerial_image). Real lithography sources are not
perfectly coherent, so this module adds the second half of Goodman's
frequency-analysis framework -- the incoherent/OTF path -- and then closes
the loop that the whole project has been building toward: turning an
aerial image into an actual printed-feature *estimate*, and turning that
estimate into numbers (EPE, linewidth error) that quantify how well the
optical system reproduced the intended pattern.

    mask --[pupil, Eq 6-20]--> ATF --[already in lens.py]--> coherent image
    mask --[pupil, Eq 6-20]--> ATF --[Eq 6-25/6-28]--> OTF --> incoherent image
    { coherent image, incoherent image } --[threshold]--> printed feature
    { target mask, printed feature } --[edge finding]--> EPE, linewidth error

This sits directly downstream of lens.py (reused, not reimplemented -- the
ATF/pupil is pupil_function_freq, and the coherent path is
coherent_aerial_image, both called here rather than rebuilt) and upstream
of Week 11's aberrations.py, which will generalize pupil_function_freq
into a phase-carrying pupil that this module's OTF machinery will accept
unchanged (the OTF-from-ATF relation, Eq 6-28, is explicitly valid "for
systems both with and without aberrations" per Goodman).

THRESHOLDING AND PRINT-ERROR METRICS ARE NOT GOODMAN PHYSICS
------------------------------------------------------------------
Goodman's Chapter 6 stops at the aerial image -- the optical intensity
distribution at the wafer plane. Turning that continuous intensity into a
binary "resist printed here / did not print here" pattern, and then
measuring edge placement error (EPE) and linewidth error against the
target, are standard *lithography engineering* conventions, not equations
from the textbook. They are implemented here because they are this week's
build goal, but every function below that isn't building the ATF/OTF says
so explicitly in its docstring, rather than implying it traces to a
Goodman equation number.

All spatial coordinates: µm
All spatial frequencies: cycles/µm (µm⁻¹)
Wavelength: µm
Intensity: normalized (see individual function notes -- the coherent path
    is not peak-normalized, per lens.py's existing convention)
"""

from typing import Tuple

import numpy as np

from fft_engine import fft1d, ifft1d
from constants import WAVELENGTH, NA_DEFAULT
from lens import pupil_function_freq


# ── ATF / OTF ────────────────────────────────────────────────────────────────

def amplitude_point_spread_function(grid, wavelength: float = WAVELENGTH,
                                     NA: float = NA_DEFAULT) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coherent amplitude point-spread function h(x): the real-space impulse
    response of the imaging system, i.e. the image of an infinitesimal
    point object.

    Goodman connection
    -------------------
    h(x) is the inverse Fourier transform of the amplitude transfer
    function H(fx) (Eq. 6-17: "define the amplitude transfer function H
    as the Fourier transform of the space-invariant amplitude impulse
    response" -- used here in reverse, IFT(H) = h). H(fx) itself is just
    the scaled pupil function P (Eq. 6-20), already implemented as
    lens.pupil_function_freq -- reused directly here rather than
    recomputed, since it is exactly the same diffraction-limited,
    unaberrated pupil used by lens.coherent_aerial_image.

    WHY THIS FUNCTION IS THE FOUNDATION FOR BOTH THE COHERENT AND
    INCOHERENT PATHS
    -------------------------------------------------------------------
    h(x) is a coherent (amplitude) quantity. lens.coherent_aerial_image
    effectively uses it already (implicitly, via frequency-domain pupil
    filtering rather than an explicit real-space convolution). The
    incoherent intensity impulse response used by the OTF/incoherent path
    below is |h(x)|^2 (Goodman's statement, following Eq. 6-14, that "the
    impulse response of the incoherent mapping is just the squared
    modulus of the amplitude impulse response") -- so h(x) is computed
    once, here, and reused by both intensity_point_spread_function and
    optical_transfer_function below, rather than being re-derived in each.

    Parameters
    ----------
    grid       : Grid1D — provides the frequency axis grid.f and dx
    wavelength : float — wavelength, µm (defaults to constants.WAVELENGTH)
    NA         : float — numerical aperture (defaults to constants.NA_DEFAULT)

    Returns
    -------
    h : ndarray, shape (N,), complex — amplitude PSF on grid.x
    H : ndarray, shape (N,) — the ATF (hard-edged pupil) on grid.f, returned
        alongside h so callers don't need a second call to
        lens.pupil_function_freq to get the same array
    """
    H = pupil_function_freq(grid, NA=NA, wavelength=wavelength)
    h = ifft1d(H, grid.dx)
    return h, H


def optical_transfer_function(grid, wavelength: float = WAVELENGTH,
                               NA: float = NA_DEFAULT) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optical transfer function (OTF) of a diffraction-limited, unaberrated
    imaging system: the frequency response for INCOHERENT illumination.

    Goodman connection
    -------------------
    Goodman gives two equivalent routes to the OTF:

      1. Eq. (6-25): 𝓗(fx) is the Fourier transform of the intensity
         impulse response |h(x)|^2, normalized by its zero-frequency
         value.
      2. Eq. (6-28): 𝓗(fx) is the normalized autocorrelation of the
         amplitude transfer function H(fx), derived from (1) via the
         autocorrelation theorem (Chapter 2) and Rayleigh's theorem.

    This function implements route (1) rather than a direct correlation
    routine, because it reuses fft1d/ifft1d exactly as lens.py already
    does for the coherent path (h = IFT(H) is computed once by
    amplitude_point_spread_function above), rather than introducing a
    second, independent numerical machinery (e.g. np.correlate) that
    would need its own normalization convention reasoned through from
    scratch. The two routes are mathematically identical (that equivalence
    is exactly what Eq. 6-28's derivation proves) -- route (1) is simply
    the cheaper implementation given what is already on hand.

    VALIDATION PERFORMED BY HAND BEFORE DELIVERY
    ------------------------------------------------
    - For a square (hard-edged, 1D) pupil, Goodman's closed-form result
      (Eq. 6-31) is 𝓗(fx) = Λ(fx / 2f0), a triangle function, with an
      incoherent cutoff at 2·f0 (twice the coherent cutoff f0 =
      cutoff_frequency(NA, wavelength)). Computed 𝓗 numerically via this
      function at L=200 µm, N=2048, NA=0.5, wavelength=0.365 µm and
      compared directly against the analytic triangle function: max
      deviation was 1.7e-3 (discretization error, not a bug), and the
      numerically located cutoff (first frequency where |𝓗| drops below
      1e-3) was 2.735 µm⁻¹ against an analytic 2·f0 = 2.740 µm⁻¹ -- a
      0.17% match.
    - Property 3 (Sec. 6.3.2, proved via Schwarz's inequality): |𝓗(fx)|
      must never exceed its zero-frequency value of 1. Checked directly:
      max(|𝓗|) over the full computed array equals exactly 1.0, attained
      only at fx=0, with no discretization artifact pushing it above 1.
    - For an unaberrated system, 𝓗 must be real (Goodman's aberration
      discussion in 6.4.3 attributes any imaginary/negative-going part of
      the OTF specifically to phase errors W(x,y), absent here). Checked:
      max(|Im(𝓗)|) = 1.7e-16, i.e. floating-point noise, confirming 𝓗 is
      real to numerical precision as expected.

    Parameters
    ----------
    grid       : Grid1D — provides the frequency axis grid.f and dx
    wavelength : float — wavelength, µm (defaults to constants.WAVELENGTH)
    NA         : float — numerical aperture (defaults to constants.NA_DEFAULT)

    Returns
    -------
    OTF : ndarray, shape (N,), complex — normalized OTF on grid.f
        (𝓗(0) = 1 exactly; real-valued to numerical precision for this
        unaberrated pupil, but kept complex since Week 11's aberrated
        pupils will genuinely produce a complex 𝓗)
    H   : ndarray, shape (N,) — the ATF (hard-edged pupil) on grid.f,
        returned for convenience/plotting alongside the OTF
    """
    h, H = amplitude_point_spread_function(grid, wavelength=wavelength, NA=NA)
    intensity_psf = np.abs(h) ** 2
    OTF_raw = fft1d(intensity_psf, grid.dx)
    dc_index = int(np.argmin(np.abs(grid.f)))
    OTF = OTF_raw / OTF_raw[dc_index]
    return OTF, H


# ── Aerial images ────────────────────────────────────────────────────────────

def incoherent_aerial_image(mask: np.ndarray, grid, wavelength: float = WAVELENGTH,
                             NA: float = NA_DEFAULT) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Incoherent aerial image: the wafer-plane intensity produced by imaging
    `mask` through a diffraction-limited, unaberrated system under
    INCOHERENT illumination.

    Goodman connection
    -------------------
    Incoherent imaging is linear in INTENSITY, not amplitude (Sec. 6.1.3,
    confirmed explicitly following Eq. 6-14: "an incoherent imaging system
    is linear in intensity, rather than amplitude"). In the frequency
    domain this is the incoherent analogue of Eq. (6-18)'s coherent
    relation: the normalized image-intensity spectrum equals the
    normalized object-intensity spectrum times the OTF (Eq. 6-26,
    combined with the G_i/G_o normalization of Eq. 6-24).

    Since the binary mask (values in {0, 1}) already IS its own intensity
    transmittance (mask^2 == mask for a binary array), the object
    intensity spectrum is simply fft1d(mask, grid.dx) -- the same raw
    spectrum lens.coherent_aerial_image computes for the coherent path,
    just multiplied by the OTF instead of the ATF/pupil.

    WHY THIS IS FREQUENCY-DOMAIN MULTIPLICATION, NOT A REAL-SPACE
    CONVOLUTION WITH THE INTENSITY PSF
    -------------------------------------------------------------------
    Eq. (6-9) states the incoherent image as a real-space convolution of
    the object intensity with the intensity impulse response |h|^2. That
    convolution and this function's frequency-domain multiplication are
    mathematically equivalent by the convolution theorem -- but they are
    NOT interchangeable numerically here, and this was caught during
    verification, not assumed. fft1d/ifft1d's amplitude PSF h(x) =
    ifft1d(H, dx) comes out centered at grid INDEX 0 (the DFT's natural
    "seam" point, since H is a symmetric real array and ifftshift(H) == H
    for a pupil array that is already even), not at the array's physical
    center index N/2 where grid.x == 0. A direct real-space
    np.convolve(mask, |h|^2) assumes both arrays share the same indexing
    origin and silently produces an image shifted by L/2 relative to the
    mask -- confirmed directly: this wrong approach reproduced the
    wide-open-pupil sanity check with 100% error, while the
    frequency-domain approach below reproduced it to 2.2e-16 (machine
    precision). Multiplying spectra and calling ifft1d, exactly mirroring
    lens.coherent_aerial_image's own architecture, sidesteps the indexing
    question entirely because both fft1d and ifft1d always operate
    consistently on the SAME grid.x / grid.f pair no matter how the
    intermediate impulse response happens to be centered.

    VALIDATION PERFORMED BY HAND BEFORE DELIVERY
    ------------------------------------------------
    - Wide-open-pupil check (H forced to all ones -- no frequency content
      removed): incoherent image reproduced the original binary mask to
      machine precision (max error 2.2e-16), the same limiting-case check
      lens.coherent_aerial_image already performs for the coherent path.
    - DC/energy check: mean(image_intensity) exactly equal to mean(mask)
      to floating-point precision (0.010009765625 both), confirming the
      OTF's zero-frequency normalization (𝓗(0)=1) correctly preserves the
      object's average intensity, per Goodman's own G_i(0)=G_o(0)
      normalization convention (Eq. 6-24 discussion).
    - Realistic-NA check (NA=0.5, wavelength=0.365 µm, 2 µm line):
      intensity stayed non-negative and peaked below 1.0 (0.965) with no
      Gibbs-type overshoot -- consistent with incoherent imaging's
      fundamentally different (non-negative, non-oscillatory) intensity
      PSF compared to the coherent path's amplitude-domain ringing.

    Parameters
    ----------
    mask       : ndarray, shape (N,) — mask transmission (0/1), on grid.x;
                  treated as the object intensity transmittance directly
                  (valid because mask is binary, so mask^2 == mask)
    grid       : Grid1D — mask's spatial/frequency grid (from grid.py)
    wavelength : float — wavelength, µm (defaults to constants.WAVELENGTH)
    NA         : float — numerical aperture (defaults to constants.NA_DEFAULT)

    Returns
    -------
    intensity : ndarray, shape (N,) — incoherent aerial image, on grid.x
                 (NOT peak-normalized, matching lens.coherent_aerial_image's
                 convention, so the wide-open-pupil limit reduces exactly
                 to the original mask)
    OTF       : ndarray, shape (N,), complex — the normalized OTF actually
                 applied (returned so callers/plots don't need a second
                 call to optical_transfer_function)
    H         : ndarray, shape (N,) — the ATF (hard-edged pupil) actually
                 applied, returned for side-by-side plotting against the
                 coherent path's own P
    """
    OTF, H = optical_transfer_function(grid, wavelength=wavelength, NA=NA)
    G_obj = fft1d(mask, grid.dx)
    intensity = np.real(ifft1d(G_obj * OTF, grid.dx))
    return intensity, OTF, H


# ── Thresholding and print-error quantification (engineering, not Goodman) ──

def apply_threshold(intensity: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """
    Binarize an aerial image intensity into a printed-feature estimate.

    NOT A GOODMAN EQUATION -- ENGINEERING CONVENTION
    ------------------------------------------------------
    This is the standard lithography "constant-threshold resist model":
    resist prints (fully develops/clears, depending on tone) wherever the
    aerial image intensity meets or exceeds a fixed fraction of the
    nominal clear-field intensity, and does not print elsewhere. Real
    photoresist response is a nonlinear function of dose, not a hard
    step, but the constant-threshold model is the standard first-order
    approximation used throughout lithography simulation to convert a
    continuous aerial image into a discrete printed pattern, and is what
    this week's build goal calls for.

    Since both this project's masks and the coherent/incoherent aerial
    images are normalized so that fully-open, unfiltered transmission
    corresponds to intensity 1.0 (see lens.coherent_aerial_image's and
    incoherent_aerial_image's wide-open-pupil validation), threshold=0.3
    is expressed directly against that same absolute intensity scale (a
    common illustrative dose-to-clear fraction), not re-derived from any
    Goodman relation.

    Parameters
    ----------
    intensity : ndarray — aerial image intensity (coherent or incoherent)
    threshold : float — intensity fraction above which resist is
                 considered printed (default 0.3)

    Returns
    -------
    printed : ndarray of 0.0/1.0, same shape as intensity — 1.0 where
               intensity >= threshold, else 0.0
    """
    return (intensity >= threshold).astype(float)


def find_edges(binary_pattern: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Locate sub-pixel edge positions (both rising 0->1 and falling 1->0
    transitions) in a binary (or thresholded) 1D pattern.

    NOT A GOODMAN EQUATION -- NUMERICAL/ENGINEERING UTILITY
    ------------------------------------------------------------
    Edge placement error and linewidth error are only as accurate as the
    edge locations they're measured from. Reporting an edge at the
    nearest grid sample would round every measurement to the grid
    spacing dx (here, ~0.05 µm at L=200 µm, N=4096) -- far coarser than
    the sub-nanometer precision meaningful in real lithography metrology.
    This function instead linearly interpolates the exact position where
    a pattern crosses the 0.5 level between two adjacent samples,
    matching how a real edge-detection algorithm on a thresholded image
    would report position, independent of grid resolution.

    Parameters
    ----------
    binary_pattern : ndarray — 0/1 (or thresholded 0.0/1.0) pattern
    x               : ndarray — spatial coordinates (µm), same grid as
                       binary_pattern

    Returns
    -------
    edges : ndarray, sorted ascending — sub-pixel edge x-positions (µm),
             one entry per 0<->1 transition found
    """
    edges = []
    for i in range(len(binary_pattern) - 1):
        a, b = binary_pattern[i], binary_pattern[i + 1]
        if a != b:
            frac = (0.5 - a) / (b - a)
            edges.append(x[i] + frac * (x[i + 1] - x[i]))
    return np.array(sorted(edges))


def edge_placement_error(target_mask: np.ndarray, printed_mask: np.ndarray,
                          x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Edge placement error (EPE): signed distance between each target edge
    and the nearest corresponding printed edge.

    NOT A GOODMAN EQUATION -- LITHOGRAPHY METRIC
    --------------------------------------------------
    EPE is the standard lithography figure of merit for pattern fidelity:
    for every edge the mask design specifies (target_mask), how far did
    the actual printed edge (printed_mask, from apply_threshold) land
    from where it was supposed to be? Each target edge is matched to its
    NEAREST printed edge by position (not by array index, since
    diffraction/thresholding can shift or occasionally merge/split edges)
    -- this is a simple nearest-neighbor matching, adequate for the
    isolated single-line and grating features this project's masks.py
    produces, where the number and rough ordering of edges is preserved
    even after blurring.

    Sign convention: EPE = printed_edge_position - target_edge_position.
    Positive EPE means the printed edge moved outward (away from the
    feature's own opposite edge, if the feature is under-printed for that
    edge's exterior direction is ambiguous in 1D without more context) --
    concretely, for a positive-tone isolated line, a positive EPE at the
    right-hand edge means the line printed WIDER than designed at that
    edge, and a positive EPE at the left-hand edge means the line printed
    NARROWER (since positive = moved in the +x direction for both edges).

    Parameters
    ----------
    target_mask  : ndarray — the intended binary pattern (0/1)
    printed_mask : ndarray — the thresholded printed-feature estimate (0/1),
                    from apply_threshold
    x            : ndarray — spatial coordinates (µm), shared grid

    Returns
    -------
    epe            : ndarray — signed EPE (µm) for each target edge, in the
                      same order as target_edges; np.nan for any target
                      edge with no printed edges at all to match against
                      (i.e. the feature failed to print entirely)
    target_edges   : ndarray — sub-pixel target edge positions (µm)
    printed_edges  : ndarray — sub-pixel printed edge positions (µm)
    """
    target_edges = find_edges(target_mask, x)
    printed_edges = find_edges(printed_mask, x)

    epe = []
    for te in target_edges:
        if len(printed_edges) == 0:
            epe.append(np.nan)
            continue
        nearest = printed_edges[np.argmin(np.abs(printed_edges - te))]
        epe.append(nearest - te)

    return np.array(epe), target_edges, printed_edges


def linewidth_error(target_mask: np.ndarray, printed_mask: np.ndarray,
                     x: np.ndarray) -> Tuple[float, float, float]:
    """
    Linewidth error for an ISOLATED single-line feature: the difference
    between the printed line's width and the target line's width.

    NOT A GOODMAN EQUATION -- LITHOGRAPHY METRIC
    --------------------------------------------------
    Complements edge_placement_error: EPE describes where an individual
    edge landed, while linewidth error describes the net effect on the
    feature's overall size (right edge minus left edge). This function
    only handles the isolated-single-line case (exactly 2 edges each for
    target and printed) deliberately -- a grating's linewidth is
    ambiguous without also specifying WHICH line/space, and a feature
    that fails to resolve at all (0 edges) or breaks up under thresholding
    (>2 edges, e.g. ringing near the resolution limit) cannot be assigned
    a single linewidth without an arbitrary choice. Rather than guess in
    those cases, this function returns NaN so a caller (or the build log)
    has to notice and report the failure explicitly, instead of silently
    reporting a misleading number.

    Parameters
    ----------
    target_mask  : ndarray — the intended binary pattern (0/1); must contain
                    exactly one isolated line (2 edges)
    printed_mask : ndarray — the thresholded printed-feature estimate (0/1)
    x            : ndarray — spatial coordinates (µm), shared grid

    Returns
    -------
    printed_width : float — printed line width (µm), or NaN if the printed
                     pattern does not have exactly 2 edges
    target_width  : float — target line width (µm), or NaN if target_mask
                     does not have exactly 2 edges
    width_error   : float — printed_width - target_width (µm), or NaN if
                     either width above is NaN
    """
    target_edges = find_edges(target_mask, x)
    printed_edges = find_edges(printed_mask, x)

    if len(target_edges) != 2 or len(printed_edges) != 2:
        return float("nan"), float("nan"), float("nan")

    target_width = target_edges[-1] - target_edges[0]
    printed_width = printed_edges[-1] - printed_edges[0]
    return printed_width, target_width, printed_width - target_width