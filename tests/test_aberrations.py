"""
tests/test_aberrations.py
----------------------------
Unit tests for physics/aberrations.py (defocus wavefront error, generalized
pupil function) and the defocus_waves extension to imaging.py's ATF/OTF
construction.

Test organization mirrors tests/test_imaging.py: the Week 10 regression
anchor first (defocus_waves=0.0 must not change anything), then the
physical invariants a phase-only aberration must satisfy regardless of
implementation detail (magnitude preservation, unchanged cutoff, Schwarz's
inequality), then the two behaviors that specifically distinguish an
aberrated system from an unaberrated one (contrast reversal, geometric-optics
sinc asymptote at large defocus).
"""

import numpy as np
import pytest

from grid import Grid1D
from constants import WAVELENGTH, NA_DEFAULT
from lens import cutoff_frequency, pupil_function_freq, coherent_aerial_image
from masks import single_line
from imaging import (
    amplitude_point_spread_function,
    optical_transfer_function,
    incoherent_aerial_image,
)
from aberrations import focus_error_wavefront, generalized_pupil_function


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def grid():
    # Same L/N as test_imaging.py's fixture, so numbers are directly
    # comparable against that module's already-hand-verified Week 10 values.
    return Grid1D(L=200.0, N=4096)


# ── Regression anchor: defocus_waves=0.0 must reproduce Week 10 exactly ──────

def test_zero_defocus_matches_unaberrated_atf_exactly(grid):
    """defocus_waves=0.0 must skip the aberration branch entirely -- H stays
    the bare real pupil, not just numerically close to it."""
    h, H = amplitude_point_spread_function(grid, wavelength=WAVELENGTH, NA=NA_DEFAULT,
                                            defocus_waves=0.0)
    H_direct = pupil_function_freq(grid, NA=NA_DEFAULT, wavelength=WAVELENGTH)
    assert H.dtype == np.float64
    assert np.array_equal(H, H_direct)


def test_zero_defocus_matches_unaberrated_otf_exactly(grid):
    OTF_ref, H_ref = optical_transfer_function(grid, wavelength=WAVELENGTH, NA=NA_DEFAULT)
    OTF_dw, H_dw = optical_transfer_function(grid, wavelength=WAVELENGTH, NA=NA_DEFAULT,
                                              defocus_waves=0.0)
    assert np.array_equal(OTF_ref, OTF_dw)
    assert np.array_equal(H_ref, H_dw)


# ── Phase-only perturbation invariants ───────────────────────────────────────

@pytest.mark.parametrize("defocus_waves", [0.25, 0.5, 1.0, 2.0, 3.0, 5.0])
def test_aberrated_atf_magnitude_matches_unaberrated(grid, defocus_waves):
    """A wavefront error is a pure phase term (Eq. 14): |P_G| == |P|
    everywhere, for any defocus amount."""
    _, H_aberrated = amplitude_point_spread_function(grid, wavelength=WAVELENGTH, NA=NA_DEFAULT,
                                                       defocus_waves=defocus_waves)
    H_unaberrated = pupil_function_freq(grid, NA=NA_DEFAULT, wavelength=WAVELENGTH)
    assert np.allclose(np.abs(H_aberrated), H_unaberrated)


@pytest.mark.parametrize("defocus_waves", [0.25, 0.5, 1.0, 2.0, 3.0, 5.0])
def test_cutoff_frequency_unchanged_by_defocus(grid, defocus_waves):
    """Defocus reshapes the phase across the pupil, not its physical extent
    -- the frequency support (which fx pass at all) must be identical to
    the unaberrated case."""
    _, H_aberrated = amplitude_point_spread_function(grid, wavelength=WAVELENGTH, NA=NA_DEFAULT,
                                                       defocus_waves=defocus_waves)
    H_unaberrated = pupil_function_freq(grid, NA=NA_DEFAULT, wavelength=WAVELENGTH)
    support_aberrated = np.abs(H_aberrated) > 0
    support_unaberrated = H_unaberrated > 0
    assert np.array_equal(support_aberrated, support_unaberrated)


# ── Schwarz's inequality (Goodman 6.4.3): aberration can only reduce contrast ─

@pytest.mark.parametrize("defocus_waves", [0.25, 0.5, 1.0, 2.0, 3.0, 5.0])
def test_mtf_never_exceeds_unaberrated_mtf(grid, defocus_waves):
    """|OTF| for any aberrated pupil is bounded above by the unaberrated
    |OTF| at every frequency -- Goodman's Schwarz's-inequality argument
    (Sec. 6.3.2/6.4.3) that applies to any P_G, not just the hard-edged P."""
    OTF_unaberrated, _ = optical_transfer_function(grid, wavelength=WAVELENGTH, NA=NA_DEFAULT)
    OTF_aberrated, _ = optical_transfer_function(grid, wavelength=WAVELENGTH, NA=NA_DEFAULT,
                                                  defocus_waves=defocus_waves)
    mtf_unaberrated = np.abs(OTF_unaberrated)
    mtf_aberrated = np.abs(OTF_aberrated)
    assert np.all(mtf_aberrated <= mtf_unaberrated + 1e-9)


# ── Contrast reversal ─────────────────────────────────────────────────────────

def test_large_defocus_produces_contrast_reversal(grid):
    """At sufficiently large defocus, Goodman 6.4.3 predicts the OTF goes
    negative at some frequencies (spurious resolution / contrast reversal)
    -- a genuinely aberration-specific effect, not just contrast loss."""
    OTF, _ = optical_transfer_function(grid, wavelength=WAVELENGTH, NA=NA_DEFAULT,
                                        defocus_waves=1.5)
    assert np.any(OTF.real < -1e-6)


# ── incoherent_aerial_image's defocus_waves passthrough ──────────────────────

def test_incoherent_aerial_image_zero_defocus_matches_unaberrated(grid):
    """Same regression anchor as the ATF/OTF functions, one level up the
    call chain: defocus_waves=0.0 must be indistinguishable from omitting
    it entirely."""
    mask = single_line(grid.x, width=2.0, center=0.0)
    intensity_ref, OTF_ref, H_ref = incoherent_aerial_image(mask, grid)
    intensity_dw, OTF_dw, H_dw = incoherent_aerial_image(mask, grid, defocus_waves=0.0)
    assert np.array_equal(intensity_ref, intensity_dw)
    assert np.array_equal(OTF_ref, OTF_dw)
    assert np.array_equal(H_ref, H_dw)


def test_incoherent_aerial_image_defocus_preserves_mean_intensity(grid):
    """OTF(0)=1 exactly regardless of aberration (Property 1, Sec. 6.3.2),
    so the DC/energy invariant already verified for the unaberrated path in
    test_imaging.py must survive defocus too: mean intensity still equals
    mean(mask)."""
    mask = single_line(grid.x, width=2.0, center=0.0)
    intensity, _, _ = incoherent_aerial_image(mask, grid, defocus_waves=3.0)
    assert np.mean(intensity) == pytest.approx(np.mean(mask), abs=1e-9)


# ── lens.coherent_aerial_image's defocus_waves passthrough ───────────────────

def test_coherent_aerial_image_zero_defocus_matches_unaberrated(grid):
    """Same regression anchor, coherent path: defocus_waves=0.0 must be
    indistinguishable from omitting it entirely."""
    mask = single_line(grid.x, width=2.0, center=0.0)
    field_ref, intensity_ref, P_ref = coherent_aerial_image(mask, grid)
    field_dw, intensity_dw, P_dw = coherent_aerial_image(mask, grid, defocus_waves=0.0)
    assert np.array_equal(intensity_ref, intensity_dw)
    assert np.array_equal(P_ref, P_dw)
    assert P_ref.dtype == np.float64


def test_coherent_aerial_image_defocus_matches_amplitude_psf_pupil(grid):
    """The pupil coherent_aerial_image applies under defocus must be
    exactly the same generalized pupil imaging.amplitude_point_spread_function
    builds -- reused, not re-derived independently in lens.py."""
    _, _, P = coherent_aerial_image(mask=single_line(grid.x, width=2.0, center=0.0),
                                     grid=grid, wavelength=WAVELENGTH, NA=NA_DEFAULT,
                                     defocus_waves=1.5)
    _, H = amplitude_point_spread_function(grid, wavelength=WAVELENGTH, NA=NA_DEFAULT,
                                            defocus_waves=1.5)
    assert np.array_equal(P, H)


# ── Large-defocus geometric-optics sinc asymptote (build notes Eq. 17) ──────

@pytest.mark.parametrize("defocus_waves", [5.0, 7.0, 10.0])
def test_large_defocus_matches_sinc_asymptote(grid, defocus_waves):
    """
    Goodman's original (2D, square-pupil) large-defocus asymptote:

        H(fx,fy) ~ sinc(8*Wm/wavelength * fx/(2*f0)) * sinc(...fy term...)

    with sinc(x) = sin(pi*x)/(pi*x) (np.sinc's own convention). Dropped to
    this codebase's 1D convention, with Wm = defocus_waves*wavelength (so
    Wm/wavelength = defocus_waves) and f0 = cutoff_frequency(NA, wavelength):

        H_asymptote(fx) ~= sinc(4 * defocus_waves * fx / f_cutoff)

    This is a large-Wm geometric-optics limit, not an identity -- checked
    only at defocus_waves >= 5 (where geometric blur dominates diffraction)
    and only over the actual passband |fx| <= f_cutoff (outside it, the
    numerical MTF is exactly zero by construction and the asymptote isn't
    meant to apply). Compared against |OTF| (real, even function) rather
    than the complex OTF, per the geometric-optics derivation.
    """
    f_cutoff = cutoff_frequency(NA_DEFAULT, WAVELENGTH)
    OTF, _ = optical_transfer_function(grid, wavelength=WAVELENGTH, NA=NA_DEFAULT,
                                        defocus_waves=defocus_waves)
    MTF = np.abs(OTF)
    H_asymptote = np.abs(np.sinc(4 * defocus_waves * grid.f / f_cutoff))

    dc_index = int(np.argmin(np.abs(grid.f)))
    assert MTF[dc_index] == pytest.approx(1.0)
    assert H_asymptote[dc_index] == pytest.approx(1.0)

    passband = np.abs(grid.f) <= f_cutoff
    diff = np.abs(MTF[passband] - H_asymptote[passband])
    assert np.mean(diff) < 0.02
    assert np.max(diff) < 0.1
