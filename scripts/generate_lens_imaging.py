"""
scripts/generate_lens_imaging.py
-----------------------------------
Generates the Week 9 deliverable figures:

  1. week9_pupil_plane_and_aerial_image.png
     A single-line mask carried through the full lens.py chain: mask ->
     lens focal-plane field (the physical pupil plane, Eq. 5-22) -> pupil
     cutoff in frequency space (Eq. 6-20) -> aerial image. Four panels,
     with the pupil plane and its frequency-domain equivalent shown
     side by side so the NA cutoff --> f_cutoff = NA/wavelength
     correspondence (lens.cutoff_frequency) is visually obvious, not just
     asserted.

  2. week9_na_sweep_grating.png
     Demonstrates that NA is a genuinely TUNABLE parameter (per this
     week's definition of done), by imaging the SAME grating at three
     different NA values and showing the aerial image lose contrast as
     NA drops below the value needed to pass the grating's fundamental
     frequency -- the classic diffraction-limited resolution cutoff.

WHY A DEDICATED SCRIPT INSTEAD OF plotting.core.four_panel_plot
--------------------------------------------------------------------
Same reasoning as Week 8's generate_diffraction_patterns.py: four_panel_plot
assumes all four panels share one spatial (x) axis. The pupil plane here
lives on a physically different axis (u = grid.f * wavelength * focal_length,
typically tens of thousands of µm for a real focal length) than the mask's
own x axis, and the frequency panel lives on a third axis (cycles/µm). Three
incompatible physical axes can't be forced into four_panel_plot's shared-x
layout without mislabeling something. This script builds dedicated figures
instead, but still reuses plotting.core's actual color constants and
_style_ax helper rather than redefining styling -- same convention Week 8
established.

Run from repo root:
    python scripts/generate_lens_imaging.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "physics"))
sys.path.insert(0, str(REPO_ROOT / "plotting"))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from grid import Grid1D
from masks import single_line, line_space_grating
from lens import (
    lens_focal_plane_field,
    pupil_radius,
    cutoff_frequency,
    pupil_function_freq,
    coherent_aerial_image,
)
from constants import WAVELENGTH, NA_DEFAULT
from core import MASK_COLOR, SPECTRUM_COLOR, IMAGE_COLOR, TARGET_COLOR, _style_ax

ASSETS_DIR = REPO_ROOT / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

# A real (if arbitrary) projection-lens focal length for the pupil-plane
# plot to be labeled in physically meaningful units. Chosen only to keep
# the pupil-plane coordinate u in an easy-to-read range for this figure;
# cutoff_frequency (and therefore the actual pupil-cutoff physics) does
# NOT depend on this choice, per lens.py's documented focal-length
# cancellation.
FOCAL_LENGTH_UM = 10_000.0  # 10 mm


def figure_pupil_and_image(grid: Grid1D) -> Figure:
    """Four-panel figure: mask, pupil plane, frequency cutoff, aerial image."""
    mask = single_line(grid.x, width=1.0)

    u, field_focal = lens_focal_plane_field(
        mask, grid, wavelength=WAVELENGTH, focal_length=FOCAL_LENGTH_UM
    )
    r_pupil = pupil_radius(FOCAL_LENGTH_UM, NA_DEFAULT)
    f_cutoff = cutoff_frequency(NA_DEFAULT, WAVELENGTH)
    P = pupil_function_freq(grid, NA=NA_DEFAULT, wavelength=WAVELENGTH)
    field_image, intensity, _ = coherent_aerial_image(
        mask, grid, wavelength=WAVELENGTH, NA=NA_DEFAULT
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor("#F8F9FA")
    ax_mask, ax_pupil, ax_freq, ax_image = axes.flatten()

    # Panel 1: mask
    ax_mask.fill_between(grid.x, mask, alpha=0.8, color=MASK_COLOR, step="mid")
    ax_mask.plot(grid.x, mask, color=MASK_COLOR, lw=1.2)
    ax_mask.set_title("① Mask", fontweight="bold", fontsize=11)
    ax_mask.set_xlabel("x  (µm)")
    ax_mask.set_ylabel("Transmission")
    ax_mask.set_ylim(-0.1, 1.3)
    _style_ax(ax_mask)

    # Panel 2: physical pupil plane (Eq. 5-22 field magnitude vs u), with
    # the actual lens aperture edge (r_pupil) marked explicitly -- this
    # IS the "clearly visualize the pupil plane" deliverable.
    ax_pupil.plot(u, np.abs(field_focal), color=SPECTRUM_COLOR, lw=1.3)
    ax_pupil.axvline(r_pupil, color=TARGET_COLOR, ls="--", lw=1.5,
                      label=f"pupil edge (±{r_pupil:.0f} µm)")
    ax_pupil.axvline(-r_pupil, color=TARGET_COLOR, ls="--", lw=1.5)
    ax_pupil.axvspan(-r_pupil, r_pupil, color=TARGET_COLOR, alpha=0.08)
    ax_pupil.set_title("② Pupil Plane  |U_f(u)|  (Eq. 5-22)", fontweight="bold", fontsize=11)
    ax_pupil.set_xlabel(f"u  (µm)   [focal length = {FOCAL_LENGTH_UM/1000:.0f} mm]")
    ax_pupil.set_ylabel("Field magnitude")
    ax_pupil.legend(fontsize=8, loc="upper right")
    _style_ax(ax_pupil)

    # Panel 3: the SAME cutoff, expressed in frequency space (Eq. 6-20),
    # directly overlaid on the mask's own spectrum -- shows explicitly
    # that panel 2's physical aperture edge and this panel's f_cutoff are
    # the same cutoff, just viewed in two different (linearly related)
    # coordinates.
    G_mag = np.abs(np.fft.fftshift(np.fft.fft(mask)) / len(mask))
    ax_freq.plot(grid.f, G_mag, color=MASK_COLOR, lw=1.0, alpha=0.5, label="mask spectrum |G(fx)|")
    ax_freq.fill_between(grid.f, G_mag, where=(P > 0), color=IMAGE_COLOR, alpha=0.4, step="mid",
                          label="passed by pupil")
    ax_freq.axvline(f_cutoff, color=TARGET_COLOR, ls="--", lw=1.5,
                     label=f"f_cutoff = NA/λ = {f_cutoff:.2f} µm⁻¹")
    ax_freq.axvline(-f_cutoff, color=TARGET_COLOR, ls="--", lw=1.5)
    ax_freq.set_xlim(-grid.f_max, grid.f_max)
    ax_freq.set_title("③ Frequency-Domain Pupil Cutoff  (Eq. 6-20)", fontweight="bold", fontsize=11)
    ax_freq.set_xlabel("Spatial frequency  (cycles/µm)")
    ax_freq.set_ylabel("|G(fx)|")
    ax_freq.legend(fontsize=8, loc="upper right")
    _style_ax(ax_freq)

    # Panel 4: aerial image vs. original mask
    ax_image.plot(grid.x, mask, color=MASK_COLOR, lw=1.0, ls=":", alpha=0.7, label="mask (ideal)")
    ax_image.plot(grid.x, intensity, color=IMAGE_COLOR, lw=1.8, label="aerial image (coherent)")
    ax_image.fill_between(grid.x, intensity, alpha=0.3, color=IMAGE_COLOR)
    ax_image.set_title(f"④ Aerial Image  (NA={NA_DEFAULT})", fontweight="bold", fontsize=11)
    ax_image.set_xlabel("x  (µm)")
    ax_image.set_ylabel("Intensity (a.u.)")
    ax_image.legend(fontsize=8, loc="upper right")
    _style_ax(ax_image)

    fig.suptitle(
        "Coherent Imaging Chain: Mask → Lens Fourier Transform → Pupil Cutoff → Aerial Image",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    return fig


def figure_na_sweep(grid: Grid1D) -> Figure:
    """
    Images the SAME grating at three NA values to demonstrate NA is a
    genuinely tunable resolution parameter, not just a fixed constant
    plugged into the pipeline once.

    Grating pitch is chosen as an exact integer number of grid samples
    (pitch = k * grid.dx) so its spectrum is a clean, leak-free comb --
    the Week 8 sampling-grid lesson re-applied here (see lens.py's
    coherent_aerial_image docstring for the full explanation of why a
    non-grid-aligned pitch produces spurious leakage that can masquerade
    as a bug in the pupil cutoff itself).
    """
    samples_per_period = 16
    pitch = samples_per_period * grid.dx
    grating = line_space_grating(grid.x, pitch=pitch, duty_cycle=0.5)
    fundamental_freq = 1.0 / pitch

    NA_values = [0.2, 0.5, 0.9]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    fig.patch.set_facecolor("#F8F9FA")

    for ax, NA in zip(axes, NA_values):
        f_cutoff = cutoff_frequency(NA, WAVELENGTH)
        _, intensity, _ = coherent_aerial_image(grating, grid, wavelength=WAVELENGTH, NA=NA)
        resolved = fundamental_freq <= f_cutoff

        ax.plot(grid.x, grating, color=MASK_COLOR, lw=0.9, ls=":", alpha=0.6, label="mask")
        ax.plot(grid.x, intensity, color=IMAGE_COLOR, lw=1.6, label="aerial image")
        ax.fill_between(grid.x, intensity, alpha=0.3, color=IMAGE_COLOR)
        ax.set_xlim(-5, 5)  # zoom to a few periods for readability
        status = "resolved" if resolved else "NOT resolved"
        ax.set_title(
            f"NA = {NA}   (f_cutoff={f_cutoff:.2f} µm⁻¹, grating fundamental={fundamental_freq:.2f} µm⁻¹)"
            f"\n→ {status}",
            fontsize=9.5, fontweight="bold",
        )
        ax.set_xlabel("x  (µm)")
        if ax is axes[0]:
            ax.set_ylabel("Intensity / Transmission (a.u.)")
            ax.legend(fontsize=8, loc="upper right")
        _style_ax(ax)

    fig.suptitle(
        f"NA as a Tunable Resolution Parameter — Same Grating (pitch={pitch:.3f} µm), Three NA Values",
        fontsize=12.5, fontweight="bold", y=1.04,
    )
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    grid = Grid1D(L=20.0, N=256)

    fig1 = figure_pupil_and_image(grid)
    out1 = ASSETS_DIR / "week9_pupil_plane_and_aerial_image.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved {out1}")

    fig2 = figure_na_sweep(grid)
    out2 = ASSETS_DIR / "week9_na_sweep_grating.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved {out2}")