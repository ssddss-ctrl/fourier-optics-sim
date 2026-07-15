"""
scripts/generate_imaging_comparison.py
-----------------------------------------
Week 10 figure generation: ATF vs OTF frequency response, and coherent vs.
incoherent aerial imaging with thresholding + print-error quantification,
for two feature sizes (Week 10 Definition of Done).

Produces two figures in assets/:
  week10_atf_vs_otf.png            -- ATF (pupil) vs OTF (Λ-function) side
                                        by side, cutoff frequencies marked
  week10_imaging_comparison.png    -- for each of two feature widths: mask,
                                        coherent aerial image + threshold,
                                        incoherent aerial image + threshold,
                                        each annotated with EPE/linewidth
                                        error against the target

Run from the repository root:
    python scripts/generate_imaging_comparison.py
"""

import sys

# Flat (non-src) repo layout, matching tests/conftest.py's existing
# convention: physics/*.py and plotting/*.py import each other flatly
# (e.g. lens.py does `from masks import make_grid`, not
# `from physics.masks import ...`), so scripts need physics/ and
# plotting/ individually on sys.path, not just the repo root.
PROJECT_ROOT = "/Users/sohamdamle/Documents/fourier_optics_sim"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, f"{PROJECT_ROOT}/physics")
sys.path.insert(0, f"{PROJECT_ROOT}/plotting")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from grid import Grid1D
from masks import single_line
from constants import WAVELENGTH, NA_DEFAULT
from lens import coherent_aerial_image, cutoff_frequency
from imaging import (
    optical_transfer_function,
    incoherent_aerial_image,
    apply_threshold,
    edge_placement_error,
    linewidth_error,
)
from core import MASK_COLOR, TARGET_COLOR, IMAGE_COLOR, _style_ax

ASSETS_DIR = f"{PROJECT_ROOT}/assets"

ATF_COLOR = "#8E44AD"       # purple, coherent/ATF
OTF_COLOR = "#2980B9"       # blue, incoherent/OTF
THRESHOLD_COLOR = "#7F8C8D"  # grey, threshold line
PRINTED_COLOR = "#E67E22"   # orange, printed feature


def make_atf_vs_otf_figure(grid, wavelength, NA):
    """
    Side-by-side ATF vs OTF frequency response, with cutoff frequencies
    marked. Physically: illustrates that incoherent (OTF) imaging passes
    frequencies up to twice the coherent (ATF) cutoff (Eq. 6-31 result),
    even though the OTF rolls off gradually (triangle function) rather
    than the ATF's hard brick-wall edge (Eq. 6-20).
    """
    OTF, H = optical_transfer_function(grid, wavelength=wavelength, NA=NA)
    f0 = cutoff_frequency(NA, wavelength)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#F8F9FA")

    ax.plot(grid.f, H, color=ATF_COLOR, lw=2.0, label="ATF  H(fx)  (coherent)")
    ax.plot(grid.f, np.real(OTF), color=OTF_COLOR, lw=2.0, label="OTF  H_otf(fx)  (incoherent)")
    ax.axvline(f0, color=ATF_COLOR, ls="--", lw=1.0, alpha=0.7)
    ax.axvline(-f0, color=ATF_COLOR, ls="--", lw=1.0, alpha=0.7)
    ax.axvline(2 * f0, color=OTF_COLOR, ls="--", lw=1.0, alpha=0.7)
    ax.axvline(-2 * f0, color=OTF_COLOR, ls="--", lw=1.0, alpha=0.7)
    ax.text(f0, 1.05, "f₀", color=ATF_COLOR, ha="center", fontsize=9)
    ax.text(2 * f0, 1.05, "2f₀", color=OTF_COLOR, ha="center", fontsize=9)

    ax.set_xlim(-3 * f0, 3 * f0)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("Spatial frequency  fx  (cycles/µm)")
    ax.set_ylabel("Transfer function magnitude")
    ax.set_title(
        f"ATF vs OTF  (NA={NA}, λ={wavelength} µm)  —  "
        "incoherent cutoff is 2× the coherent cutoff",
        fontweight="bold", fontsize=11,
    )
    ax.legend(loc="upper right", frameon=False)
    _style_ax(ax)

    fig.tight_layout()
    return fig


def make_imaging_comparison_figure(grid, wavelength, NA, widths, threshold=0.3):
    """
    For each feature width in `widths`: mask, coherent aerial image +
    threshold, incoherent aerial image + threshold -- three columns per
    row, one row per width, with EPE/linewidth error annotated on each
    imaging panel.

    Physically: shows, side by side, how the SAME optical system responds
    differently to coherent vs incoherent illumination for the same mask,
    and how well the resulting printed (thresholded) feature matches the
    designer's target once you actually measure it (EPE, linewidth error)
    rather than just eyeballing the aerial image.
    """
    n_rows = len(widths)
    fig = plt.figure(figsize=(13, 4.2 * n_rows))
    fig.patch.set_facecolor("#F8F9FA")
    gs = gridspec.GridSpec(n_rows, 3, hspace=0.55, wspace=0.3)

    xlim = (-4.0, 4.0)

    for row, width in enumerate(widths):
        mask = single_line(grid.x, width=width, center=0.0)

        _, intensity_c, _ = coherent_aerial_image(mask, grid, wavelength=wavelength, NA=NA)
        intensity_i, _, _ = incoherent_aerial_image(mask, grid, wavelength=wavelength, NA=NA)

        printed_c = apply_threshold(intensity_c, threshold=threshold)
        printed_i = apply_threshold(intensity_i, threshold=threshold)

        epe_c, target_edges, printed_edges_c = edge_placement_error(mask, printed_c, grid.x)
        epe_i, _, printed_edges_i = edge_placement_error(mask, printed_i, grid.x)

        _, _, lwe_c = linewidth_error(mask, printed_c, grid.x)
        _, _, lwe_i = linewidth_error(mask, printed_i, grid.x)

        # ── Column 1: mask/target ────────────────────────────────────────
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.fill_between(grid.x, mask, alpha=0.7, color=TARGET_COLOR, step="mid")
        ax0.plot(grid.x, mask, color=TARGET_COLOR, lw=1.2)
        ax0.set_title(f"Target  (width={width} µm)", fontweight="bold", fontsize=10)
        ax0.set_ylabel("Transmission")
        ax0.set_xlim(xlim)
        ax0.set_ylim(-0.1, 1.3)
        _style_ax(ax0)

        # ── Column 2: coherent aerial image ─────────────────────────────
        ax1 = fig.add_subplot(gs[row, 1])
        ax1.plot(grid.x, intensity_c, color=ATF_COLOR, lw=1.6)
        ax1.axhline(threshold, color=THRESHOLD_COLOR, ls="--", lw=1.0)
        ax1.plot(printed_edges_c, [threshold] * len(printed_edges_c),
                  "o", color=PRINTED_COLOR, ms=6, zorder=5)
        epe_str = ", ".join(f"{e:.3f}" for e in epe_c)
        ax1.set_title(
            f"Coherent (ATF)  —  EPE=[{epe_str}] µm\nlinewidth err={lwe_c:.3f} µm",
            fontweight="bold", fontsize=9.5,
        )
        ax1.set_xlim(xlim)
        ax1.set_ylim(-0.05, 1.4)
        _style_ax(ax1)

        # ── Column 3: incoherent aerial image ───────────────────────────
        ax2 = fig.add_subplot(gs[row, 2])
        ax2.plot(grid.x, intensity_i, color=OTF_COLOR, lw=1.6)
        ax2.axhline(threshold, color=THRESHOLD_COLOR, ls="--", lw=1.0)
        ax2.plot(printed_edges_i, [threshold] * len(printed_edges_i),
                  "o", color=PRINTED_COLOR, ms=6, zorder=5)
        epe_str_i = ", ".join(f"{e:.3f}" for e in epe_i)
        ax2.set_title(
            f"Incoherent (OTF)  —  EPE=[{epe_str_i}] µm\nlinewidth err={lwe_i:.3f} µm",
            fontweight="bold", fontsize=9.5,
        )
        ax2.set_xlim(xlim)
        ax2.set_ylim(-0.05, 1.4)
        _style_ax(ax2)

        for ax in (ax0, ax1, ax2):
            ax.set_xlabel("x  (µm)")

    fig.suptitle(
        "Week 10: Coherent vs. Incoherent Imaging, Thresholding, and Print Error",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    return fig


def print_summary_table(grid, wavelength, NA, widths, threshold=0.3):
    """Console summary (also used to fill in the build log's Validation section)."""
    print(f"{'width (um)':>10} | {'path':>10} | {'EPE (um)':>22} | {'linewidth err (um)':>18}")
    print("-" * 70)
    for width in widths:
        mask = single_line(grid.x, width=width, center=0.0)
        _, intensity_c, _ = coherent_aerial_image(mask, grid, wavelength=wavelength, NA=NA)
        intensity_i, _, _ = incoherent_aerial_image(mask, grid, wavelength=wavelength, NA=NA)
        printed_c = apply_threshold(intensity_c, threshold=threshold)
        printed_i = apply_threshold(intensity_i, threshold=threshold)
        epe_c, *_ = edge_placement_error(mask, printed_c, grid.x)
        epe_i, *_ = edge_placement_error(mask, printed_i, grid.x)
        _, _, lwe_c = linewidth_error(mask, printed_c, grid.x)
        _, _, lwe_i = linewidth_error(mask, printed_i, grid.x)
        print(f"{width:>10.2f} | {'coherent':>10} | {str(np.round(epe_c, 4)):>22} | {lwe_c:>18.4f}")
        print(f"{width:>10.2f} | {'incoherent':>10} | {str(np.round(epe_i, 4)):>22} | {lwe_i:>18.4f}")


if __name__ == "__main__":
    grid = Grid1D(L=200.0, N=4096)
    wavelength = WAVELENGTH
    NA = NA_DEFAULT

    # Two feature sizes: one comfortably above the coherent resolution
    # limit, one near/below it, so the print-error numbers actually show
    # something interesting (Definition of Done: "at least two feature
    # sizes").
    widths = [2.0, 0.8]

    fig1 = make_atf_vs_otf_figure(grid, wavelength, NA)
    fig1.savefig(f"{ASSETS_DIR}/week10_atf_vs_otf.png", dpi=150, bbox_inches="tight")

    fig2 = make_imaging_comparison_figure(grid, wavelength, NA, widths)
    fig2.savefig(f"{ASSETS_DIR}/week10_imaging_comparison.png", dpi=150, bbox_inches="tight")

    print_summary_table(grid, wavelength, NA, widths)
    print("\nSaved: assets/week10_atf_vs_otf.png, assets/week10_imaging_comparison.png")