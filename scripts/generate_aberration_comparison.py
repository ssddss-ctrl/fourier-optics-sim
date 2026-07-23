"""
scripts/generate_aberration_comparison.py
---------------------------------------------
Week 11 figure generation: defocus wavefront error's effect on the ATF/OTF
frequency response, and on the resulting printed feature, across a sweep of
defocus_waves values.

Produces two figures in assets/:
  week11_atf_otf_defocus_sweep.png  -- ATF phase (build notes Eq. 14) and OTF
                                        magnitude (Eq. 6-25/6-28) across a
                                        defocus_waves sweep, cutoff frequency
                                        marked (unchanged by defocus -- only
                                        the phase across the pupil changes)
  week11_aberration_imaging_comparison.png -- for each defocus_waves value:
                                        aberrated vs. diffraction-limited
                                        aerial image + threshold + printed
                                        feature, annotated with EPE/linewidth
                                        error against the target

Run from the repository root:
    python scripts/generate_aberration_comparison.py
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
from lens import cutoff_frequency
from imaging import (
    optical_transfer_function,
    incoherent_aerial_image,
    apply_threshold,
    edge_placement_error,
    linewidth_error,
)
from core import TARGET_COLOR, _style_ax

ASSETS_DIR = f"{PROJECT_ROOT}/assets"

DIFFRACTION_LIMITED_COLOR = "#7F8C8D"  # grey, defocus_waves=0.0 reference
THRESHOLD_COLOR = "#95A5A6"            # light grey, threshold line
PRINTED_COLOR = "#E67E22"              # orange, printed feature edge markers
DEFOCUS_CMAP = plt.colormaps["plasma"]


def _defocus_colors(defocus_values):
    """Map each defocus_waves value to a color along a single sequential
    colormap, so panels showing multiple curves read as one degrading
    progression rather than an arbitrary categorical palette."""
    span = max(defocus_values) if max(defocus_values) > 0 else 1.0
    return [DEFOCUS_CMAP(0.15 + 0.75 * (d / span)) for d in defocus_values]


def make_atf_otf_defocus_sweep_figure(grid, wavelength, NA, defocus_values):
    """
    Two panels sharing the frequency axis:
      left  -- ATF phase angle(H(fx)) (build notes Eq. 14: H = P*exp(j*k*W))
               for each defocus_waves value, masked to NaN outside the
               pupil support (phase of an exactly-zero pupil is undefined,
               not physically zero, so it's excluded rather than plotted
               as a misleading flat line).
      right -- OTF magnitude |OTF(fx)| (Eq. 6-25/6-28) for the same sweep,
               showing MTF degradation with increasing defocus. The cutoff
               frequency (dashed vertical line) is the same for every
               curve -- defocus reshapes phase across the pupil, not its
               physical extent (verified directly in
               tests/test_aberrations.py::test_cutoff_frequency_unchanged_by_defocus).

    Physically: this is the frequency-domain view of everything the second
    figure shows in real space -- the ATF panel is *why* the aerial image
    degrades (phase error across the pupil), the OTF panel is *how much*
    contrast survives at each spatial frequency once that phase error is
    folded through the OTF-from-ATF relation.
    """
    f0 = cutoff_frequency(NA, wavelength)
    colors = _defocus_colors(defocus_values)

    fig, (ax_phase, ax_mtf) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#F8F9FA")

    for defocus_waves, color in zip(defocus_values, colors):
        OTF, H = optical_transfer_function(grid, wavelength=wavelength, NA=NA,
                                            defocus_waves=defocus_waves)
        support = np.abs(H) > 0
        phase = np.full_like(grid.f, np.nan)
        phase[support] = np.angle(H[support])
        ax_phase.plot(grid.f, phase, color=color, lw=1.8,
                      label=f"defocus = {defocus_waves:.1f} waves")
        ax_mtf.plot(grid.f, np.abs(OTF), color=color, lw=1.8,
                    label=f"defocus = {defocus_waves:.1f} waves")

    for ax in (ax_phase, ax_mtf):
        ax.axvline(f0, color=DIFFRACTION_LIMITED_COLOR, ls="--", lw=1.0, alpha=0.7)
        ax.axvline(-f0, color=DIFFRACTION_LIMITED_COLOR, ls="--", lw=1.0, alpha=0.7)
        ax.set_xlim(-1.5 * f0, 1.5 * f0)
        ax.set_xlabel("Spatial frequency  fx  (cycles/µm)")
        _style_ax(ax)

    ax_phase.set_ylabel("ATF phase  angle(H)  (radians)")
    ax_phase.set_title("ATF phase across the pupil", fontweight="bold", fontsize=11)
    ax_phase.legend(loc="upper right", frameon=False, fontsize=8)

    ax_mtf.set_ylabel("OTF magnitude  |H_otf(fx)|")
    ax_mtf.set_ylim(-0.05, 1.15)
    ax_mtf.set_title("MTF degradation with defocus", fontweight="bold", fontsize=11)
    ax_mtf.legend(loc="upper right", frameon=False, fontsize=8)

    fig.suptitle(
        f"Week 11: Defocus Wavefront Error  (NA={NA}, λ={wavelength} µm)  --  "
        "cutoff frequency unchanged, phase/contrast degrade",
        fontsize=12, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    return fig


def make_aberration_imaging_comparison_figure(grid, wavelength, NA, width,
                                                defocus_values, threshold=0.3):
    """
    One row per defocus_waves value: aerial image (aberrated vs.
    diffraction-limited overlay) and printed feature (target vs. aberrated
    printed), annotated with EPE/linewidth error against the target --
    reusing incoherent_aerial_image/apply_threshold/edge_placement_error/
    linewidth_error from imaging.py directly rather than reimplementing
    any of that logic here.

    Physically: shows how the SAME optical system, imaging the SAME mask,
    prints a progressively worse copy of the target as defocus increases --
    the real-space consequence of the MTF degradation in the first figure.
    """
    colors = _defocus_colors(defocus_values)
    n_rows = len(defocus_values)
    fig = plt.figure(figsize=(11, 3.6 * n_rows))
    fig.patch.set_facecolor("#F8F9FA")
    # Explicit margins rather than fig.tight_layout(): GridSpec figures with
    # a fig.suptitle aren't tight_layout-compatible (confirmed by the
    # UserWarning it raises here) and left to its own devices tight_layout
    # leaves a large blank band above row 0 that grows with n_rows -- fixed
    # top/bottom fractions scale correctly with the sweep length instead.
    gs = gridspec.GridSpec(n_rows, 2, hspace=0.6, wspace=0.3,
                            top=1.0 - 0.5 / n_rows, bottom=0.06, left=0.07, right=0.98)

    xlim = (-4.0, 4.0)
    mask = single_line(grid.x, width=width, center=0.0)

    intensity_dl, _, _ = incoherent_aerial_image(mask, grid, wavelength=wavelength, NA=NA,
                                                  defocus_waves=0.0)

    for row, (defocus_waves, color) in enumerate(zip(defocus_values, colors)):
        intensity, _, _ = incoherent_aerial_image(mask, grid, wavelength=wavelength, NA=NA,
                                                    defocus_waves=defocus_waves)
        printed = apply_threshold(intensity, threshold=threshold)

        epe, target_edges, printed_edges = edge_placement_error(mask, printed, grid.x)
        _, _, lwe = linewidth_error(mask, printed, grid.x)

        # ── Column 1: aerial image, aberrated vs. diffraction-limited ───
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.plot(grid.x, intensity_dl, color=DIFFRACTION_LIMITED_COLOR, lw=1.4,
                 ls="--", label="diffraction-limited")
        ax0.plot(grid.x, intensity, color=color, lw=1.8, label="aberrated")
        ax0.axhline(threshold, color=THRESHOLD_COLOR, ls=":", lw=1.0)
        ax0.set_title(f"defocus = {defocus_waves:.1f} waves  --  aerial image",
                      fontweight="bold", fontsize=9.5)
        ax0.set_xlim(xlim)
        ax0.set_ylim(-0.05, 1.2)
        ax0.set_ylabel("Intensity (a.u.)")
        ax0.legend(loc="upper right", frameon=False, fontsize=7.5)
        _style_ax(ax0)

        # ── Column 2: printed feature vs. target ─────────────────────────
        ax1 = fig.add_subplot(gs[row, 1])
        ax1.fill_between(grid.x, mask, alpha=0.35, color=TARGET_COLOR, step="mid",
                          label="target")
        ax1.plot(grid.x, printed, color=color, lw=1.8, drawstyle="steps-mid",
                 label="printed")
        if len(printed_edges) > 0:
            ax1.plot(printed_edges, [0.5] * len(printed_edges),
                     "o", color=PRINTED_COLOR, ms=5, zorder=5)
        epe_str = ", ".join(f"{e:.3f}" for e in epe) if len(epe) else "n/a"
        lwe_str = f"{lwe:.3f}" if not np.isnan(lwe) else "n/a"
        ax1.set_title(
            f"printed feature  --  EPE=[{epe_str}] µm, linewidth err={lwe_str} µm",
            fontweight="bold", fontsize=9.5,
        )
        ax1.set_xlim(xlim)
        ax1.set_ylim(-0.1, 1.3)
        ax1.legend(loc="upper right", frameon=False, fontsize=7.5)
        _style_ax(ax1)

        for ax in (ax0, ax1):
            ax.set_xlabel("x  (µm)")

    fig.suptitle(
        f"Week 11: Aberrated vs. Diffraction-Limited Printed Feature  (width={width} µm)",
        fontsize=13, fontweight="bold", y=1.0 - 0.15 / n_rows,
    )
    return fig


def print_summary_table(grid, wavelength, NA, width, defocus_values, threshold=0.3):
    """Console summary (also used to fill in the build log's Validation section)."""
    mask = single_line(grid.x, width=width, center=0.0)
    print(f"{'defocus (waves)':>16} | {'EPE (um)':>22} | {'linewidth err (um)':>18}")
    print("-" * 65)
    for defocus_waves in defocus_values:
        intensity, _, _ = incoherent_aerial_image(mask, grid, wavelength=wavelength, NA=NA,
                                                    defocus_waves=defocus_waves)
        printed = apply_threshold(intensity, threshold=threshold)
        epe, *_ = edge_placement_error(mask, printed, grid.x)
        _, _, lwe = linewidth_error(mask, printed, grid.x)
        epe_str = str(np.round(epe, 4)) if len(epe) else "n/a"
        lwe_str = f"{lwe:.4f}" if not np.isnan(lwe) else "n/a"
        print(f"{defocus_waves:>16.1f} | {epe_str:>22} | {lwe_str:>18}")


if __name__ == "__main__":
    grid = Grid1D(L=200.0, N=4096)
    wavelength = WAVELENGTH
    NA = NA_DEFAULT

    # Sweep from unaberrated up through the contrast-reversal regime
    # (confirmed to onset by defocus_waves=1.0 in tests/test_aberrations.py)
    # so both figures visibly span "fine" through "badly out of focus".
    defocus_values = [0.0, 0.5, 1.0, 2.0, 3.0]

    # A single feature width comfortably within the diffraction-limited
    # resolution, so any degradation visible in the figures is attributable
    # to defocus, not to the feature already being unresolvable at best focus.
    width = 1.5

    fig1 = make_atf_otf_defocus_sweep_figure(grid, wavelength, NA, defocus_values)
    fig1.savefig(f"{ASSETS_DIR}/week11_atf_otf_defocus_sweep.png", dpi=150, bbox_inches="tight")

    fig2 = make_aberration_imaging_comparison_figure(grid, wavelength, NA, width, defocus_values)
    fig2.savefig(f"{ASSETS_DIR}/week11_aberration_imaging_comparison.png", dpi=150, bbox_inches="tight")

    print_summary_table(grid, wavelength, NA, width, defocus_values)
    print(
        "\nSaved: assets/week11_atf_otf_defocus_sweep.png, "
        "assets/week11_aberration_imaging_comparison.png"
    )
