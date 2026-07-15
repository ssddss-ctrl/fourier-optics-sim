"""
plotting/core.py
----------------
Reusable 3-panel plotting scaffold for the Fourier Optics simulator.

The three panels shown here describe the mask DESIGN side of the pipeline:
  1. Target   — what we want to print (the ideal pattern)
  2. Mask     — what we put on the reticle (input to optical system)
  3. Spectrum — Fourier transform of the mask (frequency content)

This used to be a 4-panel scaffold with a 4th "Aerial Image" quadrant that
was a locked placeholder from Week 1 through Week 9. As of Week 10, the
actual aerial image (coherent or incoherent, live) and the printed-feature/
EPE comparison are their own dedicated panels in app/main.py (using
physics/lens.py and physics/imaging.py directly) -- they show what the
optical SYSTEM does to the mask, which is a conceptually different thing
from this module's job of showing what was DESIGNED. Keeping a 4th
"Aerial Image" quadrant here after that panel existed elsewhere would just
duplicate it (or worse, show a stale copy), so it was dropped rather than
wired up twice.

All spatial units: µm
All frequency units: cycles/µm (µm⁻¹)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.axes import Axes
# Figure/Axes imported directly from their defining submodules rather than
# used as plt.Figure / plt.Axes -- Pylance's matplotlib stubs don't export
# either name from matplotlib.pyplot (reportPrivateImportUsage), the same
# issue already hit and fixed this way in the Week 9 build log
# (physics/lens.py's docstring notes the identical fix for plt.Figure).


# ── Style constants ──────────────────────────────────────────────────────────
MASK_COLOR    = "#2C3E50"   # dark blue-grey for mask bars
TARGET_COLOR  = "#E74C3C"   # red for target/desired pattern
SPECTRUM_COLOR = "#2980B9"  # blue for spectrum magnitude
IMAGE_COLOR   = "#27AE60"   # green for aerial image intensity (used by
                             # app/main.py's own live Aerial Image panel)
SPINE_COLOR   = "#B0B4B8"   # soft grey for the remaining axis spines


def _freq_axis(x: np.ndarray) -> np.ndarray:
    """
    Compute the physical frequency axis (cycles/µm) corresponding to x.

    Uses np.fft.fftfreq and fftshift so the axis runs from -f_max to +f_max.
    This is the standard way to get a centered, physical frequency axis.

    np.fft.fftfreq(N, d=dx) returns frequencies in cycles per unit of dx.
    fftshift reorders from [0, ..., f_max, -f_max, ..., -df] to
    [-f_max, ..., -df, 0, df, ..., f_max].
    """
    N = len(x)
    dx = x[1] - x[0]
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    return freqs


def three_panel_plot(
    x: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    title: str = "",
    spectrum_xlim: tuple | None = None,
) -> Figure:
    """
    Generate the standard 3-panel mask-design figure for this project.

    Layout
    ------
    [Target] [Mask] [Spectrum]

    (Formerly a 2x2 "four_panel_plot" with a 4th Aerial Image quadrant
    that was a locked placeholder from Week 1 through Week 9. Renamed and
    the 4th quadrant dropped in Week 10, once the real aerial image /
    printed-feature panels existed as their own live panels in
    app/main.py -- see this module's docstring.)

    Parameters
    ----------
    x       : 1D spatial grid (µm)
    target  : desired pattern array (0/1)
    mask    : mask transmission array (0/1)
    title   : suptitle for the figure
    spectrum_xlim : (fmin, fmax) to zoom the spectrum axis, or None for auto

    Returns
    -------
    fig : matplotlib Figure
    """
    freqs = _freq_axis(x)
    spec_mag = np.abs(np.fft.fftshift(np.fft.fft(mask)) / len(mask))

    fig = plt.figure(figsize=(15, 3.7))
    fig.patch.set_facecolor("#F8F9FA")
    gs = gridspec.GridSpec(1, 3, wspace=0.3)

    # ── Panel 1: Target ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.fill_between(x, target, alpha=0.7, color=TARGET_COLOR, step="mid")
    ax1.plot(x, target, color=TARGET_COLOR, lw=1.2)
    ax1.set_title("① Target Pattern", fontweight="bold", fontsize=11)
    ax1.set_xlabel("x  (µm)")
    ax1.set_ylabel("Transmission")
    ax1.set_ylim(-0.1, 1.3)
    ax1.set_xlim(x[0], x[-1])
    _style_ax(ax1)

    # ── Panel 2: Mask ────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill_between(x, mask, alpha=0.8, color=MASK_COLOR, step="mid")
    ax2.plot(x, mask, color=MASK_COLOR, lw=1.2)
    ax2.set_title("② Mask", fontweight="bold", fontsize=11)
    ax2.set_xlabel("x  (µm)")
    ax2.set_ylabel("Transmission")
    ax2.set_ylim(-0.1, 1.3)
    ax2.set_xlim(x[0], x[-1])
    _style_ax(ax2)

    # ── Panel 3: Spectrum ────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.stem(
        freqs, spec_mag,
        linefmt="C0-", markerfmt="C0o", basefmt=SPINE_COLOR,
    )
    ax3.set_title("③ Mask Spectrum  |G(f)|", fontweight="bold", fontsize=11)
    ax3.set_xlabel("Spatial frequency  (cycles/µm)")
    ax3.set_ylabel("|G(f)|  (normalized)")
    if spectrum_xlim:
        ax3.set_xlim(spectrum_xlim)
    _style_ax(ax3)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.05)

    return fig


def _style_ax(ax: Axes) -> None:
    """
    Apply consistent, minimal axis styling across all panels.

    Aesthetic pass (Week 10): dropped the dashed gridlines entirely --
    with filled/stem plots against a light background, gridlines mostly
    added visual noise rather than aiding readability. Kept only the
    bottom and left spines (top/right already removed), softened to a
    light grey rather than full black so they recede behind the data
    instead of competing with it, and lightened tick label size slightly.
    """
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.tick_params(labelsize=9, colors="#444444")