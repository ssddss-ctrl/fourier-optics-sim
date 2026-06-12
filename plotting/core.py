"""
plotting/core.py
----------------
Reusable 4-panel plotting scaffold for the Fourier Optics simulator.

The four panels we'll use throughout this project:
  1. Target   — what we want to print (the ideal pattern)
  2. Mask     — what we put on the reticle (input to optical system)
  3. Spectrum — Fourier transform of the mask (frequency content)
  4. Image    — aerial image at wafer plane (output of optical system)

In Week 1, panels 1–3 are populated; panel 4 is a placeholder until
we build the propagation engine (Week 7+).

All spatial units: µm
All frequency units: cycles/µm (µm⁻¹)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── Style constants ──────────────────────────────────────────────────────────
MASK_COLOR    = "#2C3E50"   # dark blue-grey for mask bars
TARGET_COLOR  = "#E74C3C"   # red for target/desired pattern
SPECTRUM_COLOR = "#2980B9"  # blue for spectrum magnitude
IMAGE_COLOR   = "#27AE60"   # green for aerial image intensity
PLACEHOLDER_ALPHA = 0.3


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


def compute_spectrum(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the magnitude spectrum of a 1D mask array.

    Returns
    -------
    freqs  : ndarray — frequency axis (cycles/µm), centered
    magnitude : ndarray — |FT(mask)|, normalized by N so peak ≈ duty cycle

    Notes
    -----
    We normalize by N (number of points) so the DC component equals the
    spatial average of the mask (e.g., 0.5 for a 50% duty cycle grating).
    This normalization makes the spectrum physically meaningful regardless
    of grid size.

    The phase spectrum is computed and accessible if you call np.fft.fftshift
    on np.fft.fft(mask) directly — we'll use that in later weeks.
    """
    N = len(mask)
    spectrum = np.fft.fftshift(np.fft.fft(mask)) / N
    magnitude = np.abs(spectrum)
    # Build frequency axis from x spacing — we need x here.
    # We'll accept freqs as a separate argument in the full function below.
    return magnitude


def four_panel_plot(
    x: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    image: np.ndarray | None = None,
    title: str = "",
    spectrum_xlim: tuple | None = None,
) -> plt.Figure:
    """
    Generate the standard 4-panel figure for this project.

    Layout
    ------
    [Target] [Mask]
    [Spectrum] [Image (or placeholder)]

    Parameters
    ----------
    x       : 1D spatial grid (µm)
    target  : desired pattern array (0/1)
    mask    : mask transmission array (0/1)
    image   : aerial image intensity array, or None (placeholder shown)
    title   : suptitle for the figure
    spectrum_xlim : (fmin, fmax) to zoom the spectrum axis, or None for auto

    Returns
    -------
    fig : matplotlib Figure
    """
    freqs = _freq_axis(x)
    spec_mag = np.abs(np.fft.fftshift(np.fft.fft(mask)) / len(mask))

    fig = plt.figure(figsize=(12, 7))
    fig.patch.set_facecolor("#F8F9FA")
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

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
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.stem(
        freqs, spec_mag,
        linefmt="C0-", markerfmt="C0o", basefmt="k-",
    )
    ax3.set_title("③ Mask Spectrum  |G(f)|", fontweight="bold", fontsize=11)
    ax3.set_xlabel("Spatial frequency  (cycles/µm)")
    ax3.set_ylabel("|G(f)|  (normalized)")
    if spectrum_xlim:
        ax3.set_xlim(spectrum_xlim)
    _style_ax(ax3)

    # ── Panel 4: Aerial Image (or placeholder) ───────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    if image is not None:
        ax4.plot(x, image, color=IMAGE_COLOR, lw=1.8)
        ax4.fill_between(x, image, alpha=0.4, color=IMAGE_COLOR)
        ax4.set_title("④ Aerial Image  (intensity)", fontweight="bold", fontsize=11)
        ax4.set_ylabel("Intensity (a.u.)")
    else:
        ax4.text(
            0.5, 0.5,
            "Aerial Image\n(Week 9+)",
            ha="center", va="center",
            fontsize=13, color="grey", style="italic",
            transform=ax4.transAxes,
        )
        ax4.set_title("④ Aerial Image", fontweight="bold", fontsize=11, color="grey")
        ax4.set_facecolor("#EFEFEF")
        ax4.set_alpha(PLACEHOLDER_ALPHA)
    ax4.set_xlabel("x  (µm)")
    ax4.set_xlim(x[0], x[-1])
    _style_ax(ax4)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    return fig


def _style_ax(ax: plt.Axes) -> None:
    """Apply consistent axis styling across all panels."""
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.7)
    ax.tick_params(labelsize=9)
