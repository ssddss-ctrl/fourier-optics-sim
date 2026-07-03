"""
scripts/generate_diffraction_patterns.py
------------------------------------------
Generates and plots the Week 8 Fraunhofer diffraction pattern library:
single line, line-space grating, and two nearby lines. For each pattern,
overlays the numerical fraunhofer_pattern() output against the analytic
closed-form prediction from physics/diffraction.py, and prints a short
verification summary (zero positions / order spacing / order heights)
confirming the numerical simulator matches Goodman's equations.

This is a one-off generation script per the project convention (plot/
generation scripts live in scripts/, not physics/) -- it contains no new
physics, only calls into physics/masks.py, physics/grid.py, and
physics/diffraction.py, and reuses plotting/core.py's existing color
constants and axis styling rather than redefining them.

Run from the repo root:
    python scripts/generate_diffraction_patterns.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# physics/ and plotting/ modules import each other with flat names
# (e.g. grid.py does `from masks import make_grid`, not
# `from physics.masks import make_grid`) -- confirmed by inspecting the
# actual uploaded files. That means physics/ and plotting/ each need to
# be on sys.path directly, exactly like tests/conftest.py already does,
# not just the repo root as a package.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_repo_root, "physics"))
sys.path.insert(0, os.path.join(_repo_root, "plotting"))

from masks import single_line, line_space_grating, two_lines
from grid import Grid1D
import diffraction as diff
from core import MASK_COLOR, IMAGE_COLOR, _style_ax


ASSETS_DIR = os.path.join(_repo_root, "assets")

# Shared simulation parameters
L, N = 200.0, 8192       # field width (µm), grid points
WAVELENGTH = 0.5         # µm


def _two_panel_figure(x, mask, x_obs, I_numeric, I_analytic, title, obs_xlim=None):
    """
    Build the aperture + far-field-pattern figure shared by all three
    pattern types below.

    WHY THIS EXISTS (rather than reusing plotting.core.four_panel_plot)
    ---------------------------------------------------------------------
    four_panel_plot assumes all four panels share one spatial axis `x`.
    That's true for target/mask/spectrum (all live in the aperture
    plane), but the Fraunhofer far-field pattern lives in a physically
    different plane at a different scale (x_obs = f*lambda*z, typically
    spanning 10-100x the aperture width) -- plotting it against the
    aperture's own x would mislabel two different physical planes as the
    same axis. This function keeps the same color/style conventions from
    plotting.core (MASK_COLOR, IMAGE_COLOR, _style_ax) so the new plots
    still look consistent with Weeks 1-3's figures, without pretending
    the two axes are interchangeable.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("#F8F9FA")

    # Panel 1: aperture (mask) in the aperture plane
    ax1.fill_between(x, mask, alpha=0.8, color=MASK_COLOR, step="mid")
    ax1.plot(x, mask, color=MASK_COLOR, lw=1.2)
    ax1.set_title("Aperture (mask)", fontweight="bold", fontsize=11)
    ax1.set_xlabel("ξ  (µm)  — aperture plane")
    ax1.set_ylabel("Transmission")
    ax1.set_ylim(-0.1, 1.3)
    _style_ax(ax1)

    # Panel 2: Fraunhofer far-field intensity, numeric vs analytic
    ax2.plot(x_obs, I_numeric, color=IMAGE_COLOR, lw=1.8, label="numeric (FFT)")
    ax2.plot(x_obs, I_analytic, color="#8E44AD", lw=1.2, ls="--", label="analytic")
    ax2.set_title("Fraunhofer far-field intensity", fontweight="bold", fontsize=11)
    ax2.set_xlabel("x  (µm)  — observation plane")
    ax2.set_ylabel("I(x) / I(0)")
    if obs_xlim:
        ax2.set_xlim(obs_xlim)
    ax2.legend(fontsize=9, frameon=False)
    _style_ax(ax2)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    return fig


def generate_slit_pattern(grid, wavelength=WAVELENGTH, z=200.0, width=1.0):
    """
    Single-line (slit) Fraunhofer pattern vs. analytic sinc^2 (Goodman
    Eq. 4-26/4-27). Confirms the numerical FFT-based simulator reproduces
    the textbook closed form for the simplest aperture case.
    """
    width = round(width / grid.dx) * grid.dx  # snap to grid, see diffraction tests
    assert diff.check_fraunhofer_validity(width, wavelength, z), (
        f"z={z} µm does not satisfy the Fraunhofer far-field condition "
        f"for width={width} µm at λ={wavelength} µm "
        f"(need z > {diff.fraunhofer_far_field_distance(width, wavelength):.2f} µm)"
    )

    mask = single_line(grid.x, width)
    x_obs, field, I = diff.fraunhofer_pattern(mask, grid, wavelength, z)
    I_analytic = diff.analytic_slit_intensity(x_obs, width, wavelength, z)

    predicted_zero = wavelength * z / width
    idx = np.argmin(np.abs(x_obs - predicted_zero))
    print(f"[slit] width={width:.4f} µm, z={z} µm")
    print(f"  predicted first zero at x = {predicted_zero:.3f} µm, "
          f"numeric I there = {I[idx]:.2e} (should be ~0)")
    print(f"  max |numeric - analytic| = {np.max(np.abs(I - I_analytic)):.4e}")

    fig = _two_panel_figure(
        grid.x, mask, x_obs, I, I_analytic,
        title="Single Line — Fraunhofer Diffraction (sinc² check)",
        obs_xlim=(-5 * predicted_zero, 5 * predicted_zero),
    )
    return fig


def generate_grating_pattern(grid, wavelength=WAVELENGTH, z=50.0, pitch=2.0, duty_cycle=0.5):
    """
    Line-space grating Fraunhofer pattern vs. analytic order positions and
    relative heights (Goodman Eq. 4-34/4-36 for order spacing; Problem
    4-12(a)'s v_k = |c_k|^2 for a square-wave grating's order heights,
    since Goodman's own worked example is for a sinusoidal grating).

    pitch is intentionally left as an exact value with L/pitch an integer
    (not snapped to grid.dx) -- see the note in tests/test_diffraction.py
    on why that alignment matters for order-height accuracy.
    """
    assert grid.L % pitch == 0, "pitch must evenly divide the field width L"
    assert diff.check_fraunhofer_validity(pitch, wavelength, z)

    mask = line_space_grating(grid.x, pitch, duty_cycle)
    x_obs, field, I = diff.fraunhofer_pattern(mask, grid, wavelength, z)

    max_order = 3
    predicted_positions = diff.analytic_grating_order_positions(pitch, wavelength, z, max_order)

    print(f"[grating] pitch={pitch} µm, duty_cycle={duty_cycle}, z={z} µm")
    for order in range(-max_order, max_order + 1):
        x_pred = order * wavelength * z / pitch
        idx = np.argmin(np.abs(x_obs - x_pred))
        rel_analytic = diff.analytic_grating_relative_intensity(order, duty_cycle)
        print(f"  order {order:+d}: predicted x={x_pred:7.3f} µm, "
              f"predicted I/I0={rel_analytic:.4f}, numeric I/I0={I[idx]:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("#F8F9FA")

    ax1.fill_between(grid.x, mask, alpha=0.8, color=MASK_COLOR, step="mid")
    ax1.plot(grid.x, mask, color=MASK_COLOR, lw=1.0)
    ax1.set_xlim(-10, 10)  # zoom in on a few periods; full mask spans L=200 µm
    ax1.set_title("Aperture (line-space grating, zoomed)", fontweight="bold", fontsize=11)
    ax1.set_xlabel("ξ  (µm)  — aperture plane")
    ax1.set_ylabel("Transmission")
    ax1.set_ylim(-0.1, 1.3)
    _style_ax(ax1)

    ax2.plot(x_obs, I, color=IMAGE_COLOR, lw=1.5, label="numeric (FFT)")
    ax2.scatter(predicted_positions,
                [diff.analytic_grating_relative_intensity(o, duty_cycle) for o in range(-max_order, max_order + 1)],
                color="#8E44AD", zorder=5, label="analytic order positions/heights", s=35)
    ax2.set_xlim(-50, 50)
    ax2.set_title("Fraunhofer far-field intensity (grating orders)", fontweight="bold", fontsize=11)
    ax2.set_xlabel("x  (µm)  — observation plane")
    ax2.set_ylabel("I(x) / I(0)")
    ax2.legend(fontsize=9, frameon=False)
    _style_ax(ax2)

    fig.suptitle("Line-Space Grating — Fraunhofer Diffraction (order spacing check)",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    return fig


def generate_two_lines_pattern(grid, wavelength=WAVELENGTH, z=200.0, width=0.5, separation=3.0):
    """
    Two-nearby-lines Fraunhofer pattern vs. analytic envelope x
    interference prediction (single-slit sinc^2 envelope times a
    cos^2(pi*x*separation/(lambda*z)) fringe term, from the shift theorem
    -- Goodman §2.1.3 -- applied to two shifted copies of single_line).
    """
    width = round(width / grid.dx) * grid.dx
    separation = round(separation / grid.dx) * grid.dx
    assert diff.check_fraunhofer_validity(separation, wavelength, z)

    mask = two_lines(grid.x, width, separation)
    x_obs, field, I = diff.fraunhofer_pattern(mask, grid, wavelength, z)

    envelope = diff.analytic_slit_intensity(x_obs, width, wavelength, z)
    fx = x_obs / (wavelength * z)
    interference = np.cos(np.pi * fx * separation) ** 2
    I_analytic = envelope * interference
    I_analytic = I_analytic / I_analytic.max()

    fringe_spacing = wavelength * z / separation
    print(f"[two lines] width={width:.4f} µm, separation={separation:.4f} µm, z={z} µm")
    print(f"  predicted fringe spacing = {fringe_spacing:.3f} µm")
    print(f"  max |numeric - analytic| = {np.max(np.abs(I - I_analytic)):.4e}")

    envelope_zero = wavelength * z / width
    fig = _two_panel_figure(
        grid.x, mask, x_obs, I, I_analytic,
        title="Two Nearby Lines — Fraunhofer Diffraction (envelope × interference check)",
        obs_xlim=(-envelope_zero, envelope_zero),
    )
    return fig


def main():
    os.makedirs(ASSETS_DIR, exist_ok=True)
    grid = Grid1D(L, N)
    print(f"Grid: L={grid.L} µm, N={grid.N}, dx={grid.dx:.5g} µm\n")

    fig1 = generate_slit_pattern(grid)
    path1 = os.path.join(ASSETS_DIR, "week8_slit_fraunhofer.png")
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    print(f"  saved -> {path1}\n")

    fig2 = generate_grating_pattern(grid)
    path2 = os.path.join(ASSETS_DIR, "week8_grating_fraunhofer.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"  saved -> {path2}\n")

    fig3 = generate_two_lines_pattern(grid)
    path3 = os.path.join(ASSETS_DIR, "week8_two_lines_fraunhofer.png")
    fig3.savefig(path3, dpi=150, bbox_inches="tight")
    print(f"  saved -> {path3}\n")

    print("All three pattern types generated and plotted successfully.")


if __name__ == "__main__":
    main()