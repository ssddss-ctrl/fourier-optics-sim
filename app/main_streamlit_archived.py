"""
Retired — replaced by frontend/ + backend/, kept for reference. See README for migration notes.

app/main.py
-----------
Fourier Optics Lithography Simulator — Streamlit App

Run with:
    streamlit run app/main.py

Feature roadmap (unlocks as physics modules are built each week):
    Week 1  ✅  Mask pattern design + spectrum viewer
    Week 6  🔜  FFT engine with correct physical units + sampling diagnostics
    Week 7  🔜  Angular spectrum propagation
    Week 8  🔜  Fraunhofer diffraction + pattern library
    Week 9  ✅  Full coherent imaging pipeline (mask → lens → aerial image)
    Week 10 ✅  Coherent vs incoherent mode (ATF vs OTF) + thresholding + EPE/linewidth error
    Week 11 ✅  Defocus aberration (ATF/OTF degradation, aberrated aerial image)
    Week 12 🔜  OPC correction loop
"""

import streamlit as st
import numpy as np
import sys, os

# Flat (non-src) repo layout: physics/*.py and plotting/*.py import each
# other flatly (e.g. lens.py does `from masks import make_grid`, not
# `from physics.masks import ...`), matching tests/conftest.py's existing
# convention. The original version of this file only added the repo root
# to sys.path, which happened to work because masks.py and plotting/core.py
# have no internal cross-module imports -- but it silently breaks on
# physics/grid.py, physics/lens.py, and physics/imaging.py, all of which
# DO import sibling modules flatly. Confirmed directly: `from physics.grid
# import Grid1D` raised `ModuleNotFoundError: No module named 'masks'`
# under the old path setup. Adding physics/ and plotting/ individually
# fixes it, and switching this file's own imports to the same flat style
# keeps one consistent convention across the whole project instead of two.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "physics"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "plotting"))

from masks import single_line, line_space_grating
from grid import Grid1D
from lens import coherent_aerial_image, cutoff_frequency
from imaging import (
    incoherent_aerial_image,
    optical_transfer_function,
    apply_threshold,
    edge_placement_error,
    linewidth_error,
)
from core import three_panel_plot
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Importing interactive registers plotting/interactive.py's dark Plotly
# template as pio.templates.default (module-level side effect, same as
# plotting/core.py's style constants being module-level) -- every go.Figure()
# built below picks it up with no per-figure boilerplate. Pattern Design
# (Week 1, panel ①) is the one panel NOT converted this week -- it still
# goes through core.py/matplotlib via three_panel_plot/st.pyplot above.
from interactive import (
    PRIMARY_COLOR,
    PRIMARY_FILL,
    PHASE_COLOR,
    TARGET_COLOR,
    TARGET_FILL,
    REFERENCE_LINE_COLOR,
    add_reference_hline,
    add_reference_vline,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fourier Optics Lithography Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Fourier Optics Lithography Simulator")
st.caption("Built week-by-week alongside Goodman's *Introduction to Fourier Optics*")

# ── Sidebar — Optical Parameters ──────────────────────────────────────────────
st.sidebar.header("Optical Parameters")

st.sidebar.subheader("Illumination")
wavelength_nm = st.sidebar.number_input(
    "Wavelength λ (nm)", value=193.0, min_value=10.0, max_value=800.0,
    help="193 nm = ArF excimer (standard DUV). 13.5 nm = EUV."
)
wavelength = wavelength_nm / 1000.0  # physics/ modules work in µm throughout

NA = st.sidebar.slider(
    "Numerical Aperture (NA)", min_value=0.1, max_value=1.4, value=0.75, step=0.05,
    help="Higher NA = finer resolution / larger pupil cutoff (f0 = NA/λ)."
)

st.sidebar.subheader("Coherence")
coherence = st.sidebar.radio(
    "Illumination mode", ["Coherent", "Incoherent"],
    help=(
        "Coherent = laser-like, linear in amplitude (ATF). "
        "Incoherent = extended source, linear in intensity (OTF, "
        "passband extends to 2x the coherent cutoff)."
    ),
)

st.sidebar.markdown("---")
st.sidebar.subheader("Printed Feature / Thresholding")
threshold = st.sidebar.slider(
    "Resist threshold (fraction of clear-field intensity)",
    min_value=0.05, max_value=0.95, value=0.3, step=0.05,
    help=(
        "Constant-threshold resist model: intensity at or above this "
        "fraction of the nominal clear-field value (1.0) prints; below "
        "it does not. Not a Goodman equation -- standard lithography "
        "engineering convention."
    ),
)

st.sidebar.markdown("---")
st.sidebar.subheader("Aberrations")
focus_error = st.sidebar.slider(
    "Focus error (waves)", min_value=0.0, max_value=2.0, value=0.0, step=0.1,
    help=(
        "Peak defocus wavefront error, in units of wavelength (0 = "
        "diffraction-limited). Feeds into both the ATF/OTF panel and the "
        "aerial image below -- pure quadratic (defocus) aberration only, "
        "per physics/aberrations.py."
    ),
)

# ── Main panel — Pattern Design (Week 1, active) ─────────────────────────────
st.header("① Pattern Design")

col1, col2 = st.columns([1, 2])

with col1:
    pattern_type = st.selectbox(
        "Pattern type",
        ["Isolated Line", "Line-Space Grating"],
    )

    st.subheader("Grid")
    L = st.number_input("Field width L (µm)", value=10.0, min_value=1.0, max_value=100.0)
    N = st.selectbox("Grid points N", [512, 1024, 2048, 4096], index=1)

    # Grid1D bundles the spatial grid with the frequency axis / dx / f_max
    # that every downstream physics module (lens.py, imaging.py) expects --
    # built once here (pattern-independent) and reused by every panel below.
    grid = Grid1D(L=L, N=N)
    x = grid.x

    st.subheader("Feature")

    # All pattern_type-dependent work -- widget collection, mask/target
    # generation, and the sampling-diagnostic message -- is done together
    # in this ONE branch, rather than in three separate `if pattern_type
    # == ...` checks scattered through the file (widgets here, mask-
    # building in what used to be col2, diagnostics in the expander). The
    # original file had exactly that three-way split, which is what
    # produced Pylance's reportPossiblyUnboundVariable warnings on
    # feature_width/pitch/duty_cycle: Pylance can't prove pattern_type is
    # invariant across separate statements, so a variable set in only one
    # branch of an earlier check looks possibly-unbound by the time a
    # later, separately-evaluated check uses it. Not a runtime bug
    # (Streamlit reruns top-to-bottom each interaction, so pattern_type
    # never actually changes mid-run) -- but collapsing to a single branch
    # removes the warning at its root instead of papering over it with
    # `= None` defaults (which was tried first here and, correctly,
    # immediately flagged its own new error: 1.0 / possibly-None).
    if pattern_type == "Isolated Line":
        feature_width = st.number_input(
            "Line width w (µm)", value=1.0, min_value=0.05, max_value=L/2, step=0.05
        )
        mask = single_line(x, width=feature_width)
        target = single_line(x, width=feature_width)
        plot_title = f"Isolated Line — w = {feature_width} µm"

        f_first_zero = 1.0 / feature_width
        nyquist_ok = f_first_zero < grid.f_max * 0.8
        diagnostic_message = (
            f"First spectral zero at f = 1/w = {f_first_zero:.2f} cycles/µm  "
            f"{'✅ well within Nyquist' if nyquist_ok else '⚠️ approaching Nyquist — consider finer grid'}"
        )
    else:
        pitch = st.number_input(
            "Pitch p (µm)", value=2.0, min_value=0.1, max_value=L/2, step=0.1
        )
        duty_cycle = st.slider("Duty cycle", 0.1, 0.9, 0.5, 0.05)
        mask = line_space_grating(x, pitch=pitch, duty_cycle=duty_cycle)
        target = line_space_grating(x, pitch=pitch, duty_cycle=duty_cycle)
        plot_title = f"Line-Space Grating — pitch = {pitch} µm, DC = {duty_cycle}"

        f_first_order = 1.0 / pitch
        diagnostic_message = f"First grating order at f = 1/p = {f_first_order:.2f} cycles/µm"

    spectrum_zoom = st.slider(
        "Spectrum zoom (±cycles/µm)", min_value=1.0, max_value=20.0, value=5.0
    )

with col2:
    fig = three_panel_plot(
        x, target, mask,
        title=plot_title,
        spectrum_xlim=(-spectrum_zoom, spectrum_zoom),
    )
    st.pyplot(fig)
    plt.close()

# ── Grid diagnostics ─────────────────────────────────────────────────────────
with st.expander("Grid & Sampling Diagnostics"):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("dx", f"{grid.dx*1000:.1f} nm")
    c2.metric("f_max (Nyquist)", f"{grid.f_max:.1f} cyc/µm")
    c3.metric("df (freq resolution)", f"{grid.df:.3f} cyc/µm")
    c4.metric("N", f"{grid.N}")

    st.info(diagnostic_message)

# ── ATF / OTF (Week 11, aberrations — active) ────────────────────────────────
st.markdown("---")
st.header("② ATF / OTF")
st.caption(
    f"Focus error = {focus_error:.1f} waves — "
    + ("diffraction-limited (no aberration)" if focus_error == 0.0
       else "defocused (build notes Eq. 14/16, physics/aberrations.py)")
)

OTF, H = optical_transfer_function(grid, wavelength=wavelength, NA=NA, defocus_waves=focus_error)
f0 = cutoff_frequency(NA, wavelength)

col_atf, col_otf = st.columns(2)

with col_atf:
    # Phase of an exactly-zero pupil is undefined, not physically zero --
    # masked to NaN outside the pupil support rather than plotted as a
    # misleading flat line (same convention as
    # scripts/generate_aberration_comparison.py's ATF phase panel).
    support = np.abs(H) > 0
    phase = np.full_like(grid.f, np.nan)
    phase[support] = np.angle(H[support])

    fig_atf = make_subplots(specs=[[{"secondary_y": True}]])
    fig_atf.add_trace(
        go.Scatter(
            x=grid.f, y=np.abs(H), mode="lines", name="|H| (ATF magnitude)",
            line=dict(color=PRIMARY_COLOR, width=2),
            hovertemplate="fx = %{x:.3f} cyc/µm<br>|H| = %{y:.4f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig_atf.add_trace(
        go.Scatter(
            x=grid.f, y=phase, mode="lines", name="angle(H) (phase)",
            line=dict(color=PHASE_COLOR, width=1.6, dash="dash"),
            hovertemplate="fx = %{x:.3f} cyc/µm<br>phase = %{y:.4f} rad<extra></extra>",
        ),
        secondary_y=True,
    )
    add_reference_vline(fig_atf, f0)
    add_reference_vline(fig_atf, -f0)
    fig_atf.update_layout(
        title="ATF: magnitude & phase", height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig_atf.update_xaxes(title_text="fx  (cycles/µm)")
    fig_atf.update_yaxes(title_text="|H|  (ATF magnitude)", secondary_y=False)
    fig_atf.update_yaxes(title_text="ATF phase (radians)", secondary_y=True)
    st.plotly_chart(fig_atf, width="stretch")

with col_otf:
    fig_otf = go.Figure()
    fig_otf.add_trace(
        go.Scatter(
            x=grid.f, y=np.abs(OTF), mode="lines", name="|OTF|",
            line=dict(color=PRIMARY_COLOR, width=2),
            hovertemplate="fx = %{x:.3f} cyc/µm<br>|OTF| = %{y:.4f}<extra></extra>",
        )
    )
    add_reference_vline(fig_otf, f0)
    add_reference_vline(fig_otf, -f0)
    fig_otf.update_layout(title="MTF (OTF magnitude)", height=380, showlegend=False)
    fig_otf.update_xaxes(title_text="fx  (cycles/µm)")
    fig_otf.update_yaxes(title_text="OTF magnitude  |H_otf(fx)|", range=[-0.05, 1.15])
    st.plotly_chart(fig_otf, width="stretch")

if focus_error > 0.0 and np.any(OTF.real < -1e-6):
    st.warning(
        "⚠️ Contrast reversal: the OTF goes negative at some spatial "
        "frequencies at this defocus (Goodman 6.4.3) -- expect spurious "
        "resolution / inverted contrast in the aerial image below."
    )

st.caption(
    "Cutoff frequency (dotted lines) is unchanged by defocus -- only the "
    "phase across the pupil (left) and the resulting contrast at each "
    "frequency (right) degrade."
)

# ── Imaging pipeline: aerial image + printed feature (Week 9/10/11, active) ─
st.markdown("---")

# OPC (⑤) is still a Week 12 placeholder with no real content -- narrower
# than the two panels that actually have live plots/metrics, so Aerial
# Image and Printed Feature get the room they need instead of three equal
# thirds crowding the printed-feature metrics into a cramped last column.
col_a, col_b, col_c = st.columns([5, 5, 2])

# Both panels below share the same aerial image computation, so it's run
# once here rather than twice (once per panel).
if coherence == "Coherent":
    _, intensity, _ = coherent_aerial_image(mask, grid, wavelength=wavelength, NA=NA,
                                             defocus_waves=focus_error)
else:
    intensity, _, _ = incoherent_aerial_image(mask, grid, wavelength=wavelength, NA=NA,
                                               defocus_waves=focus_error)

printed = apply_threshold(intensity, threshold=threshold)

with col_a:
    st.header("③ Aerial Image")
    st.caption(
        f"{coherence} illumination — NA={NA}, λ={wavelength_nm:.0f} nm"
        + (f", focus error={focus_error:.1f} waves" if focus_error > 0.0 else "")
    )

    fig_img = go.Figure()
    fig_img.add_trace(
        go.Scatter(
            x=x, y=intensity, mode="lines", name="intensity",
            line=dict(color=PRIMARY_COLOR, width=2),
            fill="tozeroy", fillcolor=PRIMARY_FILL,
            hovertemplate="x = %{x:.3f} µm<br>Intensity = %{y:.4f}<extra></extra>",
        )
    )
    add_reference_hline(fig_img, threshold, annotation_text="threshold",
                         annotation_position="top left",
                         annotation_font_color=REFERENCE_LINE_COLOR)
    fig_img.update_layout(height=380, showlegend=False)
    fig_img.update_xaxes(title_text="x  (µm)")
    fig_img.update_yaxes(title_text="Intensity (a.u.)")
    st.plotly_chart(fig_img, width="stretch")

    st.caption(
        "Coherent imaging is linear in amplitude (can overshoot 1.0 -- "
        "Gibbs-like ringing at edges, Eq. 6-20). Incoherent imaging is "
        "linear in intensity (Eq. 6-9/6-26), no ringing, but blurs edges "
        "differently."
    )

with col_b:
    st.header("④ Printed Feature vs. Target")
    st.caption(f"Threshold = {threshold}")

    fig_p = go.Figure()
    fig_p.add_trace(
        go.Scatter(
            x=x, y=target, mode="lines", name="target",
            line=dict(color=TARGET_COLOR, width=1.2, shape="hvh"),
            fill="tozeroy", fillcolor=TARGET_FILL,
            hovertemplate="x = %{x:.3f} µm<br>target = %{y:.0f}<extra></extra>",
        )
    )
    fig_p.add_trace(
        go.Scatter(
            x=x, y=printed, mode="lines", name="printed",
            line=dict(color=PRIMARY_COLOR, width=2, shape="hvh"),
            hovertemplate="x = %{x:.3f} µm<br>printed = %{y:.0f}<extra></extra>",
        )
    )
    fig_p.update_layout(
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig_p.update_xaxes(title_text="x  (µm)")
    fig_p.update_yaxes(title_text="Transmission", range=[-0.1, 1.3])
    st.plotly_chart(fig_p, width="stretch")

    epe, target_edges, printed_edges = edge_placement_error(target, printed, x)

    if len(target_edges) == 0:
        st.warning("No edges found in target pattern.")
    elif np.all(np.isnan(epe)):
        st.error("Feature did not print at this threshold (no printed edges found).")
    else:
        # EPE and linewidth metrics grouped into ONE row (EPE pair first,
        # linewidth pair after) rather than two stacked st.columns(2) rows --
        # keeps this panel's total height in line with the Aerial Image
        # panel beside it instead of running visibly longer.
        width_err = np.nan
        if pattern_type == "Isolated Line":
            printed_w, target_w, width_err = linewidth_error(target, printed, x)

        if pattern_type == "Isolated Line" and not np.isnan(width_err):
            m1, m2 = st.columns(2)
            m1.metric("Max |EPE|", f"{np.nanmax(np.abs(epe)):.4f} µm")
            m2.metric("Mean |EPE|", f"{np.nanmean(np.abs(epe)):.4f} µm")
            m3, m4 = st.columns(2)
            m3.metric("Target linewidth", f"{target_w:.4f} µm")
            m4.metric("Printed linewidth", f"{printed_w:.4f} µm",
                      delta=f"{width_err:+.4f} µm")
        else:
            m1, m2 = st.columns(2)
            m1.metric("Max |EPE|", f"{np.nanmax(np.abs(epe)):.4f} µm")
            m2.metric("Mean |EPE|", f"{np.nanmean(np.abs(epe)):.4f} µm")

            if pattern_type == "Isolated Line":
                st.warning(
                    "Printed pattern doesn't have exactly 2 edges (feature "
                    "failed to resolve cleanly at this threshold/NA) -- "
                    "linewidth error not well-defined here."
                )
            else:
                st.caption(
                    "Linewidth error is only reported for an isolated single "
                    "line (a grating has multiple ambiguous 'widths'); EPE "
                    "above covers every edge in the grating."
                )

with col_c:
    st.header("⑤ OPC")
    st.info(
        "🔜 **Week 12**\n\n"
        "Will iteratively adjust the mask to minimize edge placement "
        "error between target and printed feature."
    )