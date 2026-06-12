"""
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
    Week 9  🔜  Full coherent imaging pipeline (mask → lens → aerial image)
    Week 10 🔜  Coherent vs incoherent mode (ATF vs OTF)
    Week 11 🔜  Aberrations + focus error sweep
    Week 12 🔜  OPC correction loop
"""

import streamlit as st
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from physics.masks import make_grid, single_line, line_space_grating, grid_info
from plotting.core import four_panel_plot
import matplotlib.pyplot as plt

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fourier Optics Lithography Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Fourier Optics Lithography Simulator")
st.caption("Built week-by-week alongside Goodman's *Introduction to Fourier Optics*")

# ── Sidebar — Optical Parameters (most locked until later weeks) ─────────────
st.sidebar.header("Optical Parameters")

st.sidebar.subheader("Illumination")
wavelength = st.sidebar.number_input(
    "Wavelength λ (nm)", value=193.0, min_value=10.0, max_value=800.0,
    help="193 nm = ArF excimer (standard DUV). 13.5 nm = EUV. Unlocks in Week 9."
)
NA = st.sidebar.slider(
    "Numerical Aperture (NA)", min_value=0.1, max_value=1.4, value=0.75, step=0.05,
    help="Higher NA = finer resolution. Unlocks in Week 9."
)

st.sidebar.subheader("Coherence")
coherence = st.sidebar.radio(
    "Illumination mode", ["Coherent", "Incoherent"],
    help="Coherent = laser-like. Incoherent = extended source. Unlocks in Week 10."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Aberrations")
focus_error = st.sidebar.slider(
    "Focus error (waves)", min_value=0.0, max_value=2.0, value=0.0, step=0.1,
    help="Defocus in units of wavelength. Unlocks in Week 11."
)

# ── Main panel — Pattern Design (Week 1, active now) ─────────────────────────
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

    st.subheader("Feature")
    if pattern_type == "Isolated Line":
        feature_width = st.number_input(
            "Line width w (µm)", value=1.0, min_value=0.05, max_value=L/2, step=0.05
        )
    else:
        pitch = st.number_input(
            "Pitch p (µm)", value=2.0, min_value=0.1, max_value=L/2, step=0.1
        )
        duty_cycle = st.slider("Duty cycle", 0.1, 0.9, 0.5, 0.05)

    spectrum_zoom = st.slider(
        "Spectrum zoom (±cycles/µm)", min_value=1.0, max_value=20.0, value=5.0
    )

with col2:
    # Build mask
    x = make_grid(L, N)
    info = grid_info(x)

    if pattern_type == "Isolated Line":
        mask = single_line(x, width=feature_width)
        target = single_line(x, width=feature_width)
        plot_title = f"Isolated Line — w = {feature_width} µm"
    else:
        mask = line_space_grating(x, pitch=pitch, duty_cycle=duty_cycle)
        target = line_space_grating(x, pitch=pitch, duty_cycle=duty_cycle)
        plot_title = f"Line-Space Grating — pitch = {pitch} µm, DC = {duty_cycle}"

    fig = four_panel_plot(
        x, target, mask,
        title=plot_title,
        spectrum_xlim=(-spectrum_zoom, spectrum_zoom),
    )
    st.pyplot(fig)
    plt.close()

# ── Grid diagnostics ─────────────────────────────────────────────────────────
with st.expander("Grid & Sampling Diagnostics"):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("dx", f"{info['dx']*1000:.1f} nm")
    c2.metric("f_max (Nyquist)", f"{info['f_max']:.1f} cyc/µm")
    c3.metric("df (freq resolution)", f"{info['df']:.3f} cyc/µm")
    c4.metric("N", f"{info['N']}")

    if pattern_type == "Isolated Line":
        f_first_zero = 1.0 / feature_width
        nyquist_ok = f_first_zero < info["f_max"] * 0.8
        st.info(
            f"First spectral zero at f = 1/w = {f_first_zero:.2f} cycles/µm  "
            f"{'✅ well within Nyquist' if nyquist_ok else '⚠️ approaching Nyquist — consider finer grid'}"
        )
    else:
        f_first_order = 1.0 / pitch
        st.info(f"First grating order at f = 1/p = {f_first_order:.2f} cycles/µm")

# ── Locked future panels ──────────────────────────────────────────────────────
st.markdown("---")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.header("② Aerial Image")
    st.info("🔜 Unlocks Week 9\n\nWill show intensity pattern at the wafer plane after propagation through the lens with NA cutoff applied.")

with col_b:
    st.header("③ Printed Feature")
    st.info("🔜 Unlocks Week 10\n\nWill apply intensity threshold to aerial image to predict the physical feature that gets etched into the resist.")

with col_c:
    st.header("④ OPC Correction")
    st.info("🔜 Unlocks Week 12\n\nWill iteratively adjust the mask to minimize edge placement error between target and printed feature.")
