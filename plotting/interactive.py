"""
plotting/interactive.py
--------------------------
Shared Plotly dark theme for app/main.py's live, hover-enabled charts.

Parallel to plotting/core.py (matplotlib), NOT a replacement for it:
plotting/core.py and every scripts/generate_*.py file keep using matplotlib
exactly as before, for the static PNGs that go into the build log. This
module is consumed ONLY by app/main.py, for the panels a user actually
interacts with in the browser (Aerial Image, Printed Feature vs. Target,
ATF/OTF) -- the reason to prefer Plotly there and nowhere else is hover
tooltips on live data, which a static PNG has no use for.

WHY A REGISTERED PLOTLY TEMPLATE, NOT A PER-FIGURE STYLING FUNCTION
------------------------------------------------------------------------
plotting/core.py's _style_ax is a function called once per matplotlib Axes,
because matplotlib has no first-class "shared theme" object. Plotly does
(go.layout.Template): defining one here and registering it with
plotly.io once means every fig = go.Figure() constructed in app/main.py
picks up the dark background, font, gridlines, and axis-line color
automatically (set as pio.templates.default), the same way every matplotlib
panel in this project already shares plotting/core.py's SPINE_COLOR/
_style_ax. Panel-specific titles/labels are still set per figure; only the
chrome that should look identical everywhere lives in the template.

COLOR PALETTE -- VALIDATED, NOT EYEBALLED
------------------------------------------------------------------------
Chosen and checked against the project's dataviz skill (OKLCH lightness
band, chroma floor, simulated-CVD delta-E, normal-vision delta-E, WCAG
contrast), via that skill's scripts/validate_palette.js, mode=dark,
surface=#1a1a19 (the skill's own validated dark chart-surface reference).
The checks are pairwise and scoped to colors that actually appear together
in the SAME chart (not every color against every other color in the app):

  - PRIMARY_COLOR (blue) + PHASE_COLOR (green): co-occur in the ATF panel
    (magnitude on the primary y-axis, phase on the secondary). Validated
    pass -- normal-vision delta-E 29.9 (floor is 15).
  - PRIMARY_COLOR (blue) + TARGET_COLOR (red): co-occur in the Printed
    Feature panel (printed line vs. target fill). Validated pass --
    normal-vision delta-E 29.0.

A blue+violet pairing was tried first for the ATF panel and FAILED the
normal-vision floor outright (delta-E 9.8, well under the 15 floor) --
concrete evidence for why this gets run through the validator instead of
picked by eye. REFERENCE_LINE_COLOR (threshold / cutoff-frequency marker
lines) is deliberately NOT part of the categorical set -- it's an
annotation color, not a third data series competing for identity.

All spatial coordinates: µm
All spatial frequencies: cycles/µm (µm⁻¹)
"""

import plotly.graph_objects as go
import plotly.io as pio

# ── Surfaces / chrome (dataviz skill's dark reference palette) ──────────────
CHART_SURFACE   = "#1a1a19"   # paper_bgcolor / plot_bgcolor for every chart
PAGE_PLANE      = "#0d0d0d"   # Streamlit app background (see .streamlit/config.toml)
PRIMARY_INK     = "#ffffff"   # chart titles
SECONDARY_INK   = "#c3c2b7"   # axis titles, legend text, general chart font
MUTED_INK       = "#898781"   # tick labels
GRIDLINE_COLOR  = "#2c2c2a"   # hairline gridlines
AXIS_LINE_COLOR = "#383835"   # axis baseline / zeroline

FONT_FAMILY = "system-ui, -apple-system, 'Segoe UI', sans-serif"

# ── Categorical series colors (validated -- see module docstring) ───────────
PRIMARY_COLOR = "#3987e5"
"""Blue. The "measured/simulated signal" color, reused across every panel
that plots one such signal on its own: aerial image intensity, ATF
magnitude, OTF magnitude, and the printed-feature line."""

PHASE_COLOR = "#008300"
"""Green. ATF phase, plotted on the secondary y-axis alongside
PRIMARY_COLOR's magnitude trace in the same chart."""

TARGET_COLOR = "#e66767"
"""Red. Target-pattern fill in the Printed Feature panel, alongside
PRIMARY_COLOR's printed-feature line in the same chart."""

REFERENCE_LINE_COLOR = MUTED_INK
"""Threshold and cutoff-frequency marker lines -- non-data annotations, kept
out of the categorical set on purpose so they never compete with it for
identity."""


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """
    Convert a '#rrggbb' hex color to an 'rgba(r,g,b,a)' string.

    Plotly fill colors don't take a hex color plus a separate opacity kwarg
    the way matplotlib's `alpha=` does -- opacity has to be baked into an
    rgba() string up front. Centralized here so PRIMARY_FILL/TARGET_FILL
    below (and any future translucent fill) go through one conversion
    instead of each call site hand-computing r,g,b from the hex itself.
    """
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


PRIMARY_FILL = hex_to_rgba(PRIMARY_COLOR, 0.25)
"""Translucent fill under an aerial-image intensity trace."""

TARGET_FILL = hex_to_rgba(TARGET_COLOR, 0.30)
"""Translucent fill for the target-pattern region in the Printed Feature panel."""


_TEMPLATE_NAME = "fourier_optics_dark"

_axis_template = dict(
    gridcolor=GRIDLINE_COLOR,
    zerolinecolor=AXIS_LINE_COLOR,
    linecolor=AXIS_LINE_COLOR,
    tickfont=dict(color=MUTED_INK, size=10),
    title_font=dict(color=SECONDARY_INK, size=12),
    showline=True,
)

_template = go.layout.Template()
_template.layout = go.Layout(
    paper_bgcolor=CHART_SURFACE,
    plot_bgcolor=CHART_SURFACE,
    font=dict(family=FONT_FAMILY, color=SECONDARY_INK, size=12),
    title_font=dict(color=PRIMARY_INK, size=14),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=SECONDARY_INK, size=10)),
    hoverlabel=dict(bgcolor=CHART_SURFACE, font_color=PRIMARY_INK,
                     bordercolor=AXIS_LINE_COLOR, font_size=12),
    xaxis=_axis_template,
    yaxis=_axis_template,
    margin=dict(l=60, r=50, t=40, b=50),
)
pio.templates[_TEMPLATE_NAME] = _template
pio.templates.default = _TEMPLATE_NAME


def add_reference_vline(fig: go.Figure, x: float, dash: str = "dot") -> None:
    """A vertical reference marker (e.g. pupil cutoff frequency) in the
    shared REFERENCE_LINE_COLOR -- consistent across every panel that needs
    one, rather than each panel picking its own dash/color."""
    fig.add_vline(x=x, line_color=REFERENCE_LINE_COLOR, line_dash=dash, line_width=1)


def add_reference_hline(fig: go.Figure, y: float, dash: str = "dash", **annotation) -> None:
    """A horizontal reference marker (e.g. resist threshold) in the shared
    REFERENCE_LINE_COLOR."""
    fig.add_hline(y=y, line_color=REFERENCE_LINE_COLOR, line_dash=dash, line_width=1, **annotation)
