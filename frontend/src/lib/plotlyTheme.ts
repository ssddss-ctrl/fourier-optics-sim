/**
 * frontend/src/lib/plotlyTheme.ts
 * -----------------------------------
 * Dark Plotly theme for the Simulator page's live charts, ported from
 * plotting/interactive.py (the Plotly template the retired Streamlit app
 * used). Same surfaces, same categorical series colors -- validated there
 * against the dataviz skill's CVD/contrast checks, not re-picked here. The
 * hex values also match frontend/src/index.css's @theme tokens; kept as
 * plain constants (not CSS var() lookups) because Plotly's own SVG/canvas
 * rendering needs literal color strings in the layout/trace objects it is
 * handed, not values resolved from the page's stylesheet.
 *
 * Python's plotly.io template registration (pio.templates.default) has no
 * equivalent for react-plotly.js -- there is no global "themed Figure"
 * constructor here, so every panel calls darkLayout(...) to get the same
 * chrome applied per-figure instead of once globally.
 */

import type { Annotations, Dash, Layout, Shape } from "plotly.js";

export const CHART_SURFACE = "#1a1a19";
export const PRIMARY_INK = "#ffffff";
export const SECONDARY_INK = "#c3c2b7";
export const MUTED_INK = "#898781";
export const GRIDLINE_COLOR = "#2c2c2a";
export const AXIS_LINE_COLOR = "#383835";
export const FONT_FAMILY = "system-ui, -apple-system, 'Segoe UI', sans-serif";

// ── Categorical series colors (validated -- see plotting/interactive.py) ────
export const PRIMARY_COLOR = "#3987e5";
export const PHASE_COLOR = "#008300";
export const TARGET_COLOR = "#e66767";
export const REFERENCE_LINE_COLOR = MUTED_INK;

export function hexToRgba(hexColor: string, alpha: number): string {
  const h = hexColor.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

export const PRIMARY_FILL = hexToRgba(PRIMARY_COLOR, 0.25);
export const TARGET_FILL = hexToRgba(TARGET_COLOR, 0.3);

const axisTemplate: Partial<Layout["xaxis"]> = {
  gridcolor: GRIDLINE_COLOR,
  zerolinecolor: AXIS_LINE_COLOR,
  linecolor: AXIS_LINE_COLOR,
  tickfont: { color: MUTED_INK, size: 10 },
  showline: true,
};

/** Merge panel-specific overrides on top of the shared dark chrome. */
export function darkLayout(overrides: Partial<Layout> = {}): Partial<Layout> {
  const { xaxis, yaxis, ...rest } = overrides;
  return {
    paper_bgcolor: CHART_SURFACE,
    plot_bgcolor: CHART_SURFACE,
    font: { family: FONT_FAMILY, color: SECONDARY_INK, size: 12 },
    legend: {
      bgcolor: "rgba(0,0,0,0)",
      font: { color: SECONDARY_INK, size: 10 },
      orientation: "h",
      yanchor: "bottom",
      y: 1.02,
      x: 0,
    },
    hoverlabel: {
      bgcolor: CHART_SURFACE,
      font: { color: PRIMARY_INK, size: 12 },
      bordercolor: AXIS_LINE_COLOR,
    },
    margin: { l: 60, r: 50, t: 40, b: 50 },
    xaxis: { ...axisTemplate, ...xaxis },
    yaxis: { ...axisTemplate, ...yaxis },
    ...rest,
  };
}

/** Dotted/dashed vertical reference marker (e.g. pupil cutoff frequency). */
export function verticalReferenceLine(x: number, dash: Dash = "dot"): Partial<Shape> {
  return {
    type: "line",
    xref: "x",
    yref: "paper",
    x0: x,
    x1: x,
    y0: 0,
    y1: 1,
    line: { color: REFERENCE_LINE_COLOR, dash, width: 1 },
  };
}

/** Dashed/dotted horizontal reference marker (e.g. resist threshold). */
export function horizontalReferenceLine(y: number, dash: Dash = "dash"): Partial<Shape> {
  return {
    type: "line",
    xref: "paper",
    yref: "y",
    x0: 0,
    x1: 1,
    y0: y,
    y1: y,
    line: { color: REFERENCE_LINE_COLOR, dash, width: 1 },
  };
}

/** Label for a horizontal reference line, e.g. "threshold". */
export function horizontalReferenceLineAnnotation(y: number, text: string): Partial<Annotations> {
  return {
    xref: "paper",
    yref: "y",
    x: 0,
    y,
    xanchor: "left",
    yanchor: "bottom",
    text,
    showarrow: false,
    font: { color: REFERENCE_LINE_COLOR, size: 10 },
  };
}
