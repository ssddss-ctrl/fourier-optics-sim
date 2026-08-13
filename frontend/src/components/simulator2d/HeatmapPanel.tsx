/**
 * frontend/src/components/simulator2d/HeatmapPanel.tsx
 * ----------------------------------------------------------
 * Shared go.Heatmap panel for the 2D pager's mask/tune/optics/results
 * sections -- extracted from the original single-scroll Simulator2D.tsx so
 * every section below can reuse it without duplicating the Plot wiring.
 */

import type { ColorScale } from "plotly.js";
import Plot from "../Plot";
import { darkLayout } from "../../lib/plotlyTheme";
import { PLOT_CONFIG } from "../simulator/ui";

export function HeatmapPanel({
  title,
  x,
  y,
  z,
  colorscale,
  zmin,
  zmax,
  showscale = false,
  height = 260,
}: {
  title: string;
  x: number[];
  y: number[];
  z: number[][];
  colorscale: ColorScale;
  zmin?: number;
  zmax?: number;
  showscale?: boolean;
  height?: number;
}) {
  return (
    <div className="wave-panel p-3">
      <p className="mb-1 font-mono text-xs text-ink-muted">{title}</p>
      <Plot
        data={[
          {
            x,
            y,
            z,
            type: "heatmap",
            colorscale,
            zmin,
            zmax,
            showscale,
            hoverongaps: false,
          },
        ]}
        layout={darkLayout({
          height,
          margin: { l: 40, r: 10, t: 10, b: 30 },
          xaxis: { title: { text: "x (µm)" }, scaleanchor: "y", constrain: "domain" },
          yaxis: { title: { text: "y (µm)" } },
        })}
        config={PLOT_CONFIG}
        style={{ width: "100%", height: `${height}px` }}
        useResizeHandler
      />
    </div>
  );
}
