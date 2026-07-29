/**
 * frontend/src/pages/Simulator2D.tsx
 * ---------------------------------------
 * 2D extension (Week 12 addendum): a single scrollable showcase page, NOT
 * the fixed-viewport 4-step pager Simulator.tsx uses -- this is a portfolio
 * artifact (2D mask -> circular-pupil aerial image -> printed-feature
 * heatmap), not a guided pedagogical flow, so a normal scrolling page fits
 * better than another forced sequence of steps. Deliberately does not
 * touch Simulator.tsx or any component in components/simulator/ -- this
 * page has its own pattern picker/param controls below, reusing only the
 * generic (not 1D-specific) primitives from components/simulator/ui.tsx.
 *
 * Scope, matching backend/schemas.py's Simulate2DRequest and
 * physics/lens2d.py's own documented boundaries: coherent imaging only, no
 * defocus/aberrations, no 2D OPC, no formal 2D edge-placement-error metric
 * (fidelity is reported via IoU instead -- see physics/imaging2d.py and
 * docs/physics_assumptions.md's "2D Extension Assumptions" section).
 */

import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import type { ColorScale } from "plotly.js";
import Plot from "../components/Plot";
import {
  getSimulate2D,
  type Pattern2DType,
  type Simulate2DResponse,
} from "../lib/api";
import { CHART_SURFACE, PRIMARY_COLOR, darkLayout } from "../lib/plotlyTheme";
import {
  AGREEMENT_COLORSCALE,
  AGREEMENT_LEGEND,
  AGREEMENT_ZMAX,
  AGREEMENT_ZMIN,
  INTENSITY_COLORSCALE,
  agreementCategoryGrid,
} from "../lib/heatmapTheme";
import { SLIDER_DEBOUNCE_MS, useApiPanel, useDebouncedValue } from "../lib/hooks";
import { Metric, NumberField, PLOT_CONFIG, SliderField } from "../components/simulator/ui";

const L = 10.0;
const N = 128;

function PatternCard2D({
  label,
  description,
  selected,
  onClick,
  children,
}: {
  label: string;
  description: string;
  selected: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-lg border p-5 text-left transition-colors ${
        selected ? "border-primary bg-primary/10" : "border-axis bg-surface hover:border-ink-secondary"
      }`}
    >
      {children}
      <div className="mt-4 text-sm font-semibold text-ink">{label}</div>
      <div className="text-xs text-ink-muted">{description}</div>
    </button>
  );
}

function HeatmapPanel({
  title,
  x,
  y,
  z,
  colorscale,
  zmin,
  zmax,
  showscale = false,
}: {
  title: string;
  x: number[];
  y: number[];
  z: number[][];
  colorscale: ColorScale;
  zmin?: number;
  zmax?: number;
  showscale?: boolean;
}) {
  return (
    <div className="rounded-lg border border-axis bg-surface p-3">
      <p className="mb-1 text-xs text-ink-muted">{title}</p>
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
          height: 260,
          margin: { l: 40, r: 10, t: 10, b: 30 },
          xaxis: { title: { text: "x (µm)" }, scaleanchor: "y", constrain: "domain" },
          yaxis: { title: { text: "y (µm)" } },
        })}
        config={PLOT_CONFIG}
        style={{ width: "100%", height: "260px" }}
        useResizeHandler
      />
    </div>
  );
}

export default function Simulator2D() {
  const [patternType, setPatternType] = useState<Pattern2DType>("Contact Hole Array");
  const [holeDiameter, setHoleDiameter] = useState(0.6);
  const [pitch, setPitch] = useState(1.5);
  const [wavelengthNm, setWavelengthNm] = useState(193.0);
  const [NA, setNA] = useState(0.75);
  const [threshold, setThreshold] = useState(0.3);

  const debHoleDiameter = useDebouncedValue(holeDiameter, SLIDER_DEBOUNCE_MS);
  const debPitch = useDebouncedValue(pitch, SLIDER_DEBOUNCE_MS);
  const debWavelengthNm = useDebouncedValue(wavelengthNm, SLIDER_DEBOUNCE_MS);
  const debNA = useDebouncedValue(NA, SLIDER_DEBOUNCE_MS);
  const debThreshold = useDebouncedValue(threshold, SLIDER_DEBOUNCE_MS);

  const params = useMemo(
    () => ({
      pattern_type: patternType,
      hole_diameter: debHoleDiameter,
      pitch: debPitch,
      L,
      N,
      wavelength_nm: debWavelengthNm,
      NA: debNA,
      threshold: debThreshold,
    }),
    [patternType, debHoleDiameter, debPitch, debWavelengthNm, debNA, debThreshold],
  );

  const panel = useApiPanel<typeof params, Simulate2DResponse>(params, getSimulate2D);

  const agreementGrid = useMemo(() => {
    if (!panel.data) return null;
    return agreementCategoryGrid(panel.data.target, panel.data.printed);
  }, [panel.data]);

  return (
    <div className="min-h-screen w-full bg-page px-6 py-10 text-ink">
      <Link
        to="/"
        className="fixed top-4 left-4 z-10 text-xs text-ink-muted transition-colors hover:text-ink"
      >
        ← Home
      </Link>
      <Link
        to="/simulator"
        className="fixed top-4 right-4 z-10 text-xs text-ink-muted transition-colors hover:text-ink"
      >
        1D Simulator →
      </Link>

      <div className="mx-auto max-w-5xl space-y-8 pt-8">
        <div className="text-center">
          <h1 className="text-2xl font-semibold text-ink">2D Mask Patterns</h1>
          <p className="mx-auto mt-2 max-w-2xl text-sm text-ink-muted">
            The 1D simulator's mask, lens, and aerial-image chain, generalized to two spatial
            dimensions -- a genuine circular lens pupil (not a 1D brick-wall cross-section), a real
            2D aerial-image heatmap, and a printed-vs-target agreement map.
          </p>
        </div>

        <div className="mx-auto grid w-full max-w-md grid-cols-1 gap-3 sm:grid-cols-2">
          <PatternCard2D
            label="Contact Hole Array"
            description="Periodic 2D via/contact array"
            selected={patternType === "Contact Hole Array"}
            onClick={() => setPatternType("Contact Hole Array")}
          >
            <div className="grid h-12 grid-cols-3 place-items-center gap-1.5 rounded bg-page">
              {Array.from({ length: 9 }).map((_, i) => (
                <div key={i} className="h-2.5 w-2.5 rounded-full bg-primary" />
              ))}
            </div>
          </PatternCard2D>
          <PatternCard2D
            label="Chip Block Layout"
            description="Simple interconnect-style layout"
            selected={patternType === "Chip Block Layout"}
            onClick={() => setPatternType("Chip Block Layout")}
          >
            <div className="relative h-12 rounded bg-page">
              <div className="absolute top-2 left-1 h-1.5 w-10 rounded-sm bg-primary" />
              <div className="absolute top-2 left-2 h-8 w-1.5 rounded-sm bg-primary" />
              <div className="absolute top-4 left-6 h-6 w-1.5 rounded-sm bg-primary" />
            </div>
          </PatternCard2D>
        </div>

        {patternType === "Contact Hole Array" && (
          <div className="mx-auto grid max-w-lg grid-cols-2 gap-4">
            <SliderField
              label="Hole diameter (µm)"
              value={holeDiameter}
              min={0.2}
              max={2.0}
              step={0.05}
              onChange={setHoleDiameter}
            />
            <SliderField
              label="Pitch (µm)"
              value={pitch}
              min={0.5}
              max={4.0}
              step={0.05}
              onChange={setPitch}
            />
          </div>
        )}

        <div className="mx-auto grid max-w-lg grid-cols-2 gap-4">
          <NumberField
            label="Wavelength λ (nm)"
            value={wavelengthNm}
            min={10}
            max={800}
            step={1}
            onChange={setWavelengthNm}
          />
          <SliderField
            label="Numerical Aperture (NA)"
            value={NA}
            min={0.1}
            max={1.4}
            step={0.05}
            onChange={setNA}
          />
        </div>
        <div className="mx-auto max-w-lg">
          <SliderField
            label="Resist threshold"
            value={threshold}
            min={0.05}
            max={0.95}
            step={0.05}
            onChange={setThreshold}
          />
        </div>

        {panel.error && (
          <div className="mx-auto max-w-lg rounded border border-target/40 bg-target/10 px-3 py-2 text-sm text-target">
            Failed to load: {panel.error}
          </div>
        )}

        {panel.data && agreementGrid && (
          <>
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
              <HeatmapPanel
                title="① Mask / target"
                x={panel.data.x}
                y={panel.data.y}
                z={panel.data.mask}
                colorscale={[
                  [0, CHART_SURFACE],
                  [1, PRIMARY_COLOR],
                ]}
                zmin={0}
                zmax={1}
              />
              <HeatmapPanel
                title="② Aerial image intensity"
                x={panel.data.x}
                y={panel.data.y}
                z={panel.data.aerial_intensity}
                colorscale={INTENSITY_COLORSCALE}
                showscale
              />
              <HeatmapPanel
                title="③ Printed vs. target agreement"
                x={panel.data.x}
                y={panel.data.y}
                z={agreementGrid}
                colorscale={AGREEMENT_COLORSCALE}
                zmin={AGREEMENT_ZMIN}
                zmax={AGREEMENT_ZMAX}
              />
            </div>

            <div className="mx-auto flex max-w-lg flex-wrap justify-center gap-x-5 gap-y-1 text-xs text-ink-muted">
              {AGREEMENT_LEGEND.map(({ label, color }) => (
                <span key={label} className="flex items-center gap-1.5">
                  <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: color }} />
                  {label}
                </span>
              ))}
            </div>

            <div className="mx-auto grid max-w-md grid-cols-2 gap-3">
              <Metric
                label="Fidelity (IoU)"
                value={
                  panel.data.fidelity_score != null
                    ? `${(panel.data.fidelity_score * 100).toFixed(1)}%`
                    : "—"
                }
              />
              <Metric label="Cutoff frequency" value={`${panel.data.cutoff_frequency.toFixed(2)} µm⁻¹`} />
            </div>
          </>
        )}

        <p className="mx-auto max-w-2xl text-center text-xs text-ink-muted">
          Coherent imaging only -- no 2D OPC and no formal 2D edge-placement-error metric. A 2D
          "edge" is a contour, not a point along one axis, and correcting it needs different
          machinery than the 1D edge-bias OPC loop; fidelity here is reported via
          intersection-over-union (IoU) between the printed and target patterns instead.
        </p>
      </div>
    </div>
  );
}
