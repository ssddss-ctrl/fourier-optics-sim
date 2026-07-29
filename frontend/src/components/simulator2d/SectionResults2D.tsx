/**
 * frontend/src/components/simulator2d/SectionResults2D.tsx
 * ------------------------------------------------------------------
 * Fourth (final) 2D-pager section: the three heatmap panels (mask/target,
 * aerial intensity, printed-vs-target agreement map) plus fidelity/cutoff
 * metrics. Mirrors components/simulator/SectionResults.tsx's role, but
 * deliberately has NO OPC/mask-correction panel -- 2D OPC is an explicit,
 * documented scope boundary for this extension (see
 * docs/physics_assumptions.md's "2D Extension Assumptions" section), not
 * an oversight.
 */

import { useMemo } from "react";
import { HeatmapPanel } from "./HeatmapPanel";
import {
  getSimulate2D,
  type CoherenceMode,
  type Pattern2DType,
  type Simulate2DResponse,
} from "../../lib/api";
import { CHART_SURFACE, PRIMARY_COLOR } from "../../lib/plotlyTheme";
import {
  AGREEMENT_COLORSCALE,
  AGREEMENT_LEGEND,
  AGREEMENT_ZMAX,
  AGREEMENT_ZMIN,
  INTENSITY_COLORSCALE,
  agreementCategoryGrid,
} from "../../lib/heatmapTheme";
import { useApiPanel } from "../../lib/hooks";
import { Metric } from "../simulator/ui";
import { ScrollNavButton } from "../simulator/ScrollNavButton";

const MASK_COLORSCALE: [number, string][] = [
  [0, CHART_SURFACE],
  [1, PRIMARY_COLOR],
];

export function SectionResults2D({
  onBack,
  patternType,
  holeDiameter,
  pitch,
  wavelengthNm,
  NA,
  coherence,
  threshold,
}: {
  onBack: () => void;
  patternType: Pattern2DType;
  holeDiameter: number;
  pitch: number;
  wavelengthNm: number;
  NA: number;
  coherence: CoherenceMode;
  threshold: number;
}) {
  const params = useMemo(
    () => ({
      pattern_type: patternType,
      hole_diameter: holeDiameter,
      pitch,
      wavelength_nm: wavelengthNm,
      NA,
      coherence,
      threshold,
    }),
    [patternType, holeDiameter, pitch, wavelengthNm, NA, coherence, threshold],
  );

  const panel = useApiPanel<typeof params, Simulate2DResponse>(params, getSimulate2D);

  const agreementGrid = useMemo(() => {
    if (!panel.data) return null;
    return agreementCategoryGrid(panel.data.target, panel.data.printed);
  }, [panel.data]);

  return (
    <section className="flex h-screen w-full flex-col items-center justify-center gap-4 px-6">
      <ScrollNavButton onClick={onBack} label="back to optical system" />

      <h1 className="text-xl font-semibold text-ink">Results</h1>

      {panel.error && (
        <div className="mx-auto max-w-lg rounded border border-target/40 bg-target/10 px-3 py-2 text-sm text-target">
          Failed to load: {panel.error}
        </div>
      )}

      {panel.data && agreementGrid && (
        <>
          <div className="grid w-full max-w-5xl grid-cols-1 gap-4 lg:grid-cols-3">
            <HeatmapPanel
              title="① Mask / target"
              x={panel.data.x}
              y={panel.data.y}
              z={panel.data.mask}
              colorscale={MASK_COLORSCALE}
              zmin={0}
              zmax={1}
            />
            <HeatmapPanel
              title={`② Aerial image intensity (${coherence})`}
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

          <div className="flex max-w-lg flex-wrap justify-center gap-x-5 gap-y-1 text-xs text-ink-muted">
            {AGREEMENT_LEGEND.map(({ label, color }) => (
              <span key={label} className="flex items-center gap-1.5">
                <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: color }} />
                {label}
              </span>
            ))}
          </div>

          <div className="grid w-full max-w-md grid-cols-2 gap-3">
            <Metric
              label="Fidelity (IoU)"
              value={panel.data.fidelity_score != null ? `${(panel.data.fidelity_score * 100).toFixed(1)}%` : "—"}
            />
            <Metric label="Cutoff frequency" value={`${panel.data.cutoff_frequency.toFixed(2)} µm⁻¹`} />
          </div>
        </>
      )}

      <p className="max-w-2xl text-center text-xs text-ink-muted">
        No 2D OPC and no formal 2D edge-placement-error metric. A 2D "edge" is a contour, not a
        point along one axis, and correcting it needs different machinery than the 1D edge-bias
        OPC loop; fidelity here is reported via intersection-over-union (IoU) between the printed
        and target patterns instead.
      </p>
    </section>
  );
}
