/**
 * frontend/src/components/simulator2d/SectionResults2D.tsx
 * ------------------------------------------------------------------
 * Fourth (final) 2D-pager section: mirrors components/simulator/SectionResults.tsx's
 * minimal-primary + Advanced-modal structure exactly, rather than showing
 * every panel inline at once (an inconsistency the original 2D results
 * page had before this restructure). Primary inline view is the single
 * "printed vs. target" outcome (the agreement map, this project's 2D
 * stand-in for 1D's printed-vs-target line chart) plus a plain-language
 * verdict (ObservationsBox2D) and the fidelity/cutoff metrics; the
 * mechanism panels (mask/target, aerial intensity) move behind one
 * "Advanced" button + Modal, the 2D counterpart to 1D's "Advanced: aerial
 * image & ATF/OTF" button. Deliberately has NO OPC/mask-correction panel
 * -- 2D OPC is an explicit, documented scope boundary for this extension
 * (see docs/physics_assumptions.md's "2D Extension Assumptions" section),
 * not an oversight.
 */

import { useMemo, useState } from "react";
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
import { classifyMatch2D } from "../../lib/matchQuality2d";
import type { MatchTier } from "../../lib/matchQuality";
import { Metric } from "../simulator/ui";
import { ScrollNavButton } from "../simulator/ScrollNavButton";
import { Modal } from "../simulator/Modal";

const MASK_COLORSCALE: [number, string][] = [
  [0, CHART_SURFACE],
  [1, PRIMARY_COLOR],
];

const TIER_BORDER: Record<MatchTier, string> = {
  good: "border-phase",
  decent: "border-primary",
  bad: "border-target",
};
const TIER_TEXT: Record<MatchTier, string> = {
  good: "text-phase",
  decent: "text-primary",
  bad: "text-target",
};
const TIER_LABEL: Record<MatchTier, string> = {
  good: "Good match",
  decent: "Decent match",
  bad: "Poor match",
};

function ObservationsBox2D({ result }: { result: Simulate2DResponse | null }) {
  const { tier, message } = classifyMatch2D(result);
  return (
    <div className={`rounded-lg border-l-4 bg-surface px-3 py-2 ${TIER_BORDER[tier]}`}>
      <div className={`text-sm font-semibold ${TIER_TEXT[tier]}`}>{TIER_LABEL[tier]}</div>
      <div className="text-sm text-ink-secondary">{message}</div>
    </div>
  );
}

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
  const [showAdvanced, setShowAdvanced] = useState(false);

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
    <section className="flex h-screen w-full flex-col items-center justify-center gap-3 px-6">
      <ScrollNavButton onClick={onBack} label="back to optical system" />

      <div className="text-center">
        <h1 className="font-display text-2xl font-semibold text-ink">Results</h1>
        <div className="wave-divider mx-auto mt-2 w-16" />
      </div>

      {panel.error && (
        <div className="w-full max-w-md rounded border border-target/40 bg-target/10 px-3 py-2 text-sm text-target">
          Failed to load: {panel.error}
        </div>
      )}

      {panel.data && agreementGrid && (
        <>
          <div className="w-full max-w-md">
            <HeatmapPanel
              title="Printed vs. target agreement"
              x={panel.data.x}
              y={panel.data.y}
              z={agreementGrid}
              colorscale={AGREEMENT_COLORSCALE}
              zmin={AGREEMENT_ZMIN}
              zmax={AGREEMENT_ZMAX}
              height={280}
            />
          </div>

          <div className="flex max-w-md flex-wrap justify-center gap-x-5 gap-y-1 text-xs text-ink-muted">
            {AGREEMENT_LEGEND.map(({ label, color }) => (
              <span key={label} className="flex items-center gap-1.5">
                <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: color }} />
                {label}
              </span>
            ))}
          </div>

          <div className="w-full max-w-md">
            <ObservationsBox2D result={panel.data} />
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

      <button
        type="button"
        onClick={() => setShowAdvanced(true)}
        className="text-xs text-ink-muted underline-offset-2 transition-colors hover:text-ink-secondary hover:underline"
      >
        Advanced: mask & aerial image
      </button>

      <Modal open={showAdvanced} onClose={() => setShowAdvanced(false)} title="Mask & Aerial Image">
        {panel.data && (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
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
          </div>
        )}
      </Modal>

      <p className="max-w-2xl text-center text-xs text-ink-muted">
        Fidelity is reported via intersection-over-union (IoU) between the printed and target
        patterns.
      </p>
    </section>
  );
}
