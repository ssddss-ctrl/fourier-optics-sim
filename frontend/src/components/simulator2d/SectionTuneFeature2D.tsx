/**
 * frontend/src/components/simulator2d/SectionTuneFeature2D.tsx
 * --------------------------------------------------------------------
 * Second 2D-pager section: tune the pattern's own dimensions. Contact Hole
 * Array exposes hole-diameter/pitch sliders; Chip Block Layout has no
 * tunable geometry (a fixed illustrative layout -- see
 * physics/masks2d.py::DEFAULT_RECTS), so this page just shows its preview
 * and a Next button, mirroring components/simulator/SectionTuneFeature.tsx's
 * own per-pattern-type branching.
 */

import { useMemo } from "react";
import { HeatmapPanel } from "./HeatmapPanel";
import { getMask2D, type Mask2DResponse, type Pattern2DType } from "../../lib/api";
import { CHART_SURFACE, PRIMARY_COLOR } from "../../lib/plotlyTheme";
import { SLIDER_DEBOUNCE_MS, useApiPanel, useDebouncedValue } from "../../lib/hooks";
import { SliderField } from "../simulator/ui";
import { ScrollNavButton } from "../simulator/ScrollNavButton";

const MASK_COLORSCALE: [number, string][] = [
  [0, CHART_SURFACE],
  [1, PRIMARY_COLOR],
];

export function SectionTuneFeature2D({
  onBack,
  onNext,
  patternType,
  holeDiameter,
  onHoleDiameterChange,
  pitch,
  onPitchChange,
}: {
  onBack: () => void;
  onNext: () => void;
  patternType: Pattern2DType;
  holeDiameter: number;
  onHoleDiameterChange: (v: number) => void;
  pitch: number;
  onPitchChange: (v: number) => void;
}) {
  const debHoleDiameter = useDebouncedValue(holeDiameter, SLIDER_DEBOUNCE_MS);
  const debPitch = useDebouncedValue(pitch, SLIDER_DEBOUNCE_MS);

  const previewParams = useMemo(
    () => ({ pattern_type: patternType, hole_diameter: debHoleDiameter, pitch: debPitch }),
    [patternType, debHoleDiameter, debPitch],
  );
  const preview = useApiPanel<typeof previewParams, Mask2DResponse>(previewParams, getMask2D);

  return (
    <section className="flex h-screen w-full flex-col items-center justify-center gap-5 px-6">
      <ScrollNavButton onClick={onBack} label="back to pattern choice" />

      <div className="text-center">
        <h1 className="text-xl font-semibold text-ink">Tune your feature</h1>
        <p className="mt-1 text-sm text-ink-muted">
          {patternType === "Contact Hole Array"
            ? "Set the hole diameter and pitch — watch the array update below."
            : "This layout is fixed for now — see the preview below."}
        </p>
      </div>

      {patternType === "Contact Hole Array" && (
        <div className="grid w-full max-w-sm grid-cols-2 gap-4">
          <SliderField
            label="Hole diameter (µm)"
            value={holeDiameter}
            min={0.2}
            max={2.0}
            step={0.05}
            onChange={onHoleDiameterChange}
          />
          <SliderField
            label="Pitch (µm)"
            value={pitch}
            min={0.5}
            max={4.0}
            step={0.05}
            onChange={onPitchChange}
          />
        </div>
      )}

      <div className="w-full max-w-sm">
        {preview.error ? (
          <div className="rounded border border-target/40 bg-target/10 px-3 py-2 text-sm text-target">
            Failed to load: {preview.error}
          </div>
        ) : (
          preview.data && (
            <HeatmapPanel
              title={patternType}
              x={preview.data.x}
              y={preview.data.y}
              z={preview.data.mask}
              colorscale={MASK_COLORSCALE}
              zmin={0}
              zmax={1}
              height={220}
            />
          )
        )}
      </div>

      <button
        type="button"
        onClick={onNext}
        className="rounded-full border border-primary px-6 py-2.5 text-sm font-medium text-ink transition-colors hover:bg-primary hover:text-page"
      >
        Next: Optical system →
      </button>
    </section>
  );
}
