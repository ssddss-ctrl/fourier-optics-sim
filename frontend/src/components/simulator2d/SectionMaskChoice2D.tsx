/**
 * frontend/src/components/simulator2d/SectionMaskChoice2D.tsx
 * -------------------------------------------------------------------
 * First 2D-pager section: pick a 2D pattern archetype (Contact Hole Array /
 * Chip Block Layout), with a live mask-only preview at fixed default
 * dimensions -- mirrors components/simulator/SectionMaskChoice.tsx's own
 * "one decision at a time" structure exactly, just for the 2D patterns.
 * Uses the lightweight /api/2d/mask endpoint (getMask2D), not the full
 * /api/2d/simulate pipeline, since this page only needs to show the mask
 * shape.
 */

import { useMemo } from "react";
import { HeatmapPanel } from "./HeatmapPanel";
import { getMask2D, type Mask2DResponse, type Pattern2DType } from "../../lib/api";
import { CHART_SURFACE, PRIMARY_COLOR } from "../../lib/plotlyTheme";
import { useApiPanel } from "../../lib/hooks";

const MASK_COLORSCALE: [number, string][] = [
  [0, CHART_SURFACE],
  [1, PRIMARY_COLOR],
];

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

export function SectionMaskChoice2D({
  patternType,
  onPatternTypeChange,
  onNext,
}: {
  patternType: Pattern2DType;
  onPatternTypeChange: (p: Pattern2DType) => void;
  onNext: () => void;
}) {
  const previewParams = useMemo(
    () => ({ pattern_type: patternType, hole_diameter: 0.6, pitch: 1.5 }),
    [patternType],
  );
  const preview = useApiPanel<typeof previewParams, Mask2DResponse>(previewParams, getMask2D);

  return (
    <section className="flex h-screen w-full flex-col items-center justify-center gap-5 px-6">
      <div className="text-center">
        <h1 className="text-xl font-semibold text-ink">Choose your 2D pattern</h1>
        <p className="mt-1 text-sm text-ink-muted">
          Pick a pattern to print — you&apos;ll tune its size on the next step.
        </p>
      </div>

      <div className="grid w-full max-w-md grid-cols-1 gap-3 sm:grid-cols-2">
        <PatternCard2D
          label="Contact Hole Array"
          description="Periodic 2D via/contact array"
          selected={patternType === "Contact Hole Array"}
          onClick={() => onPatternTypeChange("Contact Hole Array")}
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
          onClick={() => onPatternTypeChange("Chip Block Layout")}
        >
          <div className="relative h-12 rounded bg-page">
            <div className="absolute top-2 left-1 h-1.5 w-10 rounded-sm bg-primary" />
            <div className="absolute top-2 left-2 h-8 w-1.5 rounded-sm bg-primary" />
            <div className="absolute top-4 left-6 h-6 w-1.5 rounded-sm bg-primary" />
          </div>
        </PatternCard2D>
      </div>

      <div className="w-full max-w-sm">
        {preview.error ? (
          <div className="rounded border border-target/40 bg-target/10 px-3 py-2 text-sm text-target">
            Failed to load: {preview.error}
          </div>
        ) : (
          preview.data && (
            <HeatmapPanel
              title="Mask preview"
              x={preview.data.x}
              y={preview.data.y}
              z={preview.data.mask}
              colorscale={MASK_COLORSCALE}
              zmin={0}
              zmax={1}
              height={200}
            />
          )
        )}
      </div>

      <button
        type="button"
        onClick={onNext}
        className="rounded-full border border-primary px-6 py-2.5 text-sm font-medium text-ink transition-colors hover:bg-primary hover:text-page"
      >
        Next: Tune your feature →
      </button>
    </section>
  );
}
