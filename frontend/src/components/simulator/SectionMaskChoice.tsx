/**
 * frontend/src/components/simulator/SectionMaskChoice.tsx
 * -----------------------------------------------------------
 * First Simulator section: "Choose your desired printed feature". Just a
 * pattern-archetype choice (Isolated Line / Line-Space Grating) with a live
 * preview at fixed default dimensions -- actual sizing (line width, pitch,
 * duty cycle) is tuned on the next section, not here, per the "one
 * decision at a time" goal for this redesign.
 */

import { useMemo } from "react";
import Plot from "../Plot";
import { getMask, type MaskResponse, type PatternType } from "../../lib/api";
import { PRIMARY_COLOR, TARGET_COLOR, TARGET_FILL, darkLayout } from "../../lib/plotlyTheme";
import { useApiPanel } from "../../lib/hooks";
import { PLOT_CONFIG } from "./ui";

function PatternCard({
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

export function SectionMaskChoice({
  patternType,
  onPatternTypeChange,
  onNext,
}: {
  patternType: PatternType;
  onPatternTypeChange: (p: PatternType) => void;
  onNext: () => void;
}) {
  const previewParams = useMemo(
    () => ({
      pattern_type: patternType,
      feature_width: 1.0,
      pitch: 2.0,
      duty_cycle: 0.5,
    }),
    [patternType],
  );
  const preview = useApiPanel<typeof previewParams, MaskResponse>(previewParams, getMask);

  return (
    <section className="flex h-screen w-full flex-col items-center justify-center gap-5 px-6">
      <div className="text-center">
        <h1 className="text-xl font-semibold text-ink">Choose your desired printed feature</h1>
        <p className="mt-1 text-sm text-ink-muted">
          Pick a pattern to print — you&apos;ll tune its size on the next step.
        </p>
      </div>

      <div className="grid w-full max-w-md grid-cols-1 gap-3 sm:grid-cols-2">
        <PatternCard
          label="Isolated Line"
          description="A single printed line"
          selected={patternType === "Isolated Line"}
          onClick={() => onPatternTypeChange("Isolated Line")}
        >
          <div className="flex h-12 items-center justify-center rounded bg-page">
            <div className="h-8 w-3 rounded-sm bg-primary" />
          </div>
        </PatternCard>
        <PatternCard
          label="Line-Space Grating"
          description="Repeating lines and spaces"
          selected={patternType === "Line-Space Grating"}
          onClick={() => onPatternTypeChange("Line-Space Grating")}
        >
          <div className="flex h-12 items-center justify-center gap-1.5 rounded bg-page">
            {[0, 1, 2, 3].map((i) => (
              <div key={i} className="h-8 w-2 rounded-sm bg-primary" />
            ))}
          </div>
        </PatternCard>
      </div>

      <div className="w-full max-w-lg rounded-lg border border-axis bg-surface p-3">
        {preview.error ? (
          <div className="rounded border border-target/40 bg-target/10 px-3 py-2 text-sm text-target">
            Failed to load: {preview.error}
          </div>
        ) : (
          preview.data && (
            <Plot
              data={[
                {
                  x: preview.data.x,
                  y: preview.data.target,
                  type: "scatter",
                  mode: "lines",
                  name: "target",
                  line: { color: TARGET_COLOR, width: 1.2, shape: "hvh" },
                  fill: "tozeroy",
                  fillcolor: TARGET_FILL,
                },
                {
                  x: preview.data.x,
                  y: preview.data.mask,
                  type: "scatter",
                  mode: "lines",
                  name: "mask",
                  line: { color: PRIMARY_COLOR, width: 2, shape: "hvh" },
                },
              ]}
              layout={darkLayout({
                height: 170,
                showlegend: false,
                margin: { l: 40, r: 20, t: 10, b: 30 },
                xaxis: { title: { text: "x (µm)" } },
                yaxis: { title: { text: "Transmission" }, range: [-0.1, 1.3] },
              })}
              config={PLOT_CONFIG}
              style={{ width: "100%", height: "170px" }}
              useResizeHandler
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
