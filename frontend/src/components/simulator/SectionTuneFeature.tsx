/**
 * frontend/src/components/simulator/SectionTuneFeature.tsx
 * --------------------------------------------------------------
 * Second Simulator section: "Tune your feature". Exposes the dimension
 * knobs specific to whichever pattern was chosen in SectionMaskChoice
 * (line width, or pitch + duty cycle), with a live mask/target preview so
 * the user can see the effect of each knob before moving on to the
 * optical system.
 */

import { useMemo } from "react";
import Plot from "../Plot";
import { getMask, type MaskResponse, type PatternType } from "../../lib/api";
import { PRIMARY_COLOR, TARGET_COLOR, TARGET_FILL, darkLayout } from "../../lib/plotlyTheme";
import { SLIDER_DEBOUNCE_MS, useApiPanel, useDebouncedValue } from "../../lib/hooks";
import { NumberField, PLOT_CONFIG, SliderField } from "./ui";
import { ScrollNavButton } from "./ScrollNavButton";

const L = 10.0;
const N = 1024;

export function SectionTuneFeature({
  onBack,
  onNext,
  patternType,
  featureWidth,
  onFeatureWidthChange,
  pitch,
  onPitchChange,
  dutyCycle,
  onDutyCycleChange,
}: {
  onBack: () => void;
  onNext: () => void;
  patternType: PatternType;
  featureWidth: number;
  onFeatureWidthChange: (v: number) => void;
  pitch: number;
  onPitchChange: (v: number) => void;
  dutyCycle: number;
  onDutyCycleChange: (v: number) => void;
}) {
  const debFeatureWidth = useDebouncedValue(featureWidth, SLIDER_DEBOUNCE_MS);
  const debPitch = useDebouncedValue(pitch, SLIDER_DEBOUNCE_MS);
  const debDutyCycle = useDebouncedValue(dutyCycle, SLIDER_DEBOUNCE_MS);

  const maskParams = useMemo(
    () => ({
      L,
      N,
      pattern_type: patternType,
      feature_width: debFeatureWidth,
      pitch: debPitch,
      duty_cycle: debDutyCycle,
    }),
    [patternType, debFeatureWidth, debPitch, debDutyCycle],
  );
  const maskPanel = useApiPanel<typeof maskParams, MaskResponse>(maskParams, getMask);

  return (
    <section className="flex h-screen w-full flex-col items-center justify-center gap-4 px-6">
      <ScrollNavButton onClick={onBack} label="back to feature choice" />

      <div className="text-center">
        <h1 className="font-display text-2xl font-semibold text-ink">Tune your feature</h1>
        <div className="wave-divider mx-auto mt-2 w-16" />
        <p className="mt-3 text-sm text-ink-muted">
          {patternType === "Isolated Line"
            ? "Set the line width — watch the mask update below."
            : "Set the pitch and duty cycle — watch the mask update below."}
        </p>
      </div>

      <div className="w-full max-w-sm space-y-3">
        {patternType === "Isolated Line" ? (
          <NumberField
            label="Line width w (µm)"
            value={featureWidth}
            min={0.05}
            max={L / 2}
            step={0.05}
            onChange={onFeatureWidthChange}
          />
        ) : (
          <>
            <NumberField
              label="Pitch p (µm)"
              value={pitch}
              min={0.1}
              max={L / 2}
              step={0.1}
              onChange={onPitchChange}
            />
            <SliderField
              label="Duty cycle"
              value={dutyCycle}
              min={0.1}
              max={0.9}
              step={0.05}
              onChange={onDutyCycleChange}
            />
          </>
        )}
      </div>

      <div className="wave-panel w-full max-w-lg p-3">
        <p className="mb-1 text-xs text-ink-muted">
          {patternType === "Isolated Line"
            ? `Isolated Line — w = ${featureWidth.toFixed(2)} µm`
            : `Line-Space Grating — pitch = ${pitch.toFixed(2)} µm, DC = ${dutyCycle.toFixed(2)}`}
        </p>
        {maskPanel.error ? (
          <div className="rounded border border-target/40 bg-target/10 px-3 py-2 text-sm text-target">
            Failed to load: {maskPanel.error}
          </div>
        ) : (
          maskPanel.data && (
            <Plot
              data={[
                {
                  x: maskPanel.data.x,
                  y: maskPanel.data.target,
                  type: "scatter",
                  mode: "lines",
                  name: "target",
                  line: { color: TARGET_COLOR, width: 1.2, shape: "hvh" },
                  fill: "tozeroy",
                  fillcolor: TARGET_FILL,
                },
                {
                  x: maskPanel.data.x,
                  y: maskPanel.data.mask,
                  type: "scatter",
                  mode: "lines",
                  name: "mask",
                  line: { color: PRIMARY_COLOR, width: 2, shape: "hvh" },
                },
              ]}
              layout={darkLayout({
                height: 210,
                margin: { l: 40, r: 20, t: 10, b: 30 },
                xaxis: { title: { text: "x (µm)" } },
                yaxis: { title: { text: "Transmission" }, range: [-0.1, 1.3] },
              })}
              config={PLOT_CONFIG}
              style={{ width: "100%", height: "210px" }}
              useResizeHandler
            />
          )
        )}
      </div>

      <button
        type="button"
        onClick={onNext}
        className="rounded-full border border-primary px-6 py-2.5 font-mono text-xs tracking-[0.08em] text-ink uppercase transition-colors hover:bg-primary hover:text-page"
      >
        Next: Optical system →
      </button>
    </section>
  );
}
