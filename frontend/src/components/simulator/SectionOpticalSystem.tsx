/**
 * frontend/src/components/simulator/SectionOpticalSystem.tsx
 * ------------------------------------------------------------------
 * Third Simulator section: optical system parameters (wavelength, NA).
 * Shows the mask's spatial-frequency spectrum with two reference cutoffs
 * overlaid:
 *   f0 = NA/wavelength   -- coherent (pupil-edge) cutoff, lens.cutoff_frequency
 *   f1 = 2*NA/wavelength -- incoherent OTF cutoff (twice the coherent one)
 * compared against the pattern's own fundamental spatial frequency
 * (1/feature_width for an isolated line -- the first sinc-zero location,
 * same B=1/min_feature convention physics/fft_engine.check_sampling
 * already uses; 1/pitch for a grating -- its fundamental harmonic), to
 * give a plain-language "will this resolve" hint.
 *
 * Coherence / focus error / threshold live in a Modal (not an inline
 * accordion) -- this page is one of four fixed-viewport pages in
 * Simulator.tsx's pager (no page-level scrolling at all), so anything that
 * doesn't fit in the base layout has to be an overlay, not appended
 * content.
 */

import { useMemo, useState } from "react";
import Plot from "../Plot";
import {
  getSpectrumPipeline,
  type CoherenceMode,
  type PatternType,
  type SpectrumPipelineResponse,
} from "../../lib/api";
import { PRIMARY_COLOR, darkLayout, verticalReferenceLine } from "../../lib/plotlyTheme";
import { SLIDER_DEBOUNCE_MS, useApiPanel, useDebouncedValue } from "../../lib/hooks";
import { NumberField, PLOT_CONFIG, SliderField } from "./ui";
import { ScrollNavButton } from "./ScrollNavButton";
import { Modal } from "./Modal";

const L = 10.0;
const N = 1024;

type Resolvability = "good" | "marginal" | "bad";

function classifyResolvability(f0: number, f1: number, patternFreq: number): Resolvability {
  if (f0 >= patternFreq) return "good";
  if (f1 >= patternFreq) return "marginal";
  return "bad";
}

const SUGGESTION_TEXT: Record<Resolvability, string> = {
  good: "✓ Resolves comfortably within the cutoff",
  marginal: "△ Marginal — try Incoherent, or raise NA / lower λ",
  bad: "✗ Unresolved — raise NA or lower λ",
};

export function SectionOpticalSystem({
  onBack,
  onNext,
  patternType,
  featureWidth,
  pitch,
  wavelengthNm,
  onWavelengthNmChange,
  NA,
  onNAChange,
  coherence,
  onCoherenceChange,
  focusError,
  onFocusErrorChange,
  threshold,
  onThresholdChange,
}: {
  onBack: () => void;
  onNext: () => void;
  patternType: PatternType;
  featureWidth: number;
  pitch: number;
  wavelengthNm: number;
  onWavelengthNmChange: (v: number) => void;
  NA: number;
  onNAChange: (v: number) => void;
  coherence: CoherenceMode;
  onCoherenceChange: (v: CoherenceMode) => void;
  focusError: number;
  onFocusErrorChange: (v: number) => void;
  threshold: number;
  onThresholdChange: (v: number) => void;
}) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const debWavelengthNm = useDebouncedValue(wavelengthNm, SLIDER_DEBOUNCE_MS);
  const debNA = useDebouncedValue(NA, SLIDER_DEBOUNCE_MS);

  const spectrumParams = useMemo(
    () => ({
      L,
      N,
      pattern_type: patternType,
      feature_width: featureWidth,
      pitch,
      wavelength_nm: debWavelengthNm,
      NA: debNA,
      defocus_waves: focusError,
      coherence,
    }),
    [patternType, featureWidth, pitch, debWavelengthNm, debNA, focusError, coherence],
  );
  const spectrumPanel = useApiPanel<typeof spectrumParams, SpectrumPipelineResponse>(
    spectrumParams,
    getSpectrumPipeline,
  );

  const wavelengthUm = debWavelengthNm / 1000.0;
  const f0 = debNA / wavelengthUm;
  const f1 = 2.0 * f0;
  const patternFreq = patternType === "Isolated Line" ? 1.0 / featureWidth : 1.0 / pitch;
  const resolvability = classifyResolvability(f0, f1, patternFreq);
  const xRange = 2.5 * Math.max(f1, patternFreq, 0.1);

  return (
    <section className="flex h-screen w-full flex-col items-center justify-center gap-3 px-6">
      <ScrollNavButton onClick={onBack} label="back to tuning" />

      <div className="text-center">
        <h1 className="font-display text-2xl font-semibold text-ink">Set your optical system</h1>
        <div className="wave-divider mx-auto mt-2 w-16" />
        <p className="mt-3 text-sm text-ink-muted">
          Wavelength and NA set the resolution limit — see how it compares to your pattern.
        </p>
      </div>

      <div className="grid w-full max-w-lg grid-cols-2 gap-4">
        <NumberField
          label="Wavelength λ (nm)"
          value={wavelengthNm}
          min={10}
          max={800}
          step={1}
          onChange={onWavelengthNmChange}
        />
        <SliderField
          label="Numerical Aperture (NA)"
          value={NA}
          min={0.1}
          max={1.4}
          step={0.05}
          onChange={onNAChange}
        />
      </div>

      <div className="wave-panel w-full max-w-lg p-3">
        <p className="mb-1 text-xs text-ink-muted">
          f0 (coherent) = {f0.toFixed(2)} µm⁻¹, f1 (incoherent) = {f1.toFixed(2)} µm⁻¹
        </p>
        {spectrumPanel.error ? (
          <div className="rounded border border-target/40 bg-target/10 px-3 py-2 text-sm text-target">
            Failed to load: {spectrumPanel.error}
          </div>
        ) : (
          spectrumPanel.data && (
            <Plot
              data={[
                {
                  x: spectrumPanel.data.fx,
                  y: spectrumPanel.data.mask_spectrum_magnitude,
                  type: "scatter",
                  mode: "lines",
                  name: "|mask spectrum|",
                  line: { color: PRIMARY_COLOR, width: 2 },
                },
              ]}
              layout={darkLayout({
                height: 180,
                showlegend: false,
                margin: { l: 40, r: 20, t: 10, b: 30 },
                xaxis: { title: { text: "fx (cycles/µm)" }, range: [-xRange, xRange] },
                yaxis: { title: { text: "|spectrum|" } },
                shapes: [
                  verticalReferenceLine(f0),
                  verticalReferenceLine(-f0),
                  verticalReferenceLine(f1, "dash"),
                  verticalReferenceLine(-f1, "dash"),
                ],
              })}
              config={PLOT_CONFIG}
              style={{ width: "100%", height: "180px" }}
              useResizeHandler
            />
          )
        )}
      </div>
      <p className="text-sm font-medium text-ink-secondary">{SUGGESTION_TEXT[resolvability]}</p>

      <button
        type="button"
        onClick={() => setShowAdvanced(true)}
        className="text-xs text-ink-muted underline-offset-2 transition-colors hover:text-ink-secondary hover:underline"
      >
        Advanced options
      </button>

      <button
        type="button"
        onClick={onNext}
        className="rounded-full bg-phase px-6 py-2.5 font-mono text-xs tracking-[0.08em] text-page uppercase transition-opacity hover:opacity-90"
      >
        View Results →
      </button>

      <Modal open={showAdvanced} onClose={() => setShowAdvanced(false)} title="Advanced options">
        <div className="space-y-5">
          <div className="space-y-2">
            <div className="text-xs font-semibold tracking-wide text-ink-muted uppercase">Coherence</div>
            <div className="flex gap-4 text-sm text-ink-secondary">
              <label className="flex items-center gap-2">
                <input
                  type="radio"
                  name="coherence"
                  checked={coherence === "Coherent"}
                  onChange={() => onCoherenceChange("Coherent")}
                />
                Coherent
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="radio"
                  name="coherence"
                  checked={coherence === "Incoherent"}
                  onChange={() => onCoherenceChange("Incoherent")}
                />
                Incoherent
              </label>
            </div>
          </div>
          <SliderField
            label="Focus error (waves)"
            value={focusError}
            min={-2}
            max={2}
            step={0.1}
            decimals={1}
            onChange={onFocusErrorChange}
          />
          <SliderField
            label="Resist threshold"
            value={threshold}
            min={0.05}
            max={0.95}
            step={0.05}
            onChange={onThresholdChange}
          />
        </div>
      </Modal>
    </section>
  );
}
