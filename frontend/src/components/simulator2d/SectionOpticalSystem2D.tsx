/**
 * frontend/src/components/simulator2d/SectionOpticalSystem2D.tsx
 * ------------------------------------------------------------------------
 * Third 2D-pager section: optical system parameters (wavelength, NA), with
 * a live aerial-image heatmap preview and a plain-language resolvability
 * hint -- mirrors components/simulator/SectionOpticalSystem.tsx's role,
 * generalized to 2D. Coherent/Incoherent + resist threshold live in an
 * "Advanced options" Modal, same precedent as the 1D page; there is no
 * focus-error control here since this 2D extension has no aberrations.
 */

import { useMemo, useState } from "react";
import { HeatmapPanel } from "./HeatmapPanel";
import {
  getSimulate2D,
  type CoherenceMode,
  type Pattern2DType,
  type Simulate2DResponse,
} from "../../lib/api";
import { INTENSITY_COLORSCALE } from "../../lib/heatmapTheme";
import { SLIDER_DEBOUNCE_MS, useApiPanel, useDebouncedValue } from "../../lib/hooks";
import { NumberField, SliderField } from "../simulator/ui";
import { ScrollNavButton } from "../simulator/ScrollNavButton";
import { Modal } from "../simulator/Modal";

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

export function SectionOpticalSystem2D({
  onBack,
  onNext,
  patternType,
  holeDiameter,
  pitch,
  wavelengthNm,
  onWavelengthNmChange,
  NA,
  onNAChange,
  coherence,
  onCoherenceChange,
  threshold,
  onThresholdChange,
}: {
  onBack: () => void;
  onNext: () => void;
  patternType: Pattern2DType;
  holeDiameter: number;
  pitch: number;
  wavelengthNm: number;
  onWavelengthNmChange: (v: number) => void;
  NA: number;
  onNAChange: (v: number) => void;
  coherence: CoherenceMode;
  onCoherenceChange: (v: CoherenceMode) => void;
  threshold: number;
  onThresholdChange: (v: number) => void;
}) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const debWavelengthNm = useDebouncedValue(wavelengthNm, SLIDER_DEBOUNCE_MS);
  const debNA = useDebouncedValue(NA, SLIDER_DEBOUNCE_MS);

  const previewParams = useMemo(
    () => ({
      pattern_type: patternType,
      hole_diameter: holeDiameter,
      pitch,
      wavelength_nm: debWavelengthNm,
      NA: debNA,
      coherence,
      threshold,
    }),
    [patternType, holeDiameter, pitch, debWavelengthNm, debNA, coherence, threshold],
  );
  const preview = useApiPanel<typeof previewParams, Simulate2DResponse>(previewParams, getSimulate2D);

  const wavelengthUm = debWavelengthNm / 1000.0;
  const f0 = debNA / wavelengthUm;
  const f1 = 2.0 * f0;
  const patternFreq = patternType === "Contact Hole Array" ? 1.0 / pitch : null;
  const resolvability = patternFreq != null ? classifyResolvability(f0, f1, patternFreq) : null;

  return (
    <section className="flex h-screen w-full flex-col items-center justify-center gap-3 px-6">
      <ScrollNavButton onClick={onBack} label="back to tuning" />

      <div className="text-center">
        <h1 className="font-display text-2xl font-semibold text-ink">Set your optical system</h1>
        <div className="wave-divider mx-auto mt-2 w-16" />
        <p className="mt-3 text-sm text-ink-muted">
          Wavelength and NA set the resolution limit — see how it shapes the aerial image.
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

      <div className="w-full max-w-sm">
        {preview.error ? (
          <div className="rounded border border-target/40 bg-target/10 px-3 py-2 text-sm text-target">
            Failed to load: {preview.error}
          </div>
        ) : (
          preview.data && (
            <HeatmapPanel
              title="Aerial image intensity (preview)"
              x={preview.data.x}
              y={preview.data.y}
              z={preview.data.aerial_intensity}
              colorscale={INTENSITY_COLORSCALE}
              height={200}
            />
          )
        )}
      </div>

      {resolvability && <p className="text-sm font-medium text-ink-secondary">{SUGGESTION_TEXT[resolvability]}</p>}

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
                  name="coherence-2d"
                  checked={coherence === "Coherent"}
                  onChange={() => onCoherenceChange("Coherent")}
                />
                Coherent
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="radio"
                  name="coherence-2d"
                  checked={coherence === "Incoherent"}
                  onChange={() => onCoherenceChange("Incoherent")}
                />
                Incoherent
              </label>
            </div>
          </div>
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
