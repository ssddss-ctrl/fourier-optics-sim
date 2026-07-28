/**
 * frontend/src/components/simulator/SectionResults.tsx
 * ----------------------------------------------------------
 * Fourth Simulator section: results. Main figure is printed-vs-target;
 * an observations box gives a plain-language good/decent/bad verdict
 * (frontend/src/lib/matchQuality.ts) with a short tuning hint. Two
 * separate "Advanced" buttons each open a Modal (aerial image + ATF/OTF;
 * the mask -> spectrum -> pupil-filtered spectrum -> aerial image
 * pipeline) rather than appending content below -- this page is one of
 * four fixed-viewport pages in Simulator.tsx's pager, so extra figures
 * that don't fit the base layout have to be an overlay, not more scroll.
 */

import { useCallback, useMemo, useState } from "react";
import Plot from "../Plot";
import {
  getAerialImage,
  getAtfOtf,
  getOpc,
  getPrintedFeature,
  getSpectrumPipeline,
  type AerialImageResponse,
  type AtfOtfResponse,
  type CoherenceMode,
  type OpcRequest,
  type OpcResponse,
  type PatternType,
  type PrintedFeatureResponse,
  type SpectrumPipelineResponse,
} from "../../lib/api";
import {
  PHASE_COLOR,
  PRIMARY_COLOR,
  PRIMARY_FILL,
  TARGET_COLOR,
  TARGET_FILL,
  darkLayout,
  horizontalReferenceLine,
  horizontalReferenceLineAnnotation,
  verticalReferenceLine,
} from "../../lib/plotlyTheme";
import { useApiPanel } from "../../lib/hooks";
import { classifyMatch, type MatchTier } from "../../lib/matchQuality";
import { Metric, PLOT_CONFIG, WarningBanner } from "./ui";
import { ScrollNavButton } from "./ScrollNavButton";
import { Modal } from "./Modal";

const L = 10.0;
const N = 1024;

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

/** "-42%" / "no printed edge to compare" style summary of naive -> corrected max|EPE|. */
function formatOpcImprovement(data: OpcResponse): string {
  const before = data.naive_max_abs_epe;
  const after = data.corrected_max_abs_epe;
  if (before == null || after == null) return "—";
  if (before === 0) return after === 0 ? "0%" : "n/a";
  const pct = ((before - after) / before) * 100;
  return `${pct >= 0 ? "-" : "+"}${Math.abs(pct).toFixed(0)}%`;
}

function ObservationsBox({ result }: { result: PrintedFeatureResponse | null }) {
  const { tier, message } = classifyMatch(result);
  return (
    <div
      className={`rounded-lg border-l-4 bg-surface px-3 py-2 ${TIER_BORDER[tier]}`}
      data-testid="observations-box"
    >
      <div className={`text-sm font-semibold ${TIER_TEXT[tier]}`}>{TIER_LABEL[tier]}</div>
      <div className="text-sm text-ink-secondary">{message}</div>
    </div>
  );
}

export function SectionResults({
  onBack,
  patternType,
  featureWidth,
  pitch,
  dutyCycle,
  wavelengthNm,
  NA,
  coherence,
  focusError,
  threshold,
}: {
  onBack: () => void;
  patternType: PatternType;
  featureWidth: number;
  pitch: number;
  dutyCycle: number;
  wavelengthNm: number;
  NA: number;
  coherence: CoherenceMode;
  focusError: number;
  threshold: number;
}) {
  const [showAerialAdvanced, setShowAerialAdvanced] = useState(false);
  const [showPipelineAdvanced, setShowPipelineAdvanced] = useState(false);
  const [showOpcAdvanced, setShowOpcAdvanced] = useState(false);

  const maskParams = useMemo(
    () => ({
      L,
      N,
      pattern_type: patternType,
      feature_width: featureWidth,
      pitch,
      duty_cycle: dutyCycle,
    }),
    [patternType, featureWidth, pitch, dutyCycle],
  );
  const opticalParams = useMemo(
    () => ({ wavelength_nm: wavelengthNm, NA, defocus_waves: focusError }),
    [wavelengthNm, NA, focusError],
  );
  const aerialParams = useMemo(
    () => ({ ...maskParams, ...opticalParams, coherence }),
    [maskParams, opticalParams, coherence],
  );
  const atfOtfParams = useMemo(() => ({ L, N, ...opticalParams }), [opticalParams]);
  const printedParams = useMemo(() => ({ ...aerialParams, threshold }), [aerialParams, threshold]);

  const printedPanel = useApiPanel<typeof printedParams, PrintedFeatureResponse>(
    printedParams,
    getPrintedFeature,
  );
  const aerialPanel = useApiPanel<typeof aerialParams, AerialImageResponse>(aerialParams, getAerialImage);
  const atfOtfPanel = useApiPanel<typeof atfOtfParams, AtfOtfResponse>(atfOtfParams, getAtfOtf);
  const pipelinePanel = useApiPanel<typeof aerialParams, SpectrumPipelineResponse>(
    aerialParams,
    getSpectrumPipeline,
  );

  // OPC is only forward-modeled up to max_iterations times per request, so
  // gate the fetch behind the modal being open rather than firing on every
  // upstream control change like the two panels above -- opcParams stays
  // `null` (a stable reference) until the toggle is opened, and fetchOpc is
  // memoized with an empty dep array so useApiPanel's effect only re-fires
  // when opcParams itself changes identity (toggle open, or printedParams
  // changing while already open), not on every render.
  const opcParams = useMemo<OpcRequest | null>(
    () => (showOpcAdvanced ? printedParams : null),
    [showOpcAdvanced, printedParams],
  );
  const fetchOpc = useCallback(
    (req: OpcRequest | null) => (req ? getOpc(req) : Promise.resolve(null)),
    [],
  );
  const opcPanel = useApiPanel<OpcRequest | null, OpcResponse | null>(opcParams, fetchOpc);

  return (
    <section className="flex h-screen w-full flex-col items-center justify-center gap-3 px-6">
      <ScrollNavButton onClick={onBack} label="back to optical system" />

      <h1 className="text-xl font-semibold text-ink">Results</h1>

      <div className="w-full max-w-lg rounded-lg border border-axis bg-surface p-3">
        <p className="mb-1 text-xs text-ink-muted">Threshold = {threshold.toFixed(2)}</p>
        {printedPanel.error ? (
          <div className="rounded border border-target/40 bg-target/10 px-3 py-2 text-sm text-target">
            Failed to load: {printedPanel.error}
          </div>
        ) : (
          printedPanel.data && (
            <Plot
              data={[
                {
                  x: printedPanel.data.x,
                  y: printedPanel.data.target,
                  type: "scatter",
                  mode: "lines",
                  name: "target",
                  line: { color: TARGET_COLOR, width: 1.2, shape: "hvh" },
                  fill: "tozeroy",
                  fillcolor: TARGET_FILL,
                },
                {
                  x: printedPanel.data.x,
                  y: printedPanel.data.printed,
                  type: "scatter",
                  mode: "lines",
                  name: "printed",
                  line: { color: PRIMARY_COLOR, width: 2, shape: "hvh" },
                },
              ]}
              layout={darkLayout({
                height: 190,
                margin: { l: 40, r: 20, t: 10, b: 30 },
                xaxis: { title: { text: "x (µm)" } },
                yaxis: { title: { text: "Transmission" }, range: [-0.1, 1.3] },
              })}
              config={PLOT_CONFIG}
              style={{ width: "100%", height: "190px" }}
              useResizeHandler
            />
          )
        )}
      </div>

      <div className="w-full max-w-lg">
        <ObservationsBox result={printedPanel.data} />
      </div>

      {printedPanel.data && !printedPanel.data.epe_warning && (
        <div className="grid w-full max-w-lg grid-cols-2 gap-3">
          <Metric label="Max |EPE|" value={`${printedPanel.data.max_abs_epe?.toFixed(4)} µm`} />
          <Metric label="Mean |EPE|" value={`${printedPanel.data.mean_abs_epe?.toFixed(4)} µm`} />
        </div>
      )}

      <div className="flex gap-6">
        <button
          type="button"
          onClick={() => setShowAerialAdvanced(true)}
          className="text-xs text-ink-muted underline-offset-2 transition-colors hover:text-ink-secondary hover:underline"
        >
          Advanced: aerial image & ATF/OTF
        </button>
        <button
          type="button"
          onClick={() => setShowPipelineAdvanced(true)}
          className="text-xs text-ink-muted underline-offset-2 transition-colors hover:text-ink-secondary hover:underline"
        >
          Advanced: spectrum pipeline
        </button>
        <button
          type="button"
          onClick={() => setShowOpcAdvanced(true)}
          className="text-xs text-ink-muted underline-offset-2 transition-colors hover:text-ink-secondary hover:underline"
        >
          Advanced: OPC correction
        </button>
      </div>

      <Modal
        open={showAerialAdvanced}
        onClose={() => setShowAerialAdvanced(false)}
        title="Aerial Image & ATF/OTF"
      >
        <div className="space-y-6">
          <div>
            <p className="mb-1 text-xs text-ink-muted">
              {coherence} illumination — NA={NA.toFixed(2)}, λ={wavelengthNm.toFixed(0)} nm
            </p>
            {aerialPanel.data && (
              <Plot
                data={[
                  {
                    x: aerialPanel.data.x,
                    y: aerialPanel.data.intensity,
                    type: "scatter",
                    mode: "lines",
                    name: "intensity",
                    line: { color: PRIMARY_COLOR, width: 2 },
                    fill: "tozeroy",
                    fillcolor: PRIMARY_FILL,
                  },
                ]}
                layout={darkLayout({
                  height: 260,
                  showlegend: false,
                  xaxis: { title: { text: "x (µm)" } },
                  yaxis: { title: { text: "Intensity (a.u.)" } },
                  shapes: [horizontalReferenceLine(threshold)],
                  annotations: [horizontalReferenceLineAnnotation(threshold, "threshold")],
                })}
                config={PLOT_CONFIG}
                style={{ width: "100%", height: "260px" }}
                useResizeHandler
              />
            )}
          </div>

          <div>
            <p className="mb-1 text-xs text-ink-muted">Focus error = {focusError.toFixed(1)} waves</p>
            {atfOtfPanel.data && (
              <>
                {atfOtfPanel.data.contrast_reversal && (
                  <div className="mb-3">
                    <WarningBanner>
                      ⚠️ Contrast reversal at this defocus — expect spurious resolution in the aerial
                      image.
                    </WarningBanner>
                  </div>
                )}
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <Plot
                    data={[
                      {
                        x: atfOtfPanel.data.fx,
                        y: atfOtfPanel.data.atf_magnitude,
                        type: "scatter",
                        mode: "lines",
                        name: "|H| (ATF magnitude)",
                        line: { color: PRIMARY_COLOR, width: 2 },
                      },
                    ]}
                    layout={darkLayout({
                      title: { text: "ATF magnitude" },
                      height: 240,
                      showlegend: false,
                      xaxis: { title: { text: "fx (cycles/µm)" } },
                      yaxis: { title: { text: "|H|" } },
                      shapes: [
                        verticalReferenceLine(atfOtfPanel.data.cutoff_frequency),
                        verticalReferenceLine(-atfOtfPanel.data.cutoff_frequency),
                      ],
                    })}
                    config={PLOT_CONFIG}
                    style={{ width: "100%", height: "240px" }}
                    useResizeHandler
                  />
                  <Plot
                    data={[
                      {
                        x: atfOtfPanel.data.fx,
                        y: atfOtfPanel.data.otf_magnitude,
                        type: "scatter",
                        mode: "lines",
                        name: "|OTF|",
                        line: { color: PHASE_COLOR, width: 2 },
                      },
                    ]}
                    layout={darkLayout({
                      title: { text: "MTF (OTF magnitude)" },
                      height: 240,
                      showlegend: false,
                      xaxis: { title: { text: "fx (cycles/µm)" } },
                      yaxis: { title: { text: "OTF magnitude" }, range: [-0.05, 1.15] },
                      shapes: [
                        verticalReferenceLine(atfOtfPanel.data.cutoff_frequency),
                        verticalReferenceLine(-atfOtfPanel.data.cutoff_frequency),
                      ],
                    })}
                    config={PLOT_CONFIG}
                    style={{ width: "100%", height: "240px" }}
                    useResizeHandler
                  />
                </div>
              </>
            )}
          </div>
        </div>
      </Modal>

      <Modal
        open={showPipelineAdvanced}
        onClose={() => setShowPipelineAdvanced(false)}
        title="Mask → Spectrum → Pupil → Image"
      >
        {pipelinePanel.data && (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <Plot
              data={[
                {
                  x: pipelinePanel.data.x,
                  y: pipelinePanel.data.mask,
                  type: "scatter",
                  mode: "lines",
                  line: { color: PRIMARY_COLOR, width: 2, shape: "hvh" },
                },
              ]}
              layout={darkLayout({
                title: { text: "① Mask (spatial)" },
                height: 220,
                showlegend: false,
                xaxis: { title: { text: "x (µm)" } },
                yaxis: { title: { text: "Transmission" } },
              })}
              config={PLOT_CONFIG}
              style={{ width: "100%", height: "220px" }}
              useResizeHandler
            />
            <Plot
              data={[
                {
                  x: pipelinePanel.data.fx,
                  y: pipelinePanel.data.mask_spectrum_magnitude,
                  type: "scatter",
                  mode: "lines",
                  line: { color: PRIMARY_COLOR, width: 2 },
                },
              ]}
              layout={darkLayout({
                title: { text: "② Mask spectrum (freq)" },
                height: 220,
                showlegend: false,
                xaxis: { title: { text: "fx (cycles/µm)" } },
                yaxis: { title: { text: "|spectrum|" } },
              })}
              config={PLOT_CONFIG}
              style={{ width: "100%", height: "220px" }}
              useResizeHandler
            />
            <Plot
              data={[
                {
                  x: pipelinePanel.data.fx,
                  y: pipelinePanel.data.filtered_spectrum_magnitude,
                  type: "scatter",
                  mode: "lines",
                  line: { color: PHASE_COLOR, width: 2 },
                },
              ]}
              layout={darkLayout({
                title: { text: "③ Filtered spectrum (freq)" },
                height: 220,
                showlegend: false,
                xaxis: { title: { text: "fx (cycles/µm)" } },
                yaxis: { title: { text: "|spectrum|" } },
              })}
              config={PLOT_CONFIG}
              style={{ width: "100%", height: "220px" }}
              useResizeHandler
            />
            <Plot
              data={[
                {
                  x: pipelinePanel.data.x,
                  y: pipelinePanel.data.aerial_intensity,
                  type: "scatter",
                  mode: "lines",
                  line: { color: PRIMARY_COLOR, width: 2 },
                  fill: "tozeroy",
                  fillcolor: PRIMARY_FILL,
                },
              ]}
              layout={darkLayout({
                title: { text: "④ Aerial image (spatial)" },
                height: 220,
                showlegend: false,
                xaxis: { title: { text: "x (µm)" } },
                yaxis: { title: { text: "Intensity" } },
              })}
              config={PLOT_CONFIG}
              style={{ width: "100%", height: "220px" }}
              useResizeHandler
            />
          </div>
        )}
      </Modal>

      <Modal
        open={showOpcAdvanced}
        onClose={() => setShowOpcAdvanced(false)}
        title="OPC Correction (Week 12)"
      >
        {opcPanel.error ? (
          <div className="rounded border border-target/40 bg-target/10 px-3 py-2 text-sm text-target">
            Failed to load: {opcPanel.error}
          </div>
        ) : (
          opcPanel.data && (
            <div className="space-y-4">
              <p className="text-xs text-ink-muted">
                Edge-bias OPC (Goodman 6.6.1) biases each mask edge opposite its measured print error,
                iteratively, so the printed feature matches the target instead of the uncorrected mask.{" "}
                {opcPanel.data.converged
                  ? `Converged in ${opcPanel.data.n_iterations} iteration${opcPanel.data.n_iterations === 1 ? "" : "s"}.`
                  : `Did not converge within ${opcPanel.data.n_iterations} iterations.`}
              </p>

              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div>
                  <p className="mb-1 text-xs text-ink-muted">Before OPC (naive mask)</p>
                  <Plot
                    data={[
                      {
                        x: opcPanel.data.x,
                        y: opcPanel.data.target,
                        type: "scatter",
                        mode: "lines",
                        name: "target",
                        line: { color: TARGET_COLOR, width: 1.2, shape: "hvh" },
                        fill: "tozeroy",
                        fillcolor: TARGET_FILL,
                      },
                      {
                        x: opcPanel.data.x,
                        y: opcPanel.data.naive_printed,
                        type: "scatter",
                        mode: "lines",
                        name: "printed",
                        line: { color: PRIMARY_COLOR, width: 2, shape: "hvh" },
                      },
                    ]}
                    layout={darkLayout({
                      height: 220,
                      margin: { l: 40, r: 20, t: 10, b: 30 },
                      xaxis: { title: { text: "x (µm)" } },
                      yaxis: { title: { text: "Transmission" }, range: [-0.1, 1.3] },
                    })}
                    config={PLOT_CONFIG}
                    style={{ width: "100%", height: "220px" }}
                    useResizeHandler
                  />
                </div>
                <div>
                  <p className="mb-1 text-xs text-ink-muted">After OPC (corrected mask)</p>
                  <Plot
                    data={[
                      {
                        x: opcPanel.data.x,
                        y: opcPanel.data.target,
                        type: "scatter",
                        mode: "lines",
                        name: "target",
                        line: { color: TARGET_COLOR, width: 1.2, shape: "hvh" },
                        fill: "tozeroy",
                        fillcolor: TARGET_FILL,
                      },
                      {
                        x: opcPanel.data.x,
                        y: opcPanel.data.corrected_printed,
                        type: "scatter",
                        mode: "lines",
                        name: "printed",
                        line: { color: PHASE_COLOR, width: 2, shape: "hvh" },
                      },
                    ]}
                    layout={darkLayout({
                      height: 220,
                      margin: { l: 40, r: 20, t: 10, b: 30 },
                      xaxis: { title: { text: "x (µm)" } },
                      yaxis: { title: { text: "Transmission" }, range: [-0.1, 1.3] },
                    })}
                    config={PLOT_CONFIG}
                    style={{ width: "100%", height: "220px" }}
                    useResizeHandler
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <Metric
                  label="Max |EPE| before"
                  value={
                    opcPanel.data.naive_max_abs_epe != null
                      ? `${opcPanel.data.naive_max_abs_epe.toFixed(4)} µm`
                      : "—"
                  }
                />
                <Metric
                  label="Max |EPE| after"
                  value={
                    opcPanel.data.corrected_max_abs_epe != null
                      ? `${opcPanel.data.corrected_max_abs_epe.toFixed(4)} µm`
                      : "—"
                  }
                />
                <Metric label="Improvement" value={formatOpcImprovement(opcPanel.data)} />
                <Metric label="Iterations" value={`${opcPanel.data.n_iterations} (${opcPanel.data.converged ? "converged" : "capped"})`} />
              </div>
            </div>
          )
        )}
      </Modal>
    </section>
  );
}
