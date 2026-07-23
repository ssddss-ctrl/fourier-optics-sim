/**
 * Functional port of the retired Streamlit app (app/main_streamlit_archived.py)
 * onto the React/FastAPI split: same controls, same five panels (Mask/Target,
 * ATF/OTF, Aerial Image, Printed Feature vs. Target, OPC placeholder), now
 * calling backend/main.py's four POST endpoints through frontend/src/lib/api.ts
 * instead of calling physics/ in-process. Grid width/points (L, N) are fixed
 * at the same defaults backend/schemas.py and the archived app both used
 * (10 µm, 1024) rather than exposed as controls -- not in this task's control
 * list, and OPC (Week 12) is the only panel that would ever need mask != target.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import Plot from "../components/Plot";
import {
  getAerialImage,
  getAtfOtf,
  getMask,
  getPrintedFeature,
  type AerialImageResponse,
  type AtfOtfResponse,
  type CoherenceMode,
  type MaskResponse,
  type PatternType,
  type PrintedFeatureResponse,
} from "../lib/api";
import {
  PRIMARY_COLOR,
  PRIMARY_FILL,
  PHASE_COLOR,
  TARGET_COLOR,
  TARGET_FILL,
  darkLayout,
  horizontalReferenceLine,
  horizontalReferenceLineAnnotation,
  verticalReferenceLine,
} from "../lib/plotlyTheme";

// Grid params fixed for this pass -- not in the control list, and identical
// to backend/schemas.py's GridParams defaults / the archived app's widgets.
const L = 10.0;
const N = 1024;

const SLIDER_DEBOUNCE_MS = 250;

function useDebouncedValue<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(id);
  }, [value, delayMs]);
  return debounced;
}

interface AsyncPanel<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

/**
 * Fires `fetcher(req)` whenever `req`'s identity changes (callers memoize
 * req so that only happens once controls settle, not on every keystroke/
 * drag tick) and keeps only the latest in-flight response, in case a slow
 * request resolves after a newer one already landed.
 */
function useApiPanel<TReq, TRes>(req: TReq, fetcher: (req: TReq) => Promise<TRes>): AsyncPanel<TRes> {
  const [state, setState] = useState<AsyncPanel<TRes>>({ data: null, loading: true, error: null });
  const requestIdRef = useRef(0);

  useEffect(() => {
    const requestId = ++requestIdRef.current;
    setState((prev) => ({ ...prev, loading: true, error: null }));
    fetcher(req)
      .then((data) => {
        if (requestIdRef.current === requestId) {
          setState({ data, loading: false, error: null });
        }
      })
      .catch((err: unknown) => {
        if (requestIdRef.current === requestId) {
          setState((prev) => ({
            ...prev,
            loading: false,
            error: err instanceof Error ? err.message : String(err),
          }));
        }
      });
  }, [req, fetcher]);

  return state;
}

// ── Small UI building blocks ─────────────────────────────────────────────────

function ControlGroup({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-3 border-b border-axis pb-5 last:border-0">
      <h2 className="text-xs font-semibold tracking-wide text-ink-muted uppercase">{title}</h2>
      {children}
    </div>
  );
}

function SliderField({
  label,
  value,
  min,
  max,
  step,
  unit,
  decimals = 2,
  onChange,
  testId,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit?: string;
  decimals?: number;
  onChange: (v: number) => void;
  testId?: string;
}) {
  return (
    <label className="block text-sm">
      <div className="mb-1 flex justify-between text-ink-secondary">
        <span>{label}</span>
        <span className="text-ink-muted">
          {value.toFixed(decimals)}
          {unit ?? ""}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-primary"
        data-testid={testId}
      />
    </label>
  );
}

function NumberField({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="block text-sm">
      <div className="mb-1 text-ink-secondary">{label}</div>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => {
          const parsed = parseFloat(e.target.value);
          if (!Number.isNaN(parsed)) onChange(parsed);
        }}
        className="w-full rounded border border-axis bg-page px-2 py-1 text-ink"
      />
    </label>
  );
}

function PanelFrame({
  title,
  caption,
  loading,
  error,
  children,
  testId,
}: {
  title: string;
  caption?: string;
  loading: boolean;
  error: string | null;
  children: React.ReactNode;
  testId: string;
}) {
  return (
    <section
      className="rounded-lg border border-axis bg-surface p-4"
      data-testid={testId}
      data-loading={loading}
    >
      <h2 className="text-base font-semibold text-ink">{title}</h2>
      {caption && <p className="mt-1 text-xs text-ink-muted">{caption}</p>}
      {error ? (
        <div className="mt-3 rounded border border-target/40 bg-target/10 px-3 py-2 text-sm text-target">
          Failed to load: {error}
        </div>
      ) : (
        <div className={loading ? "mt-3 opacity-60 transition-opacity" : "mt-3 transition-opacity"}>
          {children}
        </div>
      )}
    </section>
  );
}

function WarningBanner({ children }: { children: React.ReactNode }) {
  return (
    <div
      className="rounded border border-target/40 bg-target/10 px-3 py-2 text-sm text-target"
      role="alert"
    >
      {children}
    </div>
  );
}

function Metric({ label, value, delta }: { label: string; value: string; delta?: string }) {
  return (
    <div className="rounded border border-axis bg-page px-3 py-2">
      <div className="text-xs text-ink-muted">{label}</div>
      <div className="text-lg font-semibold text-ink">{value}</div>
      {delta && <div className="text-xs text-ink-secondary">{delta}</div>}
    </div>
  );
}

const PLOT_STYLE: React.CSSProperties = { width: "100%", height: "320px" };
const PLOT_CONFIG = { displayModeBar: false, responsive: true };

// ── Page ──────────────────────────────────────────────────────────────────────

export default function Simulator() {
  // Pattern
  const [patternType, setPatternType] = useState<PatternType>("Isolated Line");
  const [featureWidth, setFeatureWidth] = useState(1.0);
  const [pitch, setPitch] = useState(2.0);
  const [dutyCycle, setDutyCycle] = useState(0.5);

  // Optics
  const [wavelengthNm, setWavelengthNm] = useState(193.0);
  const [NA, setNA] = useState(0.75);
  const [coherence, setCoherence] = useState<CoherenceMode>("Coherent");
  const [focusError, setFocusError] = useState(0.0);
  const [threshold, setThreshold] = useState(0.3);

  // Slider/number-driven values are debounced before they reach the API;
  // pattern_type and coherence are discrete toggles, not dragged, so they
  // go straight through.
  const debFeatureWidth = useDebouncedValue(featureWidth, SLIDER_DEBOUNCE_MS);
  const debPitch = useDebouncedValue(pitch, SLIDER_DEBOUNCE_MS);
  const debDutyCycle = useDebouncedValue(dutyCycle, SLIDER_DEBOUNCE_MS);
  const debWavelengthNm = useDebouncedValue(wavelengthNm, SLIDER_DEBOUNCE_MS);
  const debNA = useDebouncedValue(NA, SLIDER_DEBOUNCE_MS);
  const debFocusError = useDebouncedValue(focusError, SLIDER_DEBOUNCE_MS);
  const debThreshold = useDebouncedValue(threshold, SLIDER_DEBOUNCE_MS);

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

  const atfOtfParams = useMemo(
    () => ({ L, N, wavelength_nm: debWavelengthNm, NA: debNA, defocus_waves: debFocusError }),
    [debWavelengthNm, debNA, debFocusError],
  );

  const aerialParams = useMemo(
    () => ({ ...maskParams, ...atfOtfParams, coherence }),
    [maskParams, atfOtfParams, coherence],
  );

  const printedParams = useMemo(
    () => ({ ...aerialParams, threshold: debThreshold }),
    [aerialParams, debThreshold],
  );

  const maskPanel = useApiPanel<typeof maskParams, MaskResponse>(maskParams, getMask);
  const atfOtfPanel = useApiPanel<typeof atfOtfParams, AtfOtfResponse>(atfOtfParams, getAtfOtf);
  const aerialPanel = useApiPanel<typeof aerialParams, AerialImageResponse>(aerialParams, getAerialImage);
  const printedPanel = useApiPanel<typeof printedParams, PrintedFeatureResponse>(
    printedParams,
    getPrintedFeature,
  );

  const plotTitle =
    patternType === "Isolated Line"
      ? `Isolated Line — w = ${featureWidth.toFixed(2)} µm`
      : `Line-Space Grating — pitch = ${pitch.toFixed(2)} µm, DC = ${dutyCycle.toFixed(2)}`;

  return (
    <div className="flex min-h-screen flex-col bg-page text-ink">
      <header className="flex items-center justify-between border-b border-axis px-6 py-4">
        <div>
          <h1 className="text-xl font-semibold">Fourier Optics Lithography Simulator</h1>
          <p className="text-xs text-ink-muted">
            Built week-by-week alongside Goodman&apos;s <em>Introduction to Fourier Optics</em>
          </p>
        </div>
        <Link to="/" className="text-sm text-ink-secondary transition-colors hover:text-ink">
          ← Home
        </Link>
      </header>

      <div className="flex flex-1 flex-col lg:flex-row">
        <aside className="w-full shrink-0 space-y-5 border-b border-axis bg-surface p-6 lg:w-80 lg:border-r lg:border-b-0">
          <p className="text-xs text-ink-muted">Field width L = {L} µm, N = {N} grid points</p>

          <ControlGroup title="Pattern">
            <label className="block text-sm">
              <div className="mb-1 text-ink-secondary">Pattern type</div>
              <select
                value={patternType}
                onChange={(e) => setPatternType(e.target.value as PatternType)}
                className="w-full rounded border border-axis bg-page px-2 py-1 text-ink"
                data-testid="pattern-type-select"
              >
                <option value="Isolated Line">Isolated Line</option>
                <option value="Line-Space Grating">Line-Space Grating</option>
              </select>
            </label>

            {patternType === "Isolated Line" ? (
              <NumberField
                label="Line width w (µm)"
                value={featureWidth}
                min={0.05}
                max={L / 2}
                step={0.05}
                onChange={setFeatureWidth}
              />
            ) : (
              <>
                <NumberField
                  label="Pitch p (µm)"
                  value={pitch}
                  min={0.1}
                  max={L / 2}
                  step={0.1}
                  onChange={setPitch}
                />
                <SliderField
                  label="Duty cycle"
                  value={dutyCycle}
                  min={0.1}
                  max={0.9}
                  step={0.05}
                  onChange={setDutyCycle}
                />
              </>
            )}
          </ControlGroup>

          <ControlGroup title="Illumination">
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
          </ControlGroup>

          <ControlGroup title="Coherence">
            <div className="flex gap-4 text-sm text-ink-secondary">
              <label className="flex items-center gap-2">
                <input
                  type="radio"
                  name="coherence"
                  checked={coherence === "Coherent"}
                  onChange={() => setCoherence("Coherent")}
                  data-testid="coherence-toggle-coherent"
                />
                Coherent
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="radio"
                  name="coherence"
                  checked={coherence === "Incoherent"}
                  onChange={() => setCoherence("Incoherent")}
                  data-testid="coherence-toggle-incoherent"
                />
                Incoherent
              </label>
            </div>
          </ControlGroup>

          <ControlGroup title="Aberrations">
            <SliderField
              label="Focus error (waves)"
              value={focusError}
              min={-2}
              max={2}
              step={0.1}
              decimals={1}
              onChange={setFocusError}
              testId="focus-error-slider"
            />
          </ControlGroup>

          <ControlGroup title="Printed Feature / Thresholding">
            <SliderField
              label="Resist threshold"
              value={threshold}
              min={0.05}
              max={0.95}
              step={0.05}
              onChange={setThreshold}
              testId="threshold-slider"
            />
          </ControlGroup>
        </aside>

        <main className="flex-1 space-y-6 p-6">
          <PanelFrame
            title="① Mask / Target"
            caption={plotTitle}
            loading={maskPanel.loading}
            error={maskPanel.error}
            testId="mask-panel"
          >
            {maskPanel.data && (
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
                  height: 320,
                  xaxis: { title: { text: "x (µm)" } },
                  yaxis: { title: { text: "Transmission" }, range: [-0.1, 1.3] },
                })}
                config={PLOT_CONFIG}
                style={PLOT_STYLE}
                useResizeHandler
              />
            )}
          </PanelFrame>

          <PanelFrame
            title="② ATF / OTF"
            caption={`Focus error = ${focusError.toFixed(1)} waves — ${
              focusError === 0.0
                ? "diffraction-limited (no aberration)"
                : "defocused (physics/aberrations.py)"
            }`}
            loading={atfOtfPanel.loading}
            error={atfOtfPanel.error}
            testId="atf-otf-panel"
          >
            {atfOtfPanel.data && (
              <>
                {atfOtfPanel.data.contrast_reversal && (
                  <div className="mb-3" data-testid="contrast-reversal-warning">
                    <WarningBanner>
                      ⚠️ Contrast reversal: the OTF goes negative at some spatial frequencies at this
                      defocus (Goodman 6.4.3) — expect spurious resolution / inverted contrast in the
                      aerial image below.
                    </WarningBanner>
                  </div>
                )}
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
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
                      {
                        x: atfOtfPanel.data.fx,
                        y: atfOtfPanel.data.atf_phase,
                        type: "scatter",
                        mode: "lines",
                        name: "angle(H) (phase)",
                        yaxis: "y2",
                        line: { color: PHASE_COLOR, width: 1.6, dash: "dash" },
                      },
                    ]}
                    layout={darkLayout({
                      title: { text: "ATF: magnitude & phase" },
                      height: 340,
                      xaxis: { title: { text: "fx (cycles/µm)" } },
                      yaxis: { title: { text: "|H| (ATF magnitude)" } },
                      yaxis2: {
                        title: { text: "ATF phase (radians)" },
                        overlaying: "y",
                        side: "right",
                        gridcolor: "transparent",
                      },
                      shapes: [
                        verticalReferenceLine(atfOtfPanel.data.cutoff_frequency),
                        verticalReferenceLine(-atfOtfPanel.data.cutoff_frequency),
                      ],
                    })}
                    config={PLOT_CONFIG}
                    style={PLOT_STYLE}
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
                        line: { color: PRIMARY_COLOR, width: 2 },
                      },
                    ]}
                    layout={darkLayout({
                      title: { text: "MTF (OTF magnitude)" },
                      height: 340,
                      showlegend: false,
                      xaxis: { title: { text: "fx (cycles/µm)" } },
                      yaxis: {
                        title: { text: "OTF magnitude |H_otf(fx)|" },
                        range: [-0.05, 1.15],
                      },
                      shapes: [
                        verticalReferenceLine(atfOtfPanel.data.cutoff_frequency),
                        verticalReferenceLine(-atfOtfPanel.data.cutoff_frequency),
                      ],
                    })}
                    config={PLOT_CONFIG}
                    style={PLOT_STYLE}
                    useResizeHandler
                  />
                </div>
                <p className="mt-2 text-xs text-ink-muted">
                  Cutoff frequency (dotted lines) is unchanged by defocus — only the phase across the
                  pupil (left) and the resulting contrast at each frequency (right) degrade.
                </p>
              </>
            )}
          </PanelFrame>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_1fr_260px]">
            <PanelFrame
              title="③ Aerial Image"
              caption={`${coherence} illumination — NA=${NA.toFixed(2)}, λ=${wavelengthNm.toFixed(0)} nm${
                focusError > 0.0 ? `, focus error=${focusError.toFixed(1)} waves` : ""
              }`}
              loading={aerialPanel.loading}
              error={aerialPanel.error}
              testId="aerial-image-panel"
            >
              {aerialPanel.data && (
                <>
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
                      height: 320,
                      showlegend: false,
                      xaxis: { title: { text: "x (µm)" } },
                      yaxis: { title: { text: "Intensity (a.u.)" } },
                      shapes: [horizontalReferenceLine(threshold)],
                      annotations: [horizontalReferenceLineAnnotation(threshold, "threshold")],
                    })}
                    config={PLOT_CONFIG}
                    style={PLOT_STYLE}
                    useResizeHandler
                  />
                  <p className="mt-2 text-xs text-ink-muted">
                    Coherent imaging is linear in amplitude (can overshoot 1.0 — Gibbs-like ringing at
                    edges, Eq. 6-20). Incoherent imaging is linear in intensity (Eq. 6-9/6-26), no
                    ringing, but blurs edges differently.
                  </p>
                </>
              )}
            </PanelFrame>

            <PanelFrame
              title="④ Printed Feature vs. Target"
              caption={`Threshold = ${threshold.toFixed(2)}`}
              loading={printedPanel.loading}
              error={printedPanel.error}
              testId="printed-feature-panel"
            >
              {printedPanel.data && (
                <>
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
                      height: 320,
                      xaxis: { title: { text: "x (µm)" } },
                      yaxis: { title: { text: "Transmission" }, range: [-0.1, 1.3] },
                    })}
                    config={PLOT_CONFIG}
                    style={PLOT_STYLE}
                    useResizeHandler
                  />

                  <div className="mt-3 space-y-3" data-testid="printed-feature-metrics">
                    {printedPanel.data.epe_warning ? (
                      <WarningBanner>{printedPanel.data.epe_warning}</WarningBanner>
                    ) : (
                      <>
                        <div className="grid grid-cols-2 gap-3">
                          <Metric
                            label="Max |EPE|"
                            value={`${printedPanel.data.max_abs_epe?.toFixed(4)} µm`}
                          />
                          <Metric
                            label="Mean |EPE|"
                            value={`${printedPanel.data.mean_abs_epe?.toFixed(4)} µm`}
                          />
                        </div>
                        {printedPanel.data.linewidth_error != null ? (
                          <div className="grid grid-cols-2 gap-3">
                            <Metric
                              label="Target linewidth"
                              value={`${printedPanel.data.target_linewidth?.toFixed(4)} µm`}
                            />
                            <Metric
                              label="Printed linewidth"
                              value={`${printedPanel.data.printed_linewidth?.toFixed(4)} µm`}
                              delta={`${printedPanel.data.linewidth_error >= 0 ? "+" : ""}${printedPanel.data.linewidth_error.toFixed(4)} µm`}
                            />
                          </div>
                        ) : printedPanel.data.linewidth_warning ? (
                          patternType === "Isolated Line" ? (
                            <WarningBanner>{printedPanel.data.linewidth_warning}</WarningBanner>
                          ) : (
                            <p className="text-xs text-ink-muted">{printedPanel.data.linewidth_warning}</p>
                          )
                        ) : null}
                      </>
                    )}
                  </div>
                </>
              )}
            </PanelFrame>

            <section
              className="rounded-lg border border-axis bg-surface p-4"
              data-testid="opc-panel"
            >
              <h2 className="text-base font-semibold text-ink">⑤ OPC</h2>
              <div className="mt-3 rounded border border-primary/30 bg-primary/10 px-3 py-2 text-sm text-ink-secondary">
                <p className="font-medium text-ink">🔜 Week 12</p>
                <p className="mt-1">
                  Will iteratively adjust the mask to minimize edge placement error between target and
                  printed feature.
                </p>
              </div>
            </section>
          </div>
        </main>
      </div>
    </div>
  );
}
