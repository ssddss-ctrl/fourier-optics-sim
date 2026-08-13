/**
 * frontend/src/components/simulator/ui.tsx
 * ---------------------------------------------
 * Small shared UI building blocks used across the Simulator's section
 * components (extracted from the original single-page Simulator.tsx).
 */

import { useEffect, useState } from "react";

export function ControlGroup({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-3 pb-5 last:pb-0">
      <div>
        <h2 className="font-mono text-xs tracking-[0.12em] text-ink-faint uppercase">{title}</h2>
        <div className="wave-divider mt-1.5" />
      </div>
      {children}
    </div>
  );
}

export function SliderField({
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
      <div className="mb-1.5 flex justify-between text-ink-secondary">
        <span>{label}</span>
        <span className="tabular font-mono text-xs text-primary">
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
        className="wave-slider"
        data-testid={testId}
      />
    </label>
  );
}

export function NumberField({
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
  // Local text buffer, not just `value.toString()` directly on the input:
  // a controlled <input> whose value prop never changes on an invalid/empty
  // keystroke (e.g. backspacing to "") snaps the DOM back to the last valid
  // number every render, which makes backspace look broken. Buffering the
  // raw text lets the field show exactly what was typed -- including
  // transient "" or "-" -- while still committing onChange only once the
  // text parses to a real number, and resyncing to the committed value on
  // blur if the field was left empty/invalid.
  const [text, setText] = useState(value.toString());

  useEffect(() => {
    setText(value.toString());
  }, [value]);

  return (
    <label className="block text-sm">
      <div className="mb-1 text-ink-secondary">{label}</div>
      <input
        type="number"
        value={text}
        min={min}
        max={max}
        step={step}
        onChange={(e) => {
          setText(e.target.value);
          const parsed = parseFloat(e.target.value);
          if (!Number.isNaN(parsed)) onChange(parsed);
        }}
        onBlur={() => setText(value.toString())}
        className="w-full border border-axis bg-page px-2 py-1 font-mono text-ink"
      />
    </label>
  );
}

export function PanelFrame({
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
    <section className="wave-panel" data-testid={testId} data-loading={loading}>
      <div className="wave-divider" />
      <div className="p-4">
        <h2 className="font-display text-base font-semibold text-ink">{title}</h2>
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
      </div>
    </section>
  );
}

export function WarningBanner({ children }: { children: React.ReactNode }) {
  return (
    <div
      className="rounded border border-target/40 bg-target/10 px-3 py-2 text-sm text-target"
      role="alert"
    >
      {children}
    </div>
  );
}

export function Metric({ label, value, delta }: { label: string; value: string; delta?: string }) {
  return (
    <div className="border border-axis bg-page px-3 py-2">
      <div className="font-mono text-xs tracking-wide text-ink-muted uppercase">{label}</div>
      <div className="tabular font-display text-lg font-semibold text-ink">{value}</div>
      {delta && <div className="font-mono text-xs text-ink-secondary">{delta}</div>}
    </div>
  );
}

export const PLOT_STYLE: React.CSSProperties = { width: "100%", height: "320px" };
export const PLOT_CONFIG = { displayModeBar: false, responsive: true };
