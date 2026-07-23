/**
 * frontend/src/lib/api.ts
 * --------------------------
 * Typed client for the FastAPI backend (backend/main.py + backend/schemas.py).
 * Every type here mirrors a Pydantic model there field-for-field -- if the
 * backend's request/response shape changes, this file needs updating to
 * match (there's no shared codegen between the two yet).
 *
 * Backend URL comes from VITE_API_URL (Vite exposes any `VITE_`-prefixed
 * env var via import.meta.env), falling back to localhost:8000 for local
 * dev so this works with zero .env setup today, and just needs
 * VITE_API_URL set (e.g. in the hosting provider's env panel, or a
 * .env.production file) once the backend is deployed somewhere else.
 */

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export type PatternType = "Isolated Line" | "Line-Space Grating";
export type CoherenceMode = "Coherent" | "Incoherent";

// ── Shared parameter groups (mirrors backend/schemas.py's mixins) ───────────

export interface GridParams {
  L?: number;
  N?: number;
}

export interface MaskParams extends GridParams {
  pattern_type?: PatternType;
  feature_width?: number;
  pitch?: number;
  duty_cycle?: number;
}

export interface OpticalParams {
  wavelength_nm?: number;
  NA?: number;
  defocus_waves?: number;
}

// ── Request types ────────────────────────────────────────────────────────────

export type MaskRequest = MaskParams;

export type AtfOtfRequest = GridParams & OpticalParams;

export interface AerialImageRequest extends MaskParams, OpticalParams {
  coherence?: CoherenceMode;
}

export interface PrintedFeatureRequest extends AerialImageRequest {
  threshold?: number;
}

// ── Response types ───────────────────────────────────────────────────────────

export interface MaskResponse {
  x: number[];
  mask: number[];
  target: number[];
}

export interface AerialImageResponse {
  x: number[];
  intensity: number[];
}

export interface AtfOtfResponse {
  fx: number[];
  atf_magnitude: number[];
  atf_phase: (number | null)[]; // null outside the pupil support
  otf_magnitude: number[];
  cutoff_frequency: number;
  contrast_reversal: boolean;
}

export interface PrintedFeatureResponse {
  x: number[];
  target: number[];
  printed: number[];
  epe: (number | null)[]; // null for any target edge with no printed edge to match
  target_edges: number[];
  printed_edges: number[];
  max_abs_epe: number | null;
  mean_abs_epe: number | null;
  target_linewidth: number | null;
  printed_linewidth: number | null;
  linewidth_error: number | null;
  epe_warning: string | null;
  linewidth_warning: string | null;
}

export interface HealthResponse {
  status: string;
  physics_import_check: {
    module: string;
    N: number;
    dx: number;
  };
}

// ── Requests ──────────────────────────────────────────────────────────────────

class ApiError extends Error {
  constructor(
    public path: string,
    public status: number,
    public body: string,
  ) {
    super(`${path} failed: ${status} ${body}`);
  }
}

async function postJson<TReq, TRes>(path: string, body: TReq): Promise<TRes> {
  const res = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new ApiError(path, res.status, await res.text());
  }
  return res.json() as Promise<TRes>;
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_URL}/health`);
  if (!res.ok) {
    throw new ApiError("/health", res.status, await res.text());
  }
  return res.json() as Promise<HealthResponse>;
}

export function getMask(req: MaskRequest = {}): Promise<MaskResponse> {
  return postJson("/api/mask", req);
}

export function getAerialImage(req: AerialImageRequest = {}): Promise<AerialImageResponse> {
  return postJson("/api/aerial-image", req);
}

export function getAtfOtf(req: AtfOtfRequest = {}): Promise<AtfOtfResponse> {
  return postJson("/api/atf-otf", req);
}

export function getPrintedFeature(req: PrintedFeatureRequest = {}): Promise<PrintedFeatureResponse> {
  return postJson("/api/printed-feature", req);
}
