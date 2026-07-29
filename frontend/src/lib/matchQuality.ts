/**
 * frontend/src/lib/matchQuality.ts
 * -------------------------------------
 * Classifies a /api/printed-feature response into a short, plain-language
 * verdict for the Results section's observations box. Pure function, no
 * new backend fields needed -- built entirely from what
 * PrintedFeatureResponse already returns (mean_abs_epe, linewidth_error,
 * epe_warning, linewidth_warning).
 *
 * Thresholds (µm) are engineering judgment calls for this project's
 * default field width (L=10µm, sub-µm features), not a physics-derived
 * equation -- same convention imaging.ts's thresholding already flags for
 * itself.
 */

import type { PrintedFeatureResponse } from "./api";

export type MatchTier = "good" | "decent" | "bad";

export interface MatchQuality {
  tier: MatchTier;
  message: string;
}

const EPE_GOOD_UM = 0.02;
const EPE_DECENT_UM = 0.06;
const LINEWIDTH_GOOD_UM = 0.05;
const LINEWIDTH_DECENT_UM = 0.15;

export function classifyMatch(result: PrintedFeatureResponse | null): MatchQuality {
  if (!result) {
    return { tier: "bad", message: "No result yet." };
  }

  if (result.epe_warning) {
    return {
      tier: "bad",
      message:
        result.mean_abs_epe == null && result.max_abs_epe == null
          ? "Feature didn't print — try a lower threshold or higher NA."
          : result.epe_warning,
    };
  }

  const meanEpe = result.mean_abs_epe ?? Infinity;
  const linewidthErr = result.linewidth_error != null ? Math.abs(result.linewidth_error) : null;

  const epeTier: MatchTier = meanEpe <= EPE_GOOD_UM ? "good" : meanEpe <= EPE_DECENT_UM ? "decent" : "bad";
  const linewidthTier: MatchTier | null =
    linewidthErr == null
      ? null
      : linewidthErr <= LINEWIDTH_GOOD_UM
        ? "good"
        : linewidthErr <= LINEWIDTH_DECENT_UM
          ? "decent"
          : "bad";

  const tierRank: Record<MatchTier, number> = { good: 0, decent: 1, bad: 2 };
  const tier: MatchTier =
    linewidthTier == null ? epeTier : tierRank[linewidthTier] > tierRank[epeTier] ? linewidthTier : epeTier;

  if (tier === "good") {
    return { tier, message: "Good match — edges and linewidth track the target closely." };
  }
  if (tier === "decent") {
    return {
      tier,
      message: "Decent match — try raising NA or nudging the threshold to tighten edge placement.",
    };
  }
  return {
    tier,
    message: "Poor match — increase NA, shorten wavelength, or retune the threshold.",
  };
}
