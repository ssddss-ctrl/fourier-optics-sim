/**
 * frontend/src/lib/matchQuality2d.ts
 * -----------------------------------------
 * Classifies a /api/2d/simulate response into a short, plain-language
 * verdict for the 2D Results section's observations box -- the 2D
 * counterpart to matchQuality.ts's classifyMatch, mirroring its structure
 * and tier names (good/decent/bad) exactly so SectionResults2D.tsx's
 * verdict box can reuse the same visual convention SectionResults.tsx
 * already established.
 *
 * IoU tier thresholds (0.9 / 0.6) are engineering judgment calls for this
 * project's default 2D field/pattern parameters, not a physics-derived
 * result -- same convention matchQuality.ts's own EPE/linewidth thresholds
 * already flag for themselves.
 */

import type { Simulate2DResponse } from "./api";
import type { MatchQuality, MatchTier } from "./matchQuality";

const IOU_GOOD = 0.9;
const IOU_DECENT = 0.6;

export function classifyMatch2D(result: Simulate2DResponse | null): MatchQuality {
  if (!result) {
    return { tier: "bad", message: "No result yet." };
  }

  if (result.fidelity_warning || result.fidelity_score == null) {
    return {
      tier: "bad",
      message: result.fidelity_warning ?? "Feature didn't print — try a lower threshold or higher NA.",
    };
  }

  const score = result.fidelity_score;
  const tier: MatchTier = score >= IOU_GOOD ? "good" : score >= IOU_DECENT ? "decent" : "bad";

  if (tier === "good") {
    return { tier, message: "Good match — the printed pattern closely tracks the target." };
  }
  if (tier === "decent") {
    return {
      tier,
      message: "Decent match — try adjusting NA, wavelength, or the resist threshold to tighten the fit.",
    };
  }
  return {
    tier,
    message: "Poor match — increase NA, shorten wavelength, or retune the threshold.",
  };
}
