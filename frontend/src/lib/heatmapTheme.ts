/**
 * frontend/src/lib/heatmapTheme.ts
 * -------------------------------------
 * Colorscales for the 2D extension's go.Heatmap panels (Simulator2D.tsx) --
 * kept separate from plotlyTheme.ts because a heatmap's colorscale encodes
 * MAGNITUDE/CATEGORY, a fundamentally different job than plotlyTheme.ts's
 * PRIMARY_COLOR/TARGET_COLOR/PHASE_COLOR, which are validated for line/
 * marker identity contrast only. Chosen via this project's "dataviz" skill
 * (validate_palette.js), not picked by eye -- see the derivation below.
 *
 * INTENSITY COLORSCALE -- DERIVATION
 * ------------------------------------------------------------------------
 * The dataviz skill's reference palette (references/palette.md) specifies
 * ONE validated sequential ramp: a single blue hue, 13 lightness steps
 * (100 lightest -> 700 darkest), documented for a LIGHT chart surface
 * ("the lightest step means 'near zero' and is allowed to recede toward
 * the surface"). This app's charts render on a DARK surface (#1a1a19,
 * matching plotlyTheme.ts's CHART_SURFACE), so "recede toward the surface"
 * means the opposite lightness direction here: near-zero should be the
 * DARKEST step (recedes into the dark surface) and the high-intensity
 * endpoint should be the LIGHTEST/brightest step (pops against it). The
 * ramp below is therefore the same 13 validated steps, in REVERSED order
 * relative to the reference doc's light-surface listing -- same hue, same
 * steps, no new colors invented, just re-anchored to this app's actual
 * surface. Step 400 (#3987e5) is, not coincidentally, exactly this app's
 * own PRIMARY_COLOR -- this app's palette was evidently already drawn from
 * the same reference system.
 *
 * AGREEMENT-MAP CATEGORICAL COLORS -- VALIDATED, NOT EYEBALLED
 * ------------------------------------------------------------------------
 * The printed-vs-target "agreement map" needs 3 visually prominent
 * categories (true positive / false positive / false negative) plus a
 * recessive background category (true negative). The obvious semantic
 * choice -- green=correct, red=extra-material-error -- was checked against
 * this project's validator (dark mode, surface #1a1a19) BEFORE assuming it
 * was safe: the dataviz skill's own fixed "status" trio (good/warning/
 * critical) FAILED the CVD-separation check between good and critical
 * (deutan ΔE 4.1, well under the 8 floor -- a real red/green colorblind
 * confusion in exactly the "correct vs. wrong" pairing this chart hinges
 * on). Re-tested candidate pairings against this app's own existing green
 * (PHASE_COLOR, #008300) instead of the skill's separate "status-good"
 * green, and found a validated trio that clears every check (worst
 * all-pairs CVD ΔE 8.6, clears the 8 floor; worst normal-vision ΔE 22.5):
 *   TP (true positive)  -- #008300 (== PHASE_COLOR; different chart, so
 *                           reusing the hue doesn't create identity
 *                           confusion with where PHASE_COLOR already
 *                           appears -- the co-occurrence rule this
 *                           project's validation is scoped to)
 *   FP (false positive)  -- #e66767 (== TARGET_COLOR; same reasoning)
 *   FN (false negative)  -- #9085e9 (new hue -- categorical slot 7,
 *                           "violet" -- not otherwise used in this app)
 *   TN (true negative)   -- #2c2c2a (GRIDLINE_COLOR; deliberately
 *                           recessive/background, not part of the
 *                           validated 3-way trio above, since it isn't a
 *                           "signal" category -- most of the field is
 *                           background outside the target feature, and a
 *                           bright 4th hue there would dominate the chart)
 */

import type { ColorScale } from "plotly.js";

// 13-step blue ramp from the dataviz skill's reference palette
// (references/palette.md), reversed for this app's dark chart surface:
// index 0 = near-zero (darkest, recedes into #1a1a19), last index = max
// intensity (lightest/brightest, pops against the dark surface).
const INTENSITY_STEPS: string[] = [
  "#0d366b", // 700
  "#104281", // 650
  "#184f95", // 600
  "#1c5cab", // 550
  "#256abf", // 500
  "#2a78d6", // 450
  "#3987e5", // 400 -- exactly this app's PRIMARY_COLOR
  "#5598e7", // 350
  "#6da7ec", // 300
  "#86b6ef", // 250
  "#9ec5f4", // 200
  "#b7d3f6", // 150
  "#cde2fb", // 100
];

export const INTENSITY_COLORSCALE: ColorScale = INTENSITY_STEPS.map(
  (color, i): [number, string] => [i / (INTENSITY_STEPS.length - 1), color],
);

// ── Agreement-map categorical colors (validated trio + recessive background) ─

export const AGREEMENT_TP_COLOR = "#008300";
export const AGREEMENT_FP_COLOR = "#e66767";
export const AGREEMENT_FN_COLOR = "#9085e9";
export const AGREEMENT_TN_COLOR = "#2c2c2a";

/** Encodes target/printed pairs into a 4-category integer grid: 0=TN, 1=FN, 2=FP, 3=TP. */
export function agreementCategoryGrid(target: number[][], printed: number[][]): number[][] {
  return target.map((targetRow, i) =>
    targetRow.map((targetOn, j) => {
      const printedOn = printed[i][j] > 0.5;
      const isTargetOn = targetOn > 0.5;
      if (isTargetOn && printedOn) return 3; // TP
      if (!isTargetOn && printedOn) return 2; // FP
      if (isTargetOn && !printedOn) return 1; // FN
      return 0; // TN
    }),
  );
}

/**
 * A discrete (stepped) Plotly colorscale for the 4-category agreement grid
 * above, paired with zmin=-0.5/zmax=3.5 so each integer category occupies
 * an equal-width band (the standard trick for a categorical Plotly
 * heatmap, since go.Heatmap's colorscale is otherwise a continuous
 * interpolation).
 */
export const AGREEMENT_COLORSCALE: ColorScale = [
  [0, AGREEMENT_TN_COLOR],
  [0.25, AGREEMENT_TN_COLOR],
  [0.25, AGREEMENT_FN_COLOR],
  [0.5, AGREEMENT_FN_COLOR],
  [0.5, AGREEMENT_FP_COLOR],
  [0.75, AGREEMENT_FP_COLOR],
  [0.75, AGREEMENT_TP_COLOR],
  [1, AGREEMENT_TP_COLOR],
];

export const AGREEMENT_ZMIN = -0.5;
export const AGREEMENT_ZMAX = 3.5;

export const AGREEMENT_LEGEND: { label: string; color: string }[] = [
  { label: "Correctly printed", color: AGREEMENT_TP_COLOR },
  { label: "Over-exposure (printed, not targeted)", color: AGREEMENT_FP_COLOR },
  { label: "Under-exposure (targeted, not printed)", color: AGREEMENT_FN_COLOR },
  { label: "Background", color: AGREEMENT_TN_COLOR },
];
