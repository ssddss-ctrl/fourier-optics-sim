/**
 * frontend/src/pages/Simulator.tsx
 * ------------------------------------
 * Simulator page shell: a fixed-viewport pager over four pages (mask
 * choice -> tune feature -> optical system -> results). The outer
 * container is `fixed inset-0 overflow-hidden` with no scrollable content
 * of its own, so there is nothing for a wheel/trackpad/touch gesture to
 * act on -- the only way to move between pages is the Next/Back buttons
 * each section calls back up to `goTo`, which slides the inner track via
 * a plain CSS transform transition (not the scroll-linked useScroll/
 * useTransform pattern Landing.tsx uses -- this is a discrete "jump to
 * page N", animated for continuity, not a scroll-position-driven
 * transition). Deliberately not framer-motion's `animate` prop here: tried
 * that first, and verified directly (via getComputedStyle in the browser)
 * that it got stuck within a couple of percent of its starting position
 * instead of reaching the target -- the same category of issue hit and
 * fixed in Modal.tsx, so the fix here is the same one: plain CSS instead
 * of framer-motion's JS-driven animate.
 *
 * `window.scrollTo(0, 0)` on mount is a defensive fix for a real bug: React
 * Router doesn't reset scroll position on client-side navigation, so
 * arriving here from partway down the two-viewport-tall Landing page used
 * to drop the user mid-page-one instead of at its top. The fixed/inset-0
 * layout makes this moot in practice (nothing here depends on window
 * scroll), but the reset is cheap insurance.
 */

import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import type { CoherenceMode, PatternType } from "../lib/api";
import { SectionMaskChoice } from "../components/simulator/SectionMaskChoice";
import { SectionTuneFeature } from "../components/simulator/SectionTuneFeature";
import { SectionOpticalSystem } from "../components/simulator/SectionOpticalSystem";
import { SectionResults } from "../components/simulator/SectionResults";

const PAGE_COUNT = 4;

export default function Simulator() {
  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  const [pageIndex, setPageIndex] = useState(0);
  const goTo = (i: number) => setPageIndex(Math.max(0, Math.min(PAGE_COUNT - 1, i)));

  // Pattern
  const [patternType, setPatternType] = useState<PatternType>("Isolated Line");
  const [featureWidth, setFeatureWidth] = useState(1.0);
  const [pitch, setPitch] = useState(2.0);
  const [dutyCycle, setDutyCycle] = useState(0.5);

  // Optics. focusError defaults to 0.8 waves (not 0.0) so the out-of-the-box
  // result is a "Decent" match, not a "Good" one -- see
  // frontend/src/lib/matchQuality.ts and backend/schemas.py's
  // OpticalParams.defocus_waves for the same default mirrored server-side.
  // Tested directly against the running backend: 0.0 waves gives ~1-pixel
  // EPE (Good) across every reasonable NA/wavelength/width combination,
  // with no smooth "Decent" middle ground on those knobs alone; a modest
  // defocus gives a clean, recoverable "Decent" result instead (fixable by
  // the user via the Optical System page's Advanced options).
  const [wavelengthNm, setWavelengthNm] = useState(193.0);
  const [NA, setNA] = useState(0.75);
  const [coherence, setCoherence] = useState<CoherenceMode>("Coherent");
  const [focusError, setFocusError] = useState(0.8);
  const [threshold, setThreshold] = useState(0.3);

  return (
    <div className="fixed inset-0 overflow-hidden bg-page text-ink">
      <Link
        to="/"
        className="fixed top-4 left-4 z-10 text-xs text-ink-muted transition-colors hover:text-ink"
      >
        ← Home
      </Link>
      <Link
        to="/simulator-2d"
        className="fixed top-4 left-24 z-10 text-xs text-ink-muted transition-colors hover:text-ink"
      >
        2D Simulator →
      </Link>

      <div className="fixed top-4 right-4 z-10 text-xs text-ink-muted">
        {pageIndex + 1} / {PAGE_COUNT}
      </div>

      <div
        className="transition-transform duration-500 ease-[cubic-bezier(0.65,0,0.35,1)]"
        style={{ transform: `translateY(-${pageIndex * 100}vh)` }}
      >
        <SectionMaskChoice
          patternType={patternType}
          onPatternTypeChange={setPatternType}
          onNext={() => goTo(1)}
        />

        <SectionTuneFeature
          onBack={() => goTo(0)}
          onNext={() => goTo(2)}
          patternType={patternType}
          featureWidth={featureWidth}
          onFeatureWidthChange={setFeatureWidth}
          pitch={pitch}
          onPitchChange={setPitch}
          dutyCycle={dutyCycle}
          onDutyCycleChange={setDutyCycle}
        />

        <SectionOpticalSystem
          onBack={() => goTo(1)}
          onNext={() => goTo(3)}
          patternType={patternType}
          featureWidth={featureWidth}
          pitch={pitch}
          wavelengthNm={wavelengthNm}
          onWavelengthNmChange={setWavelengthNm}
          NA={NA}
          onNAChange={setNA}
          coherence={coherence}
          onCoherenceChange={setCoherence}
          focusError={focusError}
          onFocusErrorChange={setFocusError}
          threshold={threshold}
          onThresholdChange={setThreshold}
        />

        <SectionResults
          onBack={() => goTo(2)}
          patternType={patternType}
          featureWidth={featureWidth}
          pitch={pitch}
          dutyCycle={dutyCycle}
          wavelengthNm={wavelengthNm}
          NA={NA}
          coherence={coherence}
          focusError={focusError}
          threshold={threshold}
        />
      </div>
    </div>
  );
}
