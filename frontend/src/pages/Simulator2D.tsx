/**
 * frontend/src/pages/Simulator2D.tsx
 * ---------------------------------------
 * 2D extension (Week 12 addendum) page shell: a fixed-viewport pager over
 * four pages (pattern choice -> tune feature -> optical system -> results),
 * mirroring pages/Simulator.tsx's own structure exactly (same
 * fixed-inset-0/CSS-transform-slide mechanism, same lifted-state-passed-
 * as-props pattern) -- the guided, one-page-at-a-time flow requested to
 * match the 1D simulator, rather than the single-scroll layout this page
 * originally shipped with.
 *
 * Scope, matching backend/schemas.py's Simulate2DRequest and
 * physics/lens2d.py's/imaging2d.py's documented boundaries: coherent AND
 * incoherent imaging (via imaging2d.py's OTF path), no defocus/aberrations,
 * no 2D OPC, no formal 2D edge-placement-error metric (fidelity is
 * reported via IoU instead -- see physics/imaging2d.py and
 * docs/physics_assumptions.md's "2D Extension Assumptions" section).
 */

import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import type { CoherenceMode, Pattern2DType } from "../lib/api";
import { SectionMaskChoice2D } from "../components/simulator2d/SectionMaskChoice2D";
import { SectionTuneFeature2D } from "../components/simulator2d/SectionTuneFeature2D";
import { SectionOpticalSystem2D } from "../components/simulator2d/SectionOpticalSystem2D";
import { SectionResults2D } from "../components/simulator2d/SectionResults2D";

const PAGE_COUNT = 4;

export default function Simulator2D() {
  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  const [pageIndex, setPageIndex] = useState(0);
  const goTo = (i: number) => setPageIndex(Math.max(0, Math.min(PAGE_COUNT - 1, i)));

  // Pattern
  const [patternType, setPatternType] = useState<Pattern2DType>("Contact Hole Array");
  const [holeDiameter, setHoleDiameter] = useState(0.6);
  const [pitch, setPitch] = useState(1.5);

  // Optics. threshold defaults to 0.5 (not the "sharpest" ~0.2-0.3 for these default
  // pattern/optics params), matching backend/schemas.py's Simulate2DRequest.threshold default --
  // same precedent as Simulator.tsx's own focusError default (0.8, not 0.0): a first-time user
  // should see a "Decent" match out of the box, not a near-perfect one, and discover the fix via
  // this page's own Advanced options panel.
  const [wavelengthNm, setWavelengthNm] = useState(193.0);
  const [NA, setNA] = useState(0.75);
  const [coherence, setCoherence] = useState<CoherenceMode>("Coherent");
  const [threshold, setThreshold] = useState(0.5);

  return (
    <div className="fixed inset-0 overflow-hidden bg-page text-ink">
      <Link
        to="/"
        className="fixed top-4 left-4 z-10 text-xs text-ink-muted transition-colors hover:text-ink"
      >
        ← Home
      </Link>
      <Link
        to="/simulator"
        className="fixed top-4 left-24 z-10 text-xs text-ink-muted transition-colors hover:text-ink"
      >
        1D Simulator →
      </Link>

      <div className="fixed top-4 right-4 z-10 text-xs text-ink-muted">
        {pageIndex + 1} / {PAGE_COUNT}
      </div>

      <div
        className="transition-transform duration-500 ease-[cubic-bezier(0.65,0,0.35,1)]"
        style={{ transform: `translateY(-${pageIndex * 100}vh)` }}
      >
        <SectionMaskChoice2D
          patternType={patternType}
          onPatternTypeChange={setPatternType}
          onNext={() => goTo(1)}
        />

        <SectionTuneFeature2D
          onBack={() => goTo(0)}
          onNext={() => goTo(2)}
          patternType={patternType}
          holeDiameter={holeDiameter}
          onHoleDiameterChange={setHoleDiameter}
          pitch={pitch}
          onPitchChange={setPitch}
        />

        <SectionOpticalSystem2D
          onBack={() => goTo(1)}
          onNext={() => goTo(3)}
          patternType={patternType}
          holeDiameter={holeDiameter}
          pitch={pitch}
          wavelengthNm={wavelengthNm}
          onWavelengthNmChange={setWavelengthNm}
          NA={NA}
          onNAChange={setNA}
          coherence={coherence}
          onCoherenceChange={setCoherence}
          threshold={threshold}
          onThresholdChange={setThreshold}
        />

        <SectionResults2D
          onBack={() => goTo(2)}
          patternType={patternType}
          holeDiameter={holeDiameter}
          pitch={pitch}
          wavelengthNm={wavelengthNm}
          NA={NA}
          coherence={coherence}
          threshold={threshold}
        />
      </div>
    </div>
  );
}
