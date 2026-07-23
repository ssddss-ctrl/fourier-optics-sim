import { useRef } from "react";
import { motion, useScroll, useTransform } from "framer-motion";
import { useNavigate } from "react-router-dom";
import OpticalPipelineScene from "../components/OpticalPipelineScene";

/**
 * Landing page, per docs/design/fo-app-ui.pdf (Page 1): a full-viewport
 * opening section (title + rotating optical-pipeline scene + credit),
 * then a scroll-revealed overview section with the button into the
 * simulator. The temporary GET /health wiring check that used to live
 * here (added for the frontend/backend integration prompt) is gone now
 * that this is real content.
 */
export default function Landing() {
  const navigate = useNavigate();

  const section1Ref = useRef<HTMLDivElement>(null);
  const section2Ref = useRef<HTMLDivElement>(null);

  // Section 1's own scroll-out progress: 0 while its top is still at the
  // viewport top, 1 once it's been scrolled fully past (its bottom edge
  // has reached the viewport top).
  const { scrollYProgress: section1Progress } = useScroll({
    target: section1Ref,
    offset: ["start start", "end start"],
  });
  const section1Opacity = useTransform(section1Progress, [0, 1], [1, 0]);
  const section1Scale = useTransform(section1Progress, [0, 1], [1, 0.92]);

  // Section 2's fade-in progress: 0 when its top edge first appears at
  // the viewport bottom (i.e. right as scrolling begins, since section 1
  // is exactly one viewport tall), 1 once its top edge reaches viewport
  // center -- finishes slightly before section 1 fully fades, so the two
  // cross-fade rather than cutting.
  const { scrollYProgress: section2Progress } = useScroll({
    target: section2Ref,
    offset: ["start end", "start center"],
  });
  const section2Opacity = useTransform(section2Progress, [0, 1], [0, 1]);

  return (
    <div className="bg-page">
      <motion.section
        ref={section1Ref}
        style={{ opacity: section1Opacity, scale: section1Scale }}
        className="relative flex h-screen flex-col items-center justify-center gap-6 bg-page px-4"
      >
        <h1 className="text-center text-4xl font-semibold text-ink sm:text-5xl">
          Fourier Optics Lithography Simulator
        </h1>
        <div className="h-[55vh] w-full max-w-2xl">
          <OpticalPipelineScene />
        </div>
        <span className="absolute right-6 bottom-4 text-xs text-ink-muted">
          Soham Damle
        </span>
      </motion.section>

      <motion.section
        ref={section2Ref}
        style={{ opacity: section2Opacity }}
        className="flex min-h-screen flex-col items-center justify-center gap-8 bg-page px-4 text-center"
      >
        <p className="max-w-xl text-lg text-ink-secondary">
          This simulator models the full coherent lithography imaging chain —
          from mask pattern to aerial image to printed feature — built from
          first-principles Fourier optics. Explore how wavelength, numerical
          aperture, coherence, and focus error shape what actually prints on
          the wafer.
        </p>
        <button
          onClick={() => navigate("/simulator")}
          className="rounded-full border border-primary px-6 py-3 text-sm font-medium text-ink transition-colors hover:bg-primary hover:text-page"
        >
          Go to Simulator
        </button>
      </motion.section>
    </div>
  );
}
