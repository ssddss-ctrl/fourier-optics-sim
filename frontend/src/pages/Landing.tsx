import { useRef } from "react";
import { motion, useScroll, useTransform } from "framer-motion";
import { useNavigate } from "react-router-dom";
import HologramHero from "../components/HologramHero";

/**
 * Landing page, per docs/design/fo-app-ui.pdf (Page 1): a full-viewport
 * opening section (title + hologram hero image + credit), then a
 * scroll-revealed overview section with the button into the simulator. The
 * temporary GET /health wiring check that used to live here (added for the
 * frontend/backend integration prompt) is gone now that this is real
 * content.
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
        style={{
          opacity: section1Opacity,
          scale: section1Scale,
          backgroundImage:
            "radial-gradient(ellipse at center, rgba(57,135,229,0.08) 0%, rgba(13,13,13,0) 60%)," +
            "linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px)," +
            "linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px)",
          backgroundSize: "auto, 48px 48px, 48px 48px",
        }}
        className="relative flex h-screen flex-col items-center justify-center gap-10 bg-page px-4"
      >
        <h1 className="text-center text-4xl font-semibold text-ink sm:text-5xl">
          Fourier Optics Lithography Simulator
        </h1>
        <div className="h-[55vh] w-full max-w-2xl">
          <HologramHero />
        </div>
        <span className="absolute right-6 bottom-4 text-xs text-ink-muted">
          Soham Damle
        </span>

        <motion.div
          className="absolute bottom-8 left-1/2 flex -translate-x-1/2 flex-col items-center gap-1 text-xs text-ink-muted"
          animate={{ y: [0, 6, 0] }}
          transition={{ duration: 1.8, repeat: Infinity, ease: "easeInOut" }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M6 9l6 6 6-6" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span>Scroll Down</span>
        </motion.div>
      </motion.section>

      <motion.section
        ref={section2Ref}
        style={{ opacity: section2Opacity }}
        className="flex min-h-screen flex-col items-center justify-center gap-8 bg-page px-4 text-center"
      >
        <p className="max-w-xl text-lg text-ink-secondary">
          This simulator models the full coherent lithography imaging chain,
          built from first-principles Fourier optics: light diffracts off a
          mask pattern, propagates and passes through a lens that cuts off
          high spatial frequencies by numerical aperture, forms an aerial
          image, and gets thresholded into a printed feature.
        </p>
        <p className="max-w-xl text-lg text-ink-secondary">
          Explore how wavelength, numerical aperture, coherence, and focus
          error shape what actually prints on the wafer.
        </p>
        <div className="flex flex-wrap items-center justify-center gap-4">
          <button
            onClick={() => navigate("/simulator")}
            className="rounded-full border border-primary px-6 py-3 text-sm font-medium text-ink transition-colors hover:bg-primary hover:text-page"
          >
            1D Simulator →
          </button>
          <button
            onClick={() => navigate("/simulator-2d")}
            className="rounded-full border border-primary px-6 py-3 text-sm font-medium text-ink transition-colors hover:bg-primary hover:text-page"
          >
            2D Simulator →
          </button>
        </div>
      </motion.section>
    </div>
  );
}
