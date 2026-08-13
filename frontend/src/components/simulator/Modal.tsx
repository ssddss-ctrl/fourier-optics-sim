/**
 * frontend/src/components/simulator/Modal.tsx
 * -----------------------------------------------
 * Shared overlay for "Advanced" content that doesn't fit inside a single
 * page of the Simulator's fixed-viewport pager (see Simulator.tsx -- pages
 * themselves never scroll, so anything beyond the base layout lives here
 * instead). Backdrop click or Escape closes it; the panel itself is the
 * one place internal scrolling is expected/acceptable, unlike the pager
 * pages which must fit in one viewport with no scrolling at all.
 *
 * Rendered via a portal straight onto document.body -- not inline where
 * it's used. Simulator.tsx's pager track is a div animated with a CSS
 * transform (translateY), and any ancestor with a transform becomes the
 * containing block for its position:fixed descendants (a CSS specifics):
 * without the portal, this modal's `fixed inset-0` would size/center
 * itself against that 400vh-tall, scroll-offset track instead of the real
 * viewport, pushing the panel partly off-screen. The portal escapes that
 * ancestor entirely so `fixed` means the actual browser viewport again.
 *
 * No enter/exit fade or scale animation, deliberately -- tried three
 * progressively simpler approaches (framer-motion's AnimatePresence/
 * animate; a `visible`-flag-flipped-after-mount + Tailwind opacity
 * classes; the same flag + inline styles) and each got stuck at its
 * "closed" appearance on this page's Results section specifically, even
 * though the exact same component worked fine on the Optical System
 * page -- verified directly via getComputedStyle each time, ruling out a
 * screenshot/timing artifact. Root cause not fully pinned down (not a
 * stylesheet conflict -- checked -- and reproducible with a raw inline
 * style set directly via the DOM, bypassing React entirely). Given
 * "advanced options actually opens and is usable" matters far more here
 * than a fade-in flourish, this version just mounts/unmounts the portal
 * directly on `open`, full-opacity immediately -- no animation to get
 * stuck.
 */

import { useEffect, type ReactNode } from "react";
import { createPortal } from "react-dom";

export function Modal({
  open,
  onClose,
  title,
  children,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
}) {
  useEffect(() => {
    if (!open) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4"
      onClick={onClose}
    >
      <div
        className="max-h-[85vh] w-full max-w-3xl overflow-y-auto rounded-lg border border-axis bg-surface p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-center justify-between gap-4">
          <h2 className="font-display text-base font-semibold text-ink">{title}</h2>
          <button
            type="button"
            onClick={onClose}
            className="font-mono text-sm text-ink-muted transition-colors hover:text-ink"
            aria-label="Close"
          >
            ✕
          </button>
        </div>
        <div className="wave-divider -mt-2 mb-4" />
        {children}
      </div>
    </div>,
    document.body,
  );
}
