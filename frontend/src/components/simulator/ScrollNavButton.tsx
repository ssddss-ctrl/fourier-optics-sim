/**
 * frontend/src/components/simulator/ScrollNavButton.tsx
 * -------------------------------------------------------
 * Small "back to revise" link shown on every Simulator page except the
 * first. Simulator.tsx is a fixed-viewport pager (see that file) with no
 * page-level scrolling at all -- this just calls back up to the shell to
 * move to the previous page, same as the "Next"/CTA buttons on each page.
 */

export function ScrollNavButton({ onClick, label = "back to revise" }: { onClick: () => void; label?: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="inline-flex items-center gap-1 text-xs text-ink-muted transition-colors hover:text-ink-secondary"
    >
      ▲ {label}
    </button>
  );
}
