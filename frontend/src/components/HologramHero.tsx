import { useMemo } from "react";
import { motion } from "framer-motion";

const PRIMARY = "#3987e5";
const WARM = "#e66767";

const EDGE_MASK =
  "radial-gradient(ellipse at center, black 65%, transparent 100%)";

interface Particle {
  left: number;
  top: number;
  size: number;
  delay: number;
  duration: number;
  color: string;
}

function useParticles(count: number): Particle[] {
  return useMemo(
    () =>
      Array.from({ length: count }, () => ({
        left: 5 + Math.random() * 90,
        top: 5 + Math.random() * 90,
        size: 2 + Math.random() * 2,
        delay: Math.random() * 3,
        duration: 2.5 + Math.random() * 2,
        color: Math.random() < 0.8 ? PRIMARY : WARM,
      })),
    [count],
  );
}

/**
 * Landing-page hero visual: a pre-rendered hologram image (replacing the
 * earlier Three.js wireframe scene) with a pulsing blurred glow layer and a
 * DOM-based twinkling particle field layered on top, so it reads as "alive"
 * without any WebGL/canvas dependency.
 */
export default function HologramHero() {
  const particles = useParticles(28);

  return (
    <div className="relative h-full w-full">
      <motion.img
        src="/hologram.png"
        alt=""
        aria-hidden="true"
        className="absolute inset-0 h-full w-full object-contain"
        style={{
          filter: "blur(24px) saturate(1.4)",
          mixBlendMode: "screen",
          WebkitMaskImage: EDGE_MASK,
          maskImage: EDGE_MASK,
        }}
        animate={{ opacity: [0.35, 0.75, 0.35] }}
        transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
      />

      <motion.img
        src="/hologram.png"
        alt="Optical pipeline hologram"
        className="absolute inset-0 h-full w-full object-contain"
        style={{
          mixBlendMode: "screen",
          WebkitMaskImage: EDGE_MASK,
          maskImage: EDGE_MASK,
        }}
        animate={{ y: [0, -8, 0] }}
        transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
      />

      {particles.map((p, i) => (
        <motion.div
          key={i}
          className="absolute rounded-full"
          style={{
            left: `${p.left}%`,
            top: `${p.top}%`,
            width: p.size,
            height: p.size,
            backgroundColor: p.color,
            boxShadow: `0 0 6px ${p.color}`,
          }}
          animate={{ opacity: [0, 1, 0], y: [0, -14, -24] }}
          transition={{
            duration: p.duration,
            delay: p.delay,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
      ))}
    </div>
  );
}
