import { useEffect, useMemo, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Line } from "@react-three/drei";
import * as THREE from "three";

/**
 * Reads a CSS custom property's live value off the document root, instead
 * of hardcoding a second copy of the hex string here -- WebGL materials
 * can't consume CSS variables directly, so this is the one place that
 * bridges index.css's --color-* tokens into a THREE.Color, keeping a
 * single source of truth for the app's accent color.
 */
function useCssColor(varName: string, fallback: string): string {
  const [color, setColor] = useState(fallback);
  useEffect(() => {
    const value = getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
    if (value) setColor(value);
  }, [varName]);
  return color;
}

// Vertical layout (light source -> mask -> lens -> photoresist) and the
// half-widths used to build the dashed "hourglass" ray bundle between
// them: mask and resist share a width, the lens is narrower, so straight
// lines between consecutive stages naturally converge then diverge again
// -- no crossing/geometry trickery needed to get that look.
const CONE_TIP_Y = 2.1;
const MASK_Y = 0.9;
const LENS_Y = -0.3;
const RESIST_Y = -1.6;
const WIDE_HALF_WIDTH = 0.85; // mask and resist
const NARROW_HALF_WIDTH = 0.65; // lens

function Pipeline({ color }: { color: string }) {
  // One shared, unlit wireframe material -- MeshBasicMaterial ignores
  // scene lighting entirely, which is both the cheapest way to render it
  // (no lights needed at all) and what gives the flat "glowing neon wire"
  // look the design calls for.
  const material = useMemo(
    () => new THREE.MeshBasicMaterial({ color, wireframe: true }),
    [color],
  );

  const rayLines = useMemo<[THREE.Vector3, THREE.Vector3][]>(
    () => [
      // cone tip -> mask edges (diverge)
      [new THREE.Vector3(0, CONE_TIP_Y, 0), new THREE.Vector3(-WIDE_HALF_WIDTH, MASK_Y, 0)],
      [new THREE.Vector3(0, CONE_TIP_Y, 0), new THREE.Vector3(WIDE_HALF_WIDTH, MASK_Y, 0)],
      // mask edges -> lens edges (converge)
      [new THREE.Vector3(-WIDE_HALF_WIDTH, MASK_Y, 0), new THREE.Vector3(-NARROW_HALF_WIDTH, LENS_Y, 0)],
      [new THREE.Vector3(WIDE_HALF_WIDTH, MASK_Y, 0), new THREE.Vector3(NARROW_HALF_WIDTH, LENS_Y, 0)],
      // lens edges -> resist edges (diverge again)
      [new THREE.Vector3(-NARROW_HALF_WIDTH, LENS_Y, 0), new THREE.Vector3(-WIDE_HALF_WIDTH, RESIST_Y, 0)],
      [new THREE.Vector3(NARROW_HALF_WIDTH, LENS_Y, 0), new THREE.Vector3(WIDE_HALF_WIDTH, RESIST_Y, 0)],
    ],
    [],
  );

  return (
    <group>
      {/* Light source: a 3-sided cone (low-poly, reads as a simple
          triangle) pointing down toward the mask. */}
      <mesh position={[0, CONE_TIP_Y + 0.3, 0]} rotation={[Math.PI, 0, 0]} material={material}>
        <coneGeometry args={[0.35, 0.6, 3]} />
      </mesh>

      {/* Mask: thin flat rectangle */}
      <mesh position={[0, MASK_Y, 0]} material={material}>
        <boxGeometry args={[WIDE_HALF_WIDTH * 2, 0.12, 0.5]} />
      </mesh>

      {/* Lens: flattened oval (scaled, low-poly sphere -- few enough
          segments to read as a simple oval outline, not a dense globe) */}
      <mesh position={[0, LENS_Y, 0]} scale={[1.5, 0.65, 1]} material={material}>
        <sphereGeometry args={[0.45, 10, 6]} />
      </mesh>

      {/* Photoresist / target: thin flat rectangle, same width as the mask */}
      <mesh position={[0, RESIST_Y, 0]} material={material}>
        <boxGeometry args={[WIDE_HALF_WIDTH * 2, 0.12, 0.5]} />
      </mesh>

      {rayLines.map(([start, end], i) => (
        <Line
          key={i}
          points={[start, end]}
          color={color}
          dashed
          dashSize={0.1}
          gapSize={0.08}
          lineWidth={1.5}
        />
      ))}
    </group>
  );
}

/**
 * Minimal, deliberately unlabeled wireframe sketch of the coherent
 * imaging chain (light source -> mask -> lens -> photoresist) for the
 * landing page's opening section. No lights, no shadows, no
 * post-processing -- a handful of low-poly meshes and line segments,
 * chosen to stay cheap on an ordinary laptop rather than for physical
 * accuracy (the real chain is what the simulator page actually computes).
 */
export default function OpticalPipelineScene() {
  const primary = useCssColor("--color-primary", "#3987e5");

  return (
    <Canvas
      camera={{ position: [0, 0.3, 6.5], fov: 45 }}
      dpr={[1, 1.5]}
      gl={{ antialias: true, alpha: true }}
    >
      <Pipeline color={primary} />
      <OrbitControls
        autoRotate
        autoRotateSpeed={0.6}
        enableDamping
        dampingFactor={0.08}
        enableZoom={false}
        enablePan={false}
      />
    </Canvas>
  );
}
