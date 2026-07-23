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

// Vertical layout (light source -> mask -> lens -> substrate/coating) and
// the half-widths used to build the dashed "hourglass" ray bundle between
// them: mask and substrate share a width, the lens is narrower, so
// straight lines between consecutive stages naturally converge then
// diverge again -- no crossing/geometry trickery needed to get that look.
const CONE_TIP_Y = 1.6;
const MASK_Y = 0.4;
const LENS_Y = -0.8;
const WIDE_HALF_WIDTH = 0.85; // mask and substrate
const NARROW_HALF_WIDTH = 0.65; // lens rim (0.5 profile radius x 1.3 scale, see below)

// Substrate + thin coating stacked on top of it (a nod to the actual
// photoresist-on-wafer stack, not just a single slab). The dashed rays
// terminate at the coating's top face -- the actual "exposure surface" --
// rather than at the old single plate's mid-plane.
const SUBSTRATE_Y = -2.1;
const SUBSTRATE_HEIGHT = 0.1;
const COATING_HEIGHT = 0.035;
const COATING_Y = SUBSTRATE_Y + SUBSTRATE_HEIGHT / 2 + COATING_HEIGHT / 2;
const EXPOSURE_SURFACE_Y = COATING_Y + COATING_HEIGHT / 2;

// Biconvex lens cross-section, as a lathe profile: (radius, height) pairs
// from the top pole (radius 0) out to the rim (height 0) and back in to
// the bottom pole -- revolved around the lathe's own Y axis this makes a
// flattened "flying saucer" lens shape, thin at the rim and bulging in
// the middle, using the SAME thin-along-Y convention the mask/substrate
// boxes already use (no extra rotation needed to match their orientation).
const LENS_RADIAL_SEGMENTS = 8;
const LENS_RIM_RADIUS = 0.5;
const LENS_PROFILE: [number, number][] = [
  [0, 0.22],
  [0.35, 0.09],
  [LENS_RIM_RADIUS, 0],
  [0.35, -0.09],
  [0, -0.22],
];
const LENS_XZ_SCALE: [number, number, number] = [1.3, 1, 1]; // stretches the rim into an oval

// Light-source housing: a short cylinder the dashed rays visibly emerge
// from, instead of converging on a bare point in space.
const SOURCE_HOUSING_HEIGHT = 0.22;
const SOURCE_HOUSING_Y = CONE_TIP_Y + SOURCE_HOUSING_HEIGHT / 2;

/**
 * A shape's silhouette + real structural edges (via EdgesGeometry, which
 * keeps only edges where adjacent face normals differ past a threshold
 * angle) drawn twice: once at full opacity as the outer boundary, once
 * smaller and fainter nested inside it as a depth/volume cue. Two things
 * this buys over the plain `wireframe: true` meshes used elsewhere in
 * this scene:
 *
 * 1. EdgesGeometry drops the diagonal each flat box/lathe face gets from
 *    its own triangulation (adjacent triangles on a flat face are
 *    coplanar, so the angle between them is ~0 and EdgesGeometry treats
 *    that as "not an edge") -- the crisscrossing look the mask/lens had
 *    before was exactly that triangulation showing through, not anything
 *    intentional.
 * 2. The outer/inner opacity split reads as "prominent boundary, fainter
 *    interior" -- NOT done via lineWidth, because WebGL's
 *    LineBasicMaterial.linewidth is ignored on almost every real
 *    browser/GPU combination (a long-standing three.js/ANGLE limitation),
 *    so two different lineWidth values would render pixel-identical here.
 *    Opacity contrast is the actual cross-platform-reliable lever.
 *
 * Used for the "plate" shapes (mask, lens, substrate, coating) -- the
 * small accent shapes (light-source cone/housing, lens rim ring) stay as
 * single plain wireframe meshes, since they're simple enough already that
 * the extra layer wouldn't read as anything but more lines.
 */
function LayeredEdges({
  geometry,
  color,
  position,
  scale,
}: {
  geometry: THREE.BufferGeometry;
  color: string;
  position: [number, number, number];
  scale?: [number, number, number];
}) {
  const edges = useMemo(() => new THREE.EdgesGeometry(geometry, 1), [geometry]);

  return (
    <group position={position} scale={scale}>
      <lineSegments geometry={edges}>
        <lineBasicMaterial color={color} transparent opacity={0.95} />
      </lineSegments>
      <lineSegments geometry={edges} scale={0.8}>
        <lineBasicMaterial color={color} transparent opacity={0.28} />
      </lineSegments>
    </group>
  );
}

function Pipeline({ color, secondaryColor }: { color: string; secondaryColor: string }) {
  // One shared, unlit wireframe material for the small accent shapes
  // (cone, source housing, lens rim ring) -- MeshBasicMaterial ignores
  // scene lighting entirely, which is both the cheapest way to render it
  // (no lights needed at all) and what gives the flat "glowing neon wire"
  // look the design calls for.
  const material = useMemo(
    () => new THREE.MeshBasicMaterial({ color, wireframe: true }),
    [color],
  );

  const maskGeometry = useMemo(() => new THREE.BoxGeometry(WIDE_HALF_WIDTH * 2, 0.12, 0.5), []);
  const substrateGeometry = useMemo(
    () => new THREE.BoxGeometry(WIDE_HALF_WIDTH * 2, SUBSTRATE_HEIGHT, 0.5),
    [],
  );
  const coatingGeometry = useMemo(
    () => new THREE.BoxGeometry(WIDE_HALF_WIDTH * 2 * 0.97, COATING_HEIGHT, 0.5 * 0.97),
    [],
  );
  const lensGeometry = useMemo(() => {
    const points = LENS_PROFILE.map(([radius, y]) => new THREE.Vector2(radius, y));
    return new THREE.LatheGeometry(points, LENS_RADIAL_SEGMENTS);
  }, []);

  const rayLines = useMemo<[THREE.Vector3, THREE.Vector3][]>(
    () => [
      // source housing -> mask edges (diverge)
      [new THREE.Vector3(0, CONE_TIP_Y, 0), new THREE.Vector3(-WIDE_HALF_WIDTH, MASK_Y, 0)],
      [new THREE.Vector3(0, CONE_TIP_Y, 0), new THREE.Vector3(WIDE_HALF_WIDTH, MASK_Y, 0)],
      // mask edges -> lens edges (converge)
      [new THREE.Vector3(-WIDE_HALF_WIDTH, MASK_Y, 0), new THREE.Vector3(-NARROW_HALF_WIDTH, LENS_Y, 0)],
      [new THREE.Vector3(WIDE_HALF_WIDTH, MASK_Y, 0), new THREE.Vector3(NARROW_HALF_WIDTH, LENS_Y, 0)],
      // lens edges -> exposure surface edges (diverge again)
      [new THREE.Vector3(-NARROW_HALF_WIDTH, LENS_Y, 0), new THREE.Vector3(-WIDE_HALF_WIDTH, EXPOSURE_SURFACE_Y, 0)],
      [new THREE.Vector3(NARROW_HALF_WIDTH, LENS_Y, 0), new THREE.Vector3(WIDE_HALF_WIDTH, EXPOSURE_SURFACE_Y, 0)],
    ],
    [],
  );

  return (
    <group>
      {/* Light source: a 3-sided cone (low-poly, reads as a simple
          triangle) plus a small cylindrical housing right at its tip --
          the dashed rays now visibly emerge from a solid-looking object
          instead of a bare point in space. */}
      <mesh position={[0, CONE_TIP_Y + 0.3, 0]} rotation={[Math.PI, 0, 0]} material={material}>
        <coneGeometry args={[0.35, 0.6, 3]} />
      </mesh>
      <mesh position={[0, SOURCE_HOUSING_Y, 0]} material={material}>
        <cylinderGeometry args={[0.11, 0.11, SOURCE_HOUSING_HEIGHT, 8]} />
      </mesh>

      {/* Mask: thin flat rectangle, outer silhouette + faint inner layer */}
      <LayeredEdges geometry={maskGeometry} color={color} position={[0, MASK_Y, 0]} />

      {/* Lens: biconvex lathe profile (thin rim, bulging center), plus a
          thin torus tracing its equator -- the rim is what makes this
          read as "lens" instead of "gem." */}
      <LayeredEdges
        geometry={lensGeometry}
        color={color}
        position={[0, LENS_Y, 0]}
        scale={LENS_XZ_SCALE}
      />
      <mesh
        position={[0, LENS_Y, 0]}
        scale={LENS_XZ_SCALE}
        rotation={[Math.PI / 2, 0, 0]}
        material={material}
      >
        <torusGeometry args={[LENS_RIM_RADIUS, 0.018, 6, 28]} />
      </mesh>

      {/* Substrate (wafer body) + thin coating layer on top, in the
          secondary theme color -- the actual photoresist/coating stack,
          not just a single undifferentiated slab. */}
      <LayeredEdges geometry={substrateGeometry} color={color} position={[0, SUBSTRATE_Y, 0]} />
      <LayeredEdges
        geometry={coatingGeometry}
        color={secondaryColor}
        position={[0, COATING_Y, 0]}
      />

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
 * imaging chain (light source -> mask -> lens -> substrate/coating) for
 * the landing page's opening section. No lights, no shadows, no
 * post-processing -- a handful of low-poly meshes and line segments,
 * chosen to stay cheap on an ordinary laptop rather than for physical
 * accuracy (the real chain is what the simulator page actually computes).
 */
export default function OpticalPipelineScene() {
  const primary = useCssColor("--color-primary", "#3987e5");
  const secondary = useCssColor("--color-phase", "#008300");

  return (
    <Canvas
      camera={{ position: [0, 0.3, 7.5], fov: 45 }}
      dpr={[1, 1.5]}
      gl={{ antialias: true, alpha: true }}
    >
      <Pipeline color={primary} secondaryColor={secondary} />
      <OrbitControls
        autoRotate
        autoRotateSpeed={6}
        enableDamping
        dampingFactor={0.08}
        enableZoom={false}
        enablePan={false}
      />
    </Canvas>
  );
}
