/**
 * Named constants for the fluid simulation.
 *
 * Every magic number that was previously hard-coded in shader or material
 * logic is gathered here so it can be tuned from a single place and,
 * more importantly, so readers know *why* the value exists.
 */

// ──────────────────────────────────────────────────────────────
//  Numerical stability
// ──────────────────────────────────────────────────────────────

/**
 * Small value added to a divisor to prevent division-by-zero when
 * normalising the vorticity force vector.  Chosen to be well below
 * any visually-relevant magnitude while still being representable
 * in half-precision float.
 */
export const VORTICITY_FORCE_EPSILON = 0.0001;

// ──────────────────────────────────────────────────────────────
//  Splat defaults
// ──────────────────────────────────────────────────────────────

/**
 * Default splat force for the WebGPU path.
 * Negative because the force pushes *away* from the moving object.
 * Expressed in UV-space units (0–1 range).
 */
export const DEFAULT_SPLAT_FORCE_GPU = -0.1;

/**
 * Default splat force for the WebGL path.
 * Uses pixel-space rather than UV-space so the magnitude is larger.
 */
export const DEFAULT_SPLAT_FORCE_WEBGL = -196;

/**
 * Default splat thickness (UV-space radius of influence around the
 * object trail).  1 = full texel width.
 */
export const DEFAULT_SPLAT_THICKNESS = 1;

// ──────────────────────────────────────────────────────────────
//  Pressure solver
// ──────────────────────────────────────────────────────────────

/**
 * Jacobi iteration count for the pressure Poisson solver.
 * Higher = more accurate incompressibility but more GPU passes.
 * 20–40 is typical for real-time; 39 was the original hard-coded value.
 */
export const DEFAULT_PRESSURE_ITERATIONS = 39;

/**
 * Pressure decay multiplier applied each frame (ClearShader).
 * Values < 1 let old pressure bleed off so the surface settles.
 */
export const DEFAULT_PRESSURE_DECAY = 0.317;

// ──────────────────────────────────────────────────────────────
//  Dissipation
// ──────────────────────────────────────────────────────────────

/**
 * How quickly velocity fades per frame (higher = faster decay).
 */
export const DEFAULT_VELOCITY_DISSIPATION = 0.283;

/**
 * How quickly dye / colour fades per frame (WebGPU).
 */
export const DEFAULT_DENSITY_DISSIPATION_GPU = 0.2;

/**
 * How quickly dye / colour fades per frame (WebGL).
 */
export const DEFAULT_DENSITY_DISSIPATION_WEBGL = 0.138;

// ──────────────────────────────────────────────────────────────
//  Vorticity / curl
// ──────────────────────────────────────────────────────────────

/** Default vorticity confinement strength (CurlShader). */
export const DEFAULT_VORTICITY_INFLUENCE = 1;

/** Default swirl intensity multiplier (VorticityShader). */
export const DEFAULT_SWIRL_INTENSITY = 2;

// ──────────────────────────────────────────────────────────────
//  Displacement / visual
// ──────────────────────────────────────────────────────────────

/** Default bump displacement scale for the WebGPU material. */
export const DEFAULT_BUMP_DISPLACEMENT_SCALE = 0.1;

/** Default displacement scale for the WebGL material. */
export const DEFAULT_DISPLACEMENT_SCALE = 0.0078;

/**
 * Default max speed in UV units per second (WebGPU).
 * 1/10 = the fastest a particle can cross the whole surface in 10 s.
 */
export const DEFAULT_MAX_SPEED = 1 / 10;

// ──────────────────────────────────────────────────────────────
//  Alpha channel sentinel
// ──────────────────────────────────────────────────────────────

/**
 * The alpha channel of the velocity render target is repurposed to
 * carry scalar fields (curl, divergence) that are only meaningful
 * for the *current* simulation step.  When writing the final
 * velocity output we reset α to this value to signal "no data".
 */
export const VELOCITY_ALPHA_CLEAR = 0.0;

// ──────────────────────────────────────────────────────────────
//  Velocity clamp (WebGL only)
// ──────────────────────────────────────────────────────────────

/**
 * Hard clamp on velocity magnitude in the WebGL vorticity shader.
 * Prevents runaway feedback when curl forces accumulate.
 */
export const WEBGL_VELOCITY_CLAMP = 1000.0;
