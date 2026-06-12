/*
 * Named constants for the WebGPU (TSL) fluid simulation.
 *
 * These used to live as bare literals scattered through the shader bodies and the
 * material class. They are collected here so the provenance of every "magic number"
 * is documented in one place. Tweak the simulation defaults here.
 *
 * Original WebGL shader code (c) 2017 Pavel Dobryakov, adapted for ThreeJs (c) 2025 Pablo Bandinopla. MIT.
 */

// --------------------------------------------------------------------------------------------
// Simulation defaults
// These are the starting values for the tunable uniforms exposed by the material / debug panel.
// --------------------------------------------------------------------------------------------

/** Max absolute fluid speed, expressed in UV units. */
export const DEFAULT_MAX_SPEED = 1 / 10;

/**
 * Default splat "force" for the WebGPU backend, expressed as a fraction of the max speed.
 * Negative so that motion pushes the fluid in the intuitive direction.
 */
export const DEFAULT_SPLAT_FORCE = -0.1;

/** Radius of a splat, in UV units (scaled per tracked object by its `ratio`). */
export const DEFAULT_SPLAT_THICKNESS = 1;

/** How strongly curl is written into the velocity field by the curl pass. */
export const DEFAULT_VORTICITY_INFLUENCE = 1;

/** Swirl strength applied by the vorticity-confinement pass. */
export const DEFAULT_SWIRL_INTENSITY = 2;

/** Per-iteration pressure decay used by the clear pass (a.k.a. "pressure" in the panel). */
export const DEFAULT_PRESSURE_DECAY = 0.317;

/** Velocity field dissipation per second applied during advection. */
export const DEFAULT_VELOCITY_DISSIPATION = 0.283;

/** Dye/colour dissipation per second applied during advection. */
export const DEFAULT_DENSITY_DISSIPATION = 0.2;

/** Generic advection dissipation default (overridden per-pass by the material). */
export const DEFAULT_ADVECTION_DISSIPATION = 0.1;

/**
 * Number of Jacobi iterations used to solve for pressure. More iterations => more accurate
 * incompressibility, at linear cost.
 */
export const DEFAULT_PRESSURE_ITERATIONS = 39;

/** Vertical displacement applied to the mesh surface from the dye height, in local units. */
export const DEFAULT_BUMP_DISPLACEMENT_SCALE = 0.1;

// --------------------------------------------------------------------------------------------
// Numerical scheme constants
// Standard finite-difference / solver coefficients. Named so the math is self-describing.
// --------------------------------------------------------------------------------------------

/** Half-texel offset used to sample at the centre of a compute pixel. */
export const PIXEL_CENTER_OFFSET = 0.5;

/** Central-difference factor: derivatives use (right - left) * 0.5 over a two-texel span. */
export const CENTRAL_DIFFERENCE_SCALE = 0.5;

/** Jacobi relaxation factor for the pressure solve: (L + R + T + B - divergence) / 4. */
export const PRESSURE_JACOBI_SCALE = 0.25;

/** Epsilon added before normalising the vorticity force to avoid a divide-by-zero. */
export const DIVISION_EPSILON = 0.0001;

/**
 * Value written to the alpha "wildcard" channel when a pass has finished using it.
 * The alpha channel is scratch space passed between passes and is not persisted.
 */
export const WILDCARD_CHANNEL_RESET = 0.0;

// --------------------------------------------------------------------------------------------
// Surface shading constants
// --------------------------------------------------------------------------------------------

/** Rec. 601 luma weights, used to derive a height field from the dye colour for normals. */
export const LUMINANCE_WEIGHTS = [0.299, 0.587, 0.114] as const;

/** Exponent applied to the colour when `emitColor` is on, to create bright accents. */
export const EMISSIVE_ACCENT_POWER = 3;
