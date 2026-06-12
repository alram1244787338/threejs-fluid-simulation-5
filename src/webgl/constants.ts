/*
 * Named constants for the WebGL fluid simulation.
 *
 * Collected here so every "magic number" that used to be buried in a GLSL string or a uniform
 * default has a documented home. JS-side values feed uniforms / material fields directly;
 * GLSL-side values are interpolated into the shader source via {@link glslFloat}.
 *
 * Original WebGL shader code (c) 2017 Pavel Dobryakov, adapted for ThreeJs (c) 2025 Pablo Bandinopla. MIT.
 */

// --------------------------------------------------------------------------------------------
// Simulation defaults (JS side — used as uniform defaults / material fields)
// --------------------------------------------------------------------------------------------

/**
 * Default splat "force" for the WebGL backend. Larger magnitude than the WebGPU backend because
 * here it is divided by the texel size in the shader rather than scaled by the max speed.
 */
export const DEFAULT_SPLAT_FORCE = -196;

/** Radius of a splat, in UV units (scaled per tracked object by its `ratio`). */
export const DEFAULT_SPLAT_THICKNESS = 1;

/** How strongly curl is written into the velocity field by the curl pass. */
export const DEFAULT_VORTICITY_INFLUENCE = 1;

/** Swirl strength applied by the vorticity-confinement pass. */
export const DEFAULT_SWIRL_INTENSITY = 1;

/** Per-iteration pressure decay used by the clear pass (a.k.a. "pressure" in the panel). */
export const DEFAULT_PRESSURE_DECAY = 0.317;

/** Default advection dissipation (overridden per-pass by the material). */
export const DEFAULT_ADVECTION_DISSIPATION = 0.2;

/** Velocity field dissipation per second applied during advection. */
export const DEFAULT_VELOCITY_DISSIPATION = 0.283;

/** Dye/colour dissipation per second applied during advection. */
export const DEFAULT_DENSITY_DISSIPATION = 0.138;

/** Number of Jacobi iterations used to solve for pressure. */
export const DEFAULT_PRESSURE_ITERATIONS = 39;

/** Vertical displacement of the mesh surface driven by the dye height. */
export const DEFAULT_DISPLACEMENT_SCALE = 0.0078;

// --------------------------------------------------------------------------------------------
// Numerical scheme constants (GLSL side — interpolated into shader source)
// --------------------------------------------------------------------------------------------

/** Central-difference factor: derivatives use (right - left) * 0.5 over a two-texel span. */
export const CENTRAL_DIFFERENCE_SCALE = 0.5;

/** Jacobi relaxation factor for the pressure solve: (L + R + T + B - divergence) / 4. */
export const PRESSURE_JACOBI_SCALE = 0.25;

/** Epsilon added before normalising the vorticity force to avoid a divide-by-zero. */
export const VORTICITY_EPSILON = 0.0001;

/** Hard clamp on the per-axis velocity, to keep the simulation from exploding. */
export const MAX_VELOCITY_CLAMP = 1000;

// --------------------------------------------------------------------------------------------
// Surface shading constants (GLSL side)
// --------------------------------------------------------------------------------------------

/** Rec. 601 luma weights, used to derive a height field from the dye colour for normals. */
export const LUMINANCE_WEIGHTS = [0.299, 0.587, 0.114] as const;

/** Neutral luminance: dye luminance above this raises the surface, below it lowers it. */
export const LUMINANCE_MIDPOINT = 0.5;

/**
 * Format a number as a GLSL float literal, guaranteeing a decimal point so integer-valued
 * constants (e.g. 1000) don't become invalid `int` literals where a `float` is expected.
 * Intended for the simple decimal constants above (not values that stringify to exponent form).
 */
export function glslFloat(value: number): string {
    return Number.isInteger(value) ? `${value}.0` : `${value}`;
}
