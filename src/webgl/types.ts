/*
 * Shared type definitions for the WebGL fluid simulation.
 *
 * Split out of FluidV3Material.ts so the shaders, the material logic and the type vocabulary
 * each live in their own file.
 */
import type { Color, Object3D } from "three";

/** One object the fluid reacts to, tracked by a fixed index into the data/position buffers. */
export type TargetObject = {
    target: Object3D | undefined;
    index: number;

    /** Fraction of a texel size; 1 = 100% of a texel. */
    ratio?: number;
    color?: Color;
};

/** Tunable simulation parameters, as serialised by the debug panel's "copy settings". */
export type FluidV3Settings = {
    splatForce: number;
    splatThickness: number;
    vorticityInfluence: number;
    swirlIntensity: number;
    pressure: number;
    velocityDissipation: number;
    densityDissipation: number;
    displacementScale: number;
    pressureIterations: number;
};
