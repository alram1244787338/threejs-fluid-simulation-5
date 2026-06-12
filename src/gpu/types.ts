/*
 * Shared type definitions for the WebGPU (TSL) fluid simulation.
 *
 * Split out of FluidMaterialGPU.ts so the shader code, the material logic and the
 * type vocabulary each live in their own file.
 */
import type { ShaderNodeObject } from "three/tsl";
import type { Color, Object3D, TextureNode, UniformNode } from "three/webgpu";

/**
 * A TSL node produced by sampling a texture (i.e. the result of `texture(...)`).
 *
 * Note: this is the *sampling node*, not the GPU texture resource. The compute passes
 * read through nodes of this type but write into `StorageTexture` resources — keeping the
 * two distinct avoids the `Texture` / `StorageTexture` mix-up the old `Sampler2D` alias had.
 */
export type TextureSampleNode = ShaderNodeObject<TextureNode>;

/** A uniform node carrying a single scalar. */
export type NumberUniform = ShaderNodeObject<UniformNode<number>>;

/**
 * Handler invoked when a tracked object's `color` or `ratio` changes.
 * Receives the object that changed so the listener doesn't have to capture it.
 */
export type TrackedObjectChangeHandler = (source: TrackedObject) => void;

/**
 * One "slot" the fluid can react to. Its current/previous positions drive the splat
 * velocity, and its colour/ratio drive the dye. Mutating `color` or `ratio` notifies
 * `onChange` so the material can sync the GPU buffers.
 */
export class TrackedObject {
    target: Object3D | undefined;
    onChange?: TrackedObjectChangeHandler;

    private _color?: Color;
    private _ratio = 0;

    /** To let the system detect changes, assign a new value here — do not mutate in place. */
    get color() { return this._color; }
    get ratio() { return this._ratio; }

    set color(c: Color | undefined) {
        this._color = c;
        this.onChange?.(this);
    }

    set ratio(r: number) {
        this._ratio = r;
        this.onChange?.(this);
    }

    constructor(readonly index: number) {}
}

/** Construction-time options for {@link FluidMaterialGPU}. */
export type FluidMaterialSettings = {
    /** Use the colour as an emission source (pow'd to create accents). */
    emitColor?: boolean;

    /** Render with a transparent background. */
    transparent?: boolean;

    /** Max absolute speed in UV units. */
    maxSpeed: number;
};

/** Tunable simulation parameters, as serialised by the debug panel's "copy settings". */
export type FluidSettings = {
    splatForce: number;
    splatThickness: number;
    vorticityInfluence: number;
    swirlIntensity: number;
    pressureDecay: number;
    velocityDissipation: number;
    densityDissipation: number;
    bumpDisplacmentScale: number;
    pressureIterations: number;
};
