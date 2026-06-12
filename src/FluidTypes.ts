import type { Color, Object3D } from "three";
import type { ShaderNodeObject, TextureNode, UniformNode } from "three/tsl";

// ──────────────────────────────────────────────────────────────
//  Tracked-object change events
// ──────────────────────────────────────────────────────────────

/** Which property was modified when a `TrackedObjectChangeEvent` fires. */
export type TrackedObjectChangeProperty = "color" | "ratio";

/**
 * Event object passed to `TrackedObject.onChange` callbacks.
 * Replaces the previous bare `VoidFunction` signature so listeners
 * know *what* changed without re-reading the whole object.
 */
export interface TrackedObjectChangeEvent {
    /** The tracked object whose property was modified. */
    readonly target: TrackedObjectBase;
    /** Which property triggered the event. */
    readonly property: TrackedObjectChangeProperty;
}

/** Callback signature for tracked-object changes. */
export type TrackedObjectChangeCallback = (event: TrackedObjectChangeEvent) => void;

// ──────────────────────────────────────────────────────────────
//  Tracked object base (shared between GPU and WebGL)
// ──────────────────────────────────────────────────────────────

/**
 * Base shape shared by the WebGPU `TrackedObject` class and the
 * WebGL plain-object representation.
 */
export interface TrackedObjectBase {
    readonly index: number;
    target: Object3D | undefined;
    color: Color | undefined;
    ratio: number;
    onChange?: TrackedObjectChangeCallback;
}

// ──────────────────────────────────────────────────────────────
//  TSL convenience aliases (WebGPU only)
// ──────────────────────────────────────────────────────────────

/**
 * A TSL texture-sampler node.  The previous `Sampler2D` alias
 * (`ShaderNodeObject<TextureNode>`) was misleading because the
 * underlying value is actually a `Texture` or `StorageTexture` at
 * runtime — this alias is intentionally narrower and only used
 * where a shader *samples* a texture, never where one is *created*.
 */
export type TSLTextureSampler = ShaderNodeObject<TextureNode>;

/** A TSL uniform that wraps a single `number`. */
export type TSLNumberUniform = ShaderNodeObject<UniformNode<number>>;

// ──────────────────────────────────────────────────────────────
//  Settings (shared shapes)
// ──────────────────────────────────────────────────────────────

/**
 * Settings that can be copied from / pasted into the debug panel
 * for the **WebGPU** material.
 */
export interface FluidGPUSettings {
    splatForce: number;
    splatThickness: number;
    vorticityInfluence: number;
    swirlIntensity: number;
    pressureDecay: number;
    velocityDissipation: number;
    densityDissipation: number;
    bumpDisplacmentScale: number;
    pressureIterations: number;
}

/**
 * Settings that can be copied from / pasted into the debug panel
 * for the **WebGL** material.
 */
export interface FluidWebGLSettings {
    splatForce: number;
    splatThickness: number;
    vorticityInfluence: number;
    swirlIntensity: number;
    pressure: number;
    velocityDissipation: number;
    densityDissipation: number;
    displacementScale: number;
    pressureIterations: number;
}

/**
 * Constructor options for `FluidMaterialGPU`.
 */
export interface FluidMaterialGPUOptions {
    /**
     * When `true` the material uses the dye colour as emissive,
     * applying a cubic power curve to create brighter accents.
     */
    emitColor?: boolean;

    /**
     * When `true` the material will be transparent, with opacity
     * driven by the maximum channel value of the dye texture.
     */
    transparent?: boolean;

    /**
     * Maximum absolute speed in UV units per second.
     * @default 1/10
     */
    maxSpeed?: number;
}
