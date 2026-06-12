/*
MIT License

Copyright (c) 2017 Pavel Dobryakov : Original WebGL shader code (https://github.com/PavelDoGreat/WebGL-Fluid-Simulation/tree/master)
Copyright (c) 2025 Pablo Bandinopla (https://x.com/bandinopla) : Modificated and addapted for ThreeJs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/**
 * Centralized configuration system for the fluid simulation parameters.
 *
 * Everything about a tweakable parameter — its default value, its valid range,
 * how it is sanitized, and how it shows up in the debug GUI — is declared once
 * in a {@link ParamSchemaEntry}. A {@link FluidParams} instance then:
 *
 *  - seeds the defaults into the shaders,
 *  - installs auto-generated get/set accessors on the material,
 *  - validates every incoming value (e.g. dissipation can't go negative,
 *    `pressureIterations` can't be `0`),
 *  - builds the debug GUI controls automatically, and
 *  - notifies listeners whenever a value changes (handy for presets/animation).
 *
 * Adding a new parameter is therefore a matter of adding one entry to a schema
 * (here) plus one binding (in the owning material) that says where the value
 * should be pushed inside the shader graph.
 */

import GUI from "three/examples/jsm/libs/lil-gui.module.min.js";

/**
 * Sanitizes/clamps a raw value before it is stored and applied.
 * Returns the value that will actually be used.
 */
export type ParamValidator = (value: number) => number;

/**
 * Declarative description of a single tweakable parameter. Pure data — it holds
 * no reference to any shader, which keeps it usable as a module-level constant
 * (and lets us derive the public key union from it).
 */
export interface ParamSchemaEntry<K extends string = string> {
    /** Public property name exposed on the material (e.g. `"splatForce"`). */
    key: K;
    /** Label shown in the debug GUI. Defaults to `key`. */
    label?: string;
    /** The single source of truth for this parameter's default value. */
    default: number;
    /** Lower bound (used by the GUI slider and the default validator). */
    min: number;
    /** Upper bound (used by the GUI slider and the default validator). */
    max: number;
    /** Optional GUI step. When set, the value also snaps to it. */
    step?: number;
    /**
     * Optional custom sanitizer. When omitted the value is simply clamped to
     * `[min, max]`.
     */
    validate?: ParamValidator;
}

/** Pushes a validated value into the underlying shader uniform. */
export type ParamBinding = (value: number) => void;

/** Map of `key -> binding`. Keys without a binding live purely in the store. */
export type ParamBindings<K extends string> = Partial<Record<K, ParamBinding>>;

/** Called after a parameter changes value. */
export type FluidParamChangeListener<K extends string> = (
    key: K,
    value: number,
    all: Record<K, number>,
) => void;

// ---------------------------------------------------------------------------
//  Reusable validators
// ---------------------------------------------------------------------------

/** Clamp to the inclusive `[min, max]` range. */
export const clampRange = (min: number, max: number): ParamValidator =>
    (v) => Math.min(max, Math.max(min, v));

/** Round to the nearest integer, then clamp to `[min, max]`. */
export const integerInRange = (min: number, max: number): ParamValidator =>
    (v) => Math.min(max, Math.max(min, Math.round(v)));

// ---------------------------------------------------------------------------
//  The controller
// ---------------------------------------------------------------------------

export class FluidParams<K extends string> {

    /** Canonical, validated values. This is the source of truth at runtime. */
    private readonly values = {} as Record<K, number>;
    private readonly byKey = new Map<K, ParamSchemaEntry<K>>();
    private readonly listeners: FluidParamChangeListener<K>[] = [];

    constructor(
        private readonly schema: readonly ParamSchemaEntry<K>[],
        private readonly bindings: ParamBindings<K> = {},
    ) {
        for (const entry of schema) {
            this.byKey.set(entry.key, entry);
        }

        // Seed defaults (validated) and push them into the shaders so the
        // simulation starts from the values declared in the schema.
        for (const entry of schema) {
            const v = this.sanitize(entry, entry.default);
            this.values[entry.key] = v;
            this.bindings[entry.key]?.(v);
        }
    }

    private sanitize(entry: ParamSchemaEntry<K>, raw: number): number {
        let v = raw;
        if (typeof v !== "number" || Number.isNaN(v)) {
            v = entry.default;
        }
        const validate = entry.validate ?? clampRange(entry.min, entry.max);
        return validate(v);
    }

    /** Current (validated) value of a parameter. */
    get(key: K): number {
        return this.values[key];
    }

    /**
     * Validate `raw`, store it, push it to the shader and notify listeners.
     * No-op (and no notification) when the validated value is unchanged.
     */
    set(key: K, raw: number): void {
        const entry = this.byKey.get(key);
        if (!entry) return;

        const v = this.sanitize(entry, raw);
        if (this.values[key] === v) return;

        this.values[key] = v;
        this.bindings[key]?.(v);

        if (this.listeners.length) {
            const snapshot = this.snapshot();
            for (const listener of this.listeners) {
                listener(key, v, snapshot);
            }
        }
    }

    /**
     * Installs an auto-generated get/set accessor on `target` for every
     * parameter, so `material.splatForce` reads/writes through this controller
     * (with validation + notifications) without any hand-written boilerplate.
     */
    install(target: object): void {
        for (const entry of this.schema) {
            const key = entry.key;
            Object.defineProperty(target, key, {
                configurable: true,
                enumerable: true,
                get: () => this.get(key),
                set: (v: number) => this.set(key, v),
            });
        }
    }

    /** Shallow copy of all current values (e.g. to serialize a preset). */
    snapshot(): Record<K, number> {
        return { ...this.values };
    }

    /** Apply a (possibly partial) set of values, validating each one. */
    apply(settings: Partial<Record<K, number>>): void {
        for (const entry of this.schema) {
            const value = settings[entry.key];
            if (value != null) {
                this.set(entry.key, value);
            }
        }
    }

    /**
     * Subscribe to value changes. Returns an unsubscribe function.
     * Useful for preset switching and parameter animation.
     */
    onChange(listener: FluidParamChangeListener<K>): () => void {
        this.listeners.push(listener);
        return () => {
            const i = this.listeners.indexOf(listener);
            if (i >= 0) this.listeners.splice(i, 1);
        };
    }

    /**
     * Auto-builds a GUI control for every parameter under `panel`. Edits flow
     * back through {@link set}, so validation and change notifications apply to
     * GUI tweaks too.
     */
    addToGUI(panel: GUI): void {
        // A small live proxy so lil-gui reads/writes go through this controller.
        const proxy = {} as Record<K, number>;
        for (const entry of this.schema) {
            const key = entry.key;
            Object.defineProperty(proxy, key, {
                enumerable: true,
                get: () => this.get(key),
                set: (v: number) => this.set(key, v),
            });

            const controller = entry.step != null
                ? panel.add(proxy, key, entry.min, entry.max, entry.step)
                : panel.add(proxy, key, entry.min, entry.max);

            controller.name(entry.label ?? entry.key);
        }
    }
}

// ---------------------------------------------------------------------------
//  Parameter schemas (the single place where defaults / ranges / validation
//  for each simulation parameter live).
// ---------------------------------------------------------------------------

/** WebGL ({@link FluidV3Material}) parameter schema. */
export const FLUID_V3_PARAM_SCHEMA = [
    { key: "splatForce", default: -196, min: -1000, max: 1000 },
    { key: "splatThickness", default: 1, min: 0.001, max: 0.2 },
    { key: "vorticityInfluence", default: 1, min: 0.1, max: 1 },
    { key: "swirlIntensity", default: 1, min: 1, max: 100 },
    { key: "pressure", default: 0.317, min: 0, max: 1 },
    { key: "velocityDissipation", default: 0.283, min: 0, max: 1 },
    { key: "densityDissipation", default: 0.138, min: 0, max: 1 },
    { key: "displacementScale", default: 0.0078, min: -0.1, max: 0.1 },
    {
        key: "pressureIterations",
        default: 39,
        min: 1,
        max: 100,
        step: 1,
        validate: integerInRange(1, 100),
    },
] as const satisfies readonly ParamSchemaEntry[];

/** WebGPU ({@link FluidMaterialGPU}) parameter schema. */
export const FLUID_GPU_PARAM_SCHEMA = [
    { key: "splatForce", default: -0.1, min: -0.5, max: 0.5 },
    { key: "splatThickness", default: 1, min: 0.001, max: 1 },
    { key: "vorticityInfluence", default: 1, min: 0.1, max: 1 },
    { key: "swirlIntensity", default: 2, min: 1, max: 100 },
    { key: "pressureDecay", default: 0.317, min: 0, max: 1 },
    { key: "velocityDissipation", default: 0.283, min: 0, max: 1 },
    { key: "densityDissipation", default: 0.2, min: 0, max: 1 },
    { key: "bumpDisplacmentScale", default: 0.1, min: -1, max: 1 },
    {
        key: "pressureIterations",
        default: 39,
        min: 1,
        max: 100,
        step: 1,
        validate: integerInRange(1, 100),
    },
] as const satisfies readonly ParamSchemaEntry[];

export type FluidV3ParamKey = (typeof FLUID_V3_PARAM_SCHEMA)[number]["key"];
export type FluidGPUParamKey = (typeof FLUID_GPU_PARAM_SCHEMA)[number]["key"];

/** Serializable settings shapes derived directly from the schemas. */
export type FluidV3Settings = Record<FluidV3ParamKey, number>;
export type FluidGPUSettings = Record<FluidGPUParamKey, number>;
