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
import GUI from "three/examples/jsm/libs/lil-gui.module.min.js";
import { Color, ColorRepresentation, Mesh, Object3D, Raycaster, Vector2, Vector3 } from "three";

/**
 * Tunable parameters that drive the simulation. The names here are the single,
 * canonical set shared by every rendering backend (WebGL, WebGPU, ...). Backends
 * that historically used different names (e.g. `pressureDecay`,
 * `bumpDisplacmentScale`) map onto these in their `applySettings` adapter and in
 * {@link FluidSimulation.setSettings} (which still accepts the legacy aliases).
 */
export type FluidSettings = {
    /** Strength of the velocity injected along an object's movement. */
    splatForce: number;
    /** Radius (relative to a texel) of an object's influence. */
    splatThickness: number;
    /** How much curl is stored before the vorticity confinement pass. */
    vorticityInfluence: number;
    /** Strength of the vorticity confinement (swirl) force. */
    swirlIntensity: number;
    /** Per-iteration pressure decay (was `pressureDecay` on the WebGPU backend). */
    pressure: number;
    /** Dissipation applied to the velocity field while advecting. */
    velocityDissipation: number;
    /** Dissipation applied to the dye/color field while advecting. */
    densityDissipation: number;
    /** Height of the surface displacement (was `bumpDisplacmentScale` on WebGPU). */
    displacementScale: number;
    /** Jacobi iterations used to solve for pressure. */
    pressureIterations: number;
};

/** `[min, max]` slider bounds for each setting, used to build the debug panel. */
export type FluidSettingRanges = Record<keyof FluidSettings, [number, number]>;

/** Canonical ordered list of every tunable setting. */
export const FLUID_SETTING_KEYS: (keyof FluidSettings)[] = [
    "splatForce",
    "splatThickness",
    "vorticityInfluence",
    "swirlIntensity",
    "pressure",
    "velocityDissipation",
    "densityDissipation",
    "displacementScale",
    "pressureIterations",
];

/**
 * Everything a rendering backend (WebGL / WebGPU / ...) must provide so that the
 * shared {@link FluidSimulation} can drive the simulation. The backend only deals
 * with the API-specific plumbing: how a single step is dispatched, how settings
 * reach the shaders, and where the per-object data lives. The order in which the
 * steps run, the tracking bookkeeping and the follow logic all live in the
 * shared layer so the two backends can never drift apart.
 */
export interface FluidSimulationBackend {
    /** Per-object data buffer: `[r, g, b, ratio]` for each tracked slot. */
    readonly objectDataArray: Float32Array;
    /** Per-object positions: `[curr.x, curr.y, prev.x, prev.y]` for each slot. */
    readonly objectPositionArray: Float32Array;

    /** Flag the object data buffer as changed so the GPU picks it up. */
    markObjectDataDirty(): void;
    /** Flag the object position buffer as changed so the GPU picks it up. */
    markObjectPositionDirty(): void;

    /** Push the canonical settings into the backend's shader uniforms / nodes. */
    applySettings(settings: FluidSettings): void;

    /** Scroll both the velocity and dye fields by `uvStep` (used when following). */
    scroll(uvStep: Vector2): void;

    /** Inject velocity from the tracked objects into the velocity field. */
    splatVelocity(): void;
    /** Inject color from the tracked objects into the dye field. */
    splatColor(): void;
    /** Compute curl into the velocity field. */
    curl(): void;
    /** Apply the vorticity confinement force. */
    vorticity(delta: number): void;
    /** Compute the velocity divergence. */
    divergence(): void;
    /** Decay the stored pressure before the solve. */
    clearPressure(): void;
    /** A single Jacobi pressure iteration. */
    pressureStep(): void;
    /** Subtract the pressure gradient from the velocity field. */
    gradientSubtract(): void;
    /** Advect the velocity field. */
    advectVelocity(delta: number, dissipation: number): void;
    /** Advect the dye/color field. */
    advectColor(delta: number, dissipation: number): void;

    /** Publish the freshly simulated textures to the material for rendering. */
    present(): void;
}

/** One tracked object slot. Backends share this representation. */
type TrackedSlot = {
    target: Object3D | undefined;
    readonly index: number;
    ratio: number;
    readonly color: Color;
};

/**
 * Backend-agnostic fluid simulation pipeline.
 *
 * Owns the canonical {@link FluidSettings}, the tracked-object bookkeeping, the
 * "follow a target" behaviour and the fixed order of the simulation steps. A
 * concrete {@link FluidSimulationBackend} supplies the API-specific primitives.
 */
export class FluidSimulation {
    /** The single source of truth for every tunable parameter. */
    readonly settings: FluidSettings;

    private _follow?: Object3D;
    private readonly lastFollowPos = new Vector3();
    private readonly followOffset = new Vector3();

    private readonly tracking: TrackedSlot[];
    private readonly raycaster = new Raycaster();
    private readonly tmp = new Vector3();
    private readonly tmp2 = new Vector3();

    /** Accumulated simulation time (seconds). */
    private t = 0;

    constructor(
        private readonly backend: FluidSimulationBackend,
        objectCount: number,
        settings: FluidSettings,
    ) {
        this.settings = { ...settings };
        this.tracking = new Array(objectCount).fill(0).map((_, index) => ({
            target: undefined as Object3D | undefined,
            index,
            ratio: 1,
            color: new Color(),
        }));
    }

    /**
     * The mesh will follow this target, scrolling the textures so the target
     * stays centred (UV 0.5, 0.5).
     */
    get follow() {
        return this._follow;
    }
    set follow(obj: Object3D | undefined) {
        this._follow = obj;
        obj?.getWorldPosition(this.lastFollowPos);
    }

    /**
     * Register an object so its motion drives the liquid. Its current and past
     * positions are sampled (via raycast onto the surface) to derive a velocity.
     */
    track(object: Object3D, ratio = 1, color: ColorRepresentation = Color.NAMES.black): void {
        const slot = this.tracking.find((s) => !s.target);
        if (!slot) {
            throw new Error(`No room for tracking, all slots taken!`);
        }

        slot.target = object;
        slot.ratio = ratio;
        slot.color.set(color);

        const i = slot.index;
        const data = this.backend.objectDataArray;
        data[i * 4 + 0] = slot.color.r;
        data[i * 4 + 1] = slot.color.g;
        data[i * 4 + 2] = slot.color.b;
        data[i * 4 + 3] = ratio;

        this.backend.markObjectDataDirty();
    }

    /** Stop tracking an object, clearing both its color and position data. */
    untrack(object: Object3D): void {
        for (const slot of this.tracking) {
            if (slot.target !== object) continue;

            slot.target = undefined;
            slot.ratio = 0;
            slot.color.setRGB(0, 0, 0);

            const i = slot.index;

            const data = this.backend.objectDataArray;
            data[i * 4 + 0] = 0;
            data[i * 4 + 1] = 0;
            data[i * 4 + 2] = 0;
            data[i * 4 + 3] = 0;

            const pos = this.backend.objectPositionArray;
            pos[i * 4 + 0] = 0;
            pos[i * 4 + 1] = 0;
            pos[i * 4 + 2] = 0;
            pos[i * 4 + 3] = 0;

            this.backend.markObjectDataDirty();
            this.backend.markObjectPositionDirty();
        }
    }

    /**
     * Advance the simulation by `delta` seconds. The step order is fixed here so
     * both backends stay in lock-step:
     * scroll (via follow) → splat → curl → vorticity → divergence → clear →
     * pressure × N → gradient → advect velocity → advect dye → present.
     */
    update(delta: number, mesh: Mesh): void {
        this.t += delta;

        this.backend.applySettings(this.settings);
        this.updatePositions(mesh);

        // 1. add new velocities / colors based on object movement
        this.backend.splatVelocity();
        this.backend.splatColor();

        // 2. curl -> 3. vorticity confinement
        this.backend.curl();
        this.backend.vorticity(delta);

        // 4. divergence -> 5. clear pressure -> 6. solve pressure
        this.backend.divergence();
        this.backend.clearPressure();
        for (let i = 0; i < this.settings.pressureIterations; i++) {
            this.backend.pressureStep();
        }

        // 7. subtract pressure gradient
        this.backend.gradientSubtract();

        // 8. advect the velocity and dye fields
        this.backend.advectVelocity(delta, this.settings.velocityDissipation);
        this.backend.advectColor(delta, this.settings.densityDissipation);

        // 9. hand the result to the material for rendering
        this.backend.present();
    }

    /**
     * Update tracked object positions. We use UVs as positions: a ray is cast
     * from each object down onto the liquid surface and the hit UV becomes its
     * position. When following a target the textures are scrolled so it stays
     * centred.
     */
    private updatePositions(mesh: Mesh): void {
        if (this._follow) {
            // assumes Y is up and we only follow on the XZ plane
            this._follow.getWorldPosition(this.tmp);

            this.followOffset.copy(this.tmp).sub(this.lastFollowPos);
            this.followOffset.y = 0; // ignore the Y axis

            this.lastFollowPos.copy(this.tmp);

            if (mesh.parent) {
                mesh.parent.worldToLocal(this.tmp);
            }

            mesh.position.x = this.tmp.x;
            mesh.position.z = this.tmp.z;
        }

        const positions = this.backend.objectPositionArray;
        let offset: Vector2 | undefined; // UV offset produced by following

        for (const obj of this.tracking) {
            if (!obj.target) continue;

            this.tmp.set(0, 1, 0); // assumes the object origin is at the bottom of the model
            const wpos = obj.target.localToWorld(this.tmp);
            const followingObj = obj.target === this._follow;

            if (followingObj) {
                // sample the UV at the previous position: a followed object is
                // pinned to the dead-centre UV (0.5, 0.5).
                wpos.sub(this.followOffset);
            }

            this.tmp2.copy(wpos);

            const rpos = mesh.worldToLocal(this.tmp2);
            rpos.y = 0; // drop onto the surface of the mesh
            mesh.localToWorld(rpos); // point at the surface

            this.raycaster.set(wpos, rpos.sub(wpos).normalize());

            const hit = this.raycaster.intersectObject(mesh, true);
            if (!hit.length) continue;

            const uv = hit[0].uv; // UV under the object
            if (!uv) continue;

            const i = obj.index;

            if (followingObj) {
                // old positions
                positions[i * 4 + 2] = uv.x;
                positions[i * 4 + 3] = uv.y;

                // new positions (pinned to centre)
                positions[i * 4 + 0] = 0.5;
                positions[i * 4 + 1] = 0.5;

                offset = new Vector2(0.5 - uv.x, 0.5 - uv.y);
                this.scrollTextures(offset);
            } else {
                // old positions <- previous new positions
                positions[i * 4 + 2] = positions[i * 4 + 0];
                positions[i * 4 + 3] = positions[i * 4 + 1];

                // new positions
                positions[i * 4 + 0] = uv.x;
                positions[i * 4 + 1] = uv.y;
            }
        }

        if (this._follow && offset != null) {
            // the UV was scrolled, so remove the offset from every non-followed object
            for (const obj of this.tracking) {
                if (obj.target && obj.target !== this._follow) {
                    const i = obj.index;
                    positions[i * 4 + 2] -= offset.x;
                    positions[i * 4 + 3] -= offset.y;
                }
            }
        }

        this.backend.markObjectPositionDirty();
    }

    private scrollTextures(uvStep: Vector2): void {
        this.backend.scroll(uvStep);
    }

    /**
     * Apply a settings object previously copied from the debug panel. Legacy
     * per-backend aliases (`pressureDecay`, `bumpDisplacmentScale`) are still
     * accepted so older snapshots keep working.
     */
    setSettings(s: Partial<FluidSettings> & { pressureDecay?: number; bumpDisplacmentScale?: number }): void {
        const pick = (value: number | undefined, fallback: number) => (value !== undefined ? value : fallback);

        this.settings.splatForce = pick(s.splatForce, this.settings.splatForce);
        this.settings.splatThickness = pick(s.splatThickness, this.settings.splatThickness);
        this.settings.vorticityInfluence = pick(s.vorticityInfluence, this.settings.vorticityInfluence);
        this.settings.swirlIntensity = pick(s.swirlIntensity, this.settings.swirlIntensity);
        this.settings.pressure = pick(s.pressure, pick(s.pressureDecay, this.settings.pressure));
        this.settings.velocityDissipation = pick(s.velocityDissipation, this.settings.velocityDissipation);
        this.settings.densityDissipation = pick(s.densityDissipation, this.settings.densityDissipation);
        this.settings.displacementScale = pick(s.displacementScale, pick(s.bumpDisplacmentScale, this.settings.displacementScale));
        this.settings.pressureIterations = pick(s.pressureIterations, this.settings.pressureIterations);
    }

    /** Snapshot of the current settings (canonical names). */
    getSettings(): FluidSettings {
        return { ...this.settings };
    }

    /**
     * Build a debug GUI folder bound directly to {@link settings}. The slider
     * `ranges` are backend-specific (the two backends use different units), so
     * each material supplies its own. Returns the created folder so a backend can
     * append extra controls to it.
     */
    addDebugPanelFolder(gui: GUI, ranges: FluidSettingRanges, name = "Fluid Material"): GUI {
        const panel = gui.addFolder(name);

        for (const key of FLUID_SETTING_KEYS) {
            const [min, max] = ranges[key];
            // pressure iterations is an integer count
            const step = key === "pressureIterations" ? 1 : undefined;
            if (step !== undefined) {
                panel.add(this.settings as Record<string, number>, key, min, max, step);
            } else {
                panel.add(this.settings as Record<string, number>, key, min, max);
            }
        }

        panel.add(
            {
                copySettings: () => {
                    navigator.clipboard.writeText(JSON.stringify(this.getSettings(), null, 2));
                },
            },
            "copySettings",
        );

        return panel;
    }
}

/**
 * Install the canonical settings getters/setters onto a material instance so
 * `material.splatForce`, `material.pressure`, ... read and write the shared
 * {@link FluidSimulation.settings}. Defining them here once (instead of on each
 * material class) keeps the names and behaviour identical across backends.
 *
 * Materials declare the typed surface via interface declaration-merging, e.g.
 * `export interface MyMaterial extends FluidSettings {}`.
 */
export function installSettingsAccessors(target: object, simulation: FluidSimulation): void {
    for (const key of FLUID_SETTING_KEYS) {
        Object.defineProperty(target, key, {
            get() {
                return simulation.settings[key];
            },
            set(value: number) {
                simulation.settings[key] = value;
            },
            enumerable: true,
            configurable: true,
        });
    }
}
