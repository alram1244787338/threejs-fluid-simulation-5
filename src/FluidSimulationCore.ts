/**
 * Shared fluid simulation core: types, tracking, and unified pipeline.
 *
 * The `FluidSimulationPipeline` orchestrates the simulation loop and manages
 * tracked objects + settings. Each rendering backend (WebGL / WebGPU) implements
 * the `FluidBackend` interface and delegates public API calls to the pipeline.
 */
import GUI from "three/examples/jsm/libs/lil-gui.module.min.js";
import { Color, Mesh, Object3D, Raycaster, Vector2, Vector3, type ColorRepresentation } from "three";

// ─── Unified Settings ────────────────────────────────────────────────────────

export type FluidSimSettings = {
    splatForce: number;
    splatThickness: number;
    vorticityInfluence: number;
    swirlIntensity: number;
    pressureDecay: number;
    velocityDissipation: number;
    densityDissipation: number;
    bumpDisplacementScale: number;
    pressureIterations: number;
};

// ─── Tracked Object ──────────────────────────────────────────────────────────

export class TrackedObject {
    target: Object3D | undefined;
    onChange?: VoidFunction;

    private _color?: Color;
    private _ratio: number = 0;

    get color() { return this._color; }
    get ratio() { return this._ratio; }

    set color(c: Color | undefined) {
        this._color = c;
        this.onChange?.();
    }

    set ratio(r: number) {
        this._ratio = r;
        this.onChange?.();
    }

    constructor(readonly index: number) {}
}

// ─── Backend Interface ───────────────────────────────────────────────────────

/**
 * Each rendering backend (WebGPU / WebGL) implements this to provide the
 * pipeline with the concrete shader execution steps.
 */
export interface FluidBackend {
    scrollTextures(uvStep: Vector2): void;
    syncObjectBuffers(): void;

    stepSplatVelocity(): void;
    stepSplatColor(): void;
    stepCurl(): void;
    stepVorticity(delta: number): void;
    stepDivergence(): void;
    stepClearPressure(): void;
    stepPressure(): void;
    stepGradient(): void;
    stepAdvectVelocity(delta: number): void;
    stepAdvectDye(delta: number): void;
    finalizeFrame(): void;
}

// ─── Simulation Pipeline ─────────────────────────────────────────────────────

export class FluidSimulationPipeline {

    // Simulation parameters
    velocityDissipation = 0.283;
    densityDissipation = 0.2;
    pressureIterations = 39;

    // Tracking state
    private tracking: TrackedObject[];
    private objectPositionsArray: Float32Array;
    private objectDataArray: Float32Array;

    private raycaster = new Raycaster();
    private tmp = new Vector3();
    private tmp2 = new Vector3();

    private _follow?: Object3D;
    private lastFollowPos = new Vector3();
    private followOffset = new Vector3();

    get follow() { return this._follow; }
    set follow(obj: Object3D | undefined) {
        this._follow = obj;
        obj?.getWorldPosition(this.lastFollowPos);
    }

    constructor(
        private backend: FluidBackend,
        objectCount: number,
    ) {
        this.objectPositionsArray = new Float32Array(objectCount * 4);
        this.objectDataArray = new Float32Array(objectCount * 4);
        this.tracking = Array.from({ length: objectCount }, (_, i) => new TrackedObject(i));
    }

    // ── Tracking ─────────────────────────────────────────────────────────

    track(object: Object3D, ratio = 1, color: ColorRepresentation = Color.NAMES.black): TrackedObject {
        const freeSlot = this.tracking.find(slot => !slot.target);
        if (!freeSlot) throw new Error("No room for tracking, all slots taken!");

        const i = freeSlot.index;
        freeSlot.target = object;

        freeSlot.onChange = () => {
            this.objectDataArray[i * 4 + 0] = freeSlot.color?.r ?? 0;
            this.objectDataArray[i * 4 + 1] = freeSlot.color?.g ?? 0;
            this.objectDataArray[i * 4 + 2] = freeSlot.color?.b ?? 0;
            this.objectDataArray[i * 4 + 3] = freeSlot.ratio;

            if (!freeSlot.target) {
                this.objectPositionsArray[i * 4 + 0] = 0;
                this.objectPositionsArray[i * 4 + 1] = 0;
                this.objectPositionsArray[i * 4 + 2] = 0;
                this.objectPositionsArray[i * 4 + 3] = 0;
            }

            this.backend.syncObjectBuffers();
        };

        freeSlot.ratio = ratio;
        freeSlot.color = new Color(color);

        return freeSlot;
    }

    untrack(object: Object3D) {
        for (const t of this.tracking) {
            if (t.target === object) {
                t.target = undefined;
                t.ratio = 0;
                t.color = undefined;

                const i = t.index;
                this.objectPositionsArray[i * 4 + 0] = 0;
                this.objectPositionsArray[i * 4 + 1] = 0;
                this.objectPositionsArray[i * 4 + 2] = 0;
                this.objectPositionsArray[i * 4 + 3] = 0;
                this.backend.syncObjectBuffers();
            }
        }
    }

    // ── Position Update ──────────────────────────────────────────────────

    private updatePositions(mesh: Mesh) {
        if (this._follow) {
            this._follow.getWorldPosition(this.tmp);
            this.followOffset.copy(this.tmp).sub(this.lastFollowPos);
            this.followOffset.y = 0;
            this.lastFollowPos.copy(this.tmp);

            if (mesh.parent) mesh.parent.worldToLocal(this.tmp);
            mesh.position.x = this.tmp.x;
            mesh.position.z = this.tmp.z;
        }

        let offset: Vector2 | undefined;

        for (const obj of this.tracking) {
            if (!obj.target) continue;

            this.tmp.set(0, 1, 0);
            const wpos = obj.target.localToWorld(this.tmp);
            const followingObj = obj.target === this._follow;

            if (followingObj) wpos.sub(this.followOffset);

            this.tmp2.copy(wpos);
            const rpos = mesh.worldToLocal(this.tmp2);
            rpos.y = 0;
            mesh.localToWorld(rpos);

            this.raycaster.set(wpos, rpos.sub(wpos).normalize());
            const hit = this.raycaster.intersectObject(mesh, true);

            if (hit.length) {
                const uv = hit[0].uv;
                if (!uv) continue;

                const i = obj.index;
                if (followingObj) {
                    this.objectPositionsArray[i * 4 + 2] = uv.x;
                    this.objectPositionsArray[i * 4 + 3] = uv.y;
                    this.objectPositionsArray[i * 4 + 0] = 0.5;
                    this.objectPositionsArray[i * 4 + 1] = 0.5;
                    offset = new Vector2(0.5 - uv.x, 0.5 - uv.y);
                    this.backend.scrollTextures(offset);
                } else {
                    this.objectPositionsArray[i * 4 + 2] = this.objectPositionsArray[i * 4 + 0];
                    this.objectPositionsArray[i * 4 + 3] = this.objectPositionsArray[i * 4 + 1];
                    this.objectPositionsArray[i * 4 + 0] = uv.x;
                    this.objectPositionsArray[i * 4 + 1] = uv.y;
                }
            }
        }

        if (this._follow && offset) {
            for (const obj of this.tracking) {
                if (obj.target && obj.target !== this._follow) {
                    const i = obj.index;
                    this.objectPositionsArray[i * 4 + 2] -= offset.x;
                    this.objectPositionsArray[i * 4 + 3] -= offset.y;
                }
            }
        }

        this.backend.syncObjectBuffers();
    }

    // ── Template Method: Unified Simulation Loop ─────────────────────────

    update(delta: number, mesh: Mesh) {
        this.updatePositions(mesh);

        this.backend.stepSplatVelocity();
        this.backend.stepSplatColor();
        this.backend.stepCurl();
        this.backend.stepVorticity(delta);
        this.backend.stepDivergence();
        this.backend.stepClearPressure();

        for (let i = 0; i < this.pressureIterations; i++) {
            this.backend.stepPressure();
        }

        this.backend.stepGradient();
        this.backend.stepAdvectVelocity(delta);
        this.backend.stepAdvectDye(delta);
        this.backend.finalizeFrame();
    }

    // ── Debug GUI ────────────────────────────────────────────────────────

    addDebugPanelFolder(gui: GUI, settingsAccessor: FluidSettingsAccessor, name = "Fluid Material") {
        const panel = gui.addFolder(name);
        panel.add(settingsAccessor, "splatForce", -.5, .5);
        panel.add(settingsAccessor, "splatThickness", 0.001, 1);
        panel.add(settingsAccessor, "vorticityInfluence", 0.1, 1);
        panel.add(settingsAccessor, "swirlIntensity", 1, 100);
        panel.add(settingsAccessor, "pressureDecay", 0, 1);
        panel.add(settingsAccessor, "velocityDissipation", 0, 1);
        panel.add(settingsAccessor, "densityDissipation", 0, 1);
        panel.add(settingsAccessor, "bumpDisplacementScale", -1, 1);
        panel.add(settingsAccessor, "pressureIterations", 1, 100, 1);

        panel.add({
            copySettings: () => {
                const s: FluidSimSettings = {
                    splatForce: settingsAccessor.splatForce,
                    splatThickness: settingsAccessor.splatThickness,
                    vorticityInfluence: settingsAccessor.vorticityInfluence,
                    swirlIntensity: settingsAccessor.swirlIntensity,
                    pressureDecay: settingsAccessor.pressureDecay,
                    velocityDissipation: settingsAccessor.velocityDissipation,
                    densityDissipation: settingsAccessor.densityDissipation,
                    bumpDisplacementScale: settingsAccessor.bumpDisplacementScale,
                    pressureIterations: settingsAccessor.pressureIterations,
                };
                navigator.clipboard.writeText(JSON.stringify(s, null, 2));
            }
        }, "copySettings");

        return panel;
    }

    setSettingsFromPartial(s: Partial<FluidSimSettings>, accessor: FluidSettingsAccessor) {
        if (s.splatForce !== undefined) accessor.splatForce = s.splatForce;
        if (s.splatThickness !== undefined) accessor.splatThickness = s.splatThickness;
        if (s.vorticityInfluence !== undefined) accessor.vorticityInfluence = s.vorticityInfluence;
        if (s.swirlIntensity !== undefined) accessor.swirlIntensity = s.swirlIntensity;
        if (s.pressureDecay !== undefined) accessor.pressureDecay = s.pressureDecay;
        if (s.velocityDissipation !== undefined) accessor.velocityDissipation = s.velocityDissipation;
        if (s.densityDissipation !== undefined) accessor.densityDissipation = s.densityDissipation;
        if (s.bumpDisplacementScale !== undefined) accessor.bumpDisplacementScale = s.bumpDisplacementScale;
        if (s.pressureIterations !== undefined) accessor.pressureIterations = s.pressureIterations;
    }
}

// ─── Settings Accessor Interface ─────────────────────────────────────────────

/**
 * Implemented by each material to bridge the pipeline's unified setting names
 * to the backend-specific shader uniforms.
 */
export interface FluidSettingsAccessor {
    splatForce: number;
    splatThickness: number;
    vorticityInfluence: number;
    swirlIntensity: number;
    pressureDecay: number;
    velocityDissipation: number;
    densityDissipation: number;
    bumpDisplacementScale: number;
    pressureIterations: number;
}
