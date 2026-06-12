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
import {
    cross, dot, Fn, max, modelNormalMatrix, normalGeometry, normalize, positionLocal, texture,
    uniform, uv, vec2, vec3, type ShaderNodeObject,
} from "three/tsl";
import {
    Color, MeshPhysicalNodeMaterial, Raycaster, StorageBufferAttribute, Vector2, Vector3,
    type ColorRepresentation, type Mesh, type Node, type Object3D, type WebGPURenderer,
} from "three/webgpu";
import {
    DEFAULT_BUMP_DISPLACEMENT_SCALE, DEFAULT_DENSITY_DISSIPATION, DEFAULT_MAX_SPEED,
    DEFAULT_PRESSURE_ITERATIONS, DEFAULT_VELOCITY_DISSIPATION, EMISSIVE_ACCENT_POWER, LUMINANCE_WEIGHTS,
} from "./gpu/constants";
import { PingPongTexture } from "./gpu/PingPongTexture";
import {
    AdvectShader, ClearShader, ComputeShader, CurlShader, DivergenceShader, GradientSubtractShader,
    placeholderTexture, PressureShader, ScrollShader, SplatShader, VorticityShader,
} from "./gpu/shaders";
import { TrackedObject } from "./gpu/types";
import type { FluidMaterialSettings, FluidSettings, NumberUniform, TextureSampleNode } from "./gpu/types";

// Re-exported so existing imports of `{ TrackedObject }` from this module keep working.
export { TrackedObject };
export type { FluidMaterialSettings, FluidSettings };

export class FluidMaterialGPU extends MeshPhysicalNodeMaterial {

    private _bumpDisplacmentScale = uniform(DEFAULT_BUMP_DISPLACEMENT_SCALE);

    /**
     * The mesh will follow this target, and the previous/current position will scroll the texture.
     */
    private _follow?: Object3D;
    public get follow() { return this._follow; }
    public set follow(obj: Object3D | undefined) {
        this._follow = obj;
        obj?.getWorldPosition(this.lastFollowPos);
    }

    private lastFollowPos: Vector3 = new Vector3();
    private followOffset: Vector3 = new Vector3();

    private raycaster: Raycaster;
    private tmp: Vector3 = new Vector3();
    private tmp2: Vector3 = new Vector3();

    // Ping-pong targets: `velocity` holds pressure + velocity, `dye` holds colour + elevation.
    private velocity: PingPongTexture;
    private dye: PingPongTexture;

    private uTarget: TextureSampleNode;

    private objectDataArray: Float32Array;
    private objectPositionsArray: Float32Array;
    private objectDataAttribute: StorageBufferAttribute;
    private objectPositionAttribute: StorageBufferAttribute;

    private tracking: TrackedObject[];
    private renderMaterial: (material: ComputeShader, target: PingPongTexture["write"]) => void;

    get splatForce() { return this.splat.splatForce.value; }
    set splatForce(v: number) { this.splat.splatForce.value = v; }

    get splatThickness() { return this.splat.thickness.value; }
    set splatThickness(v: number) { this.splat.thickness.value = v; }
    get vorticityInfluence() { return this.curl.vorticityInfluence.value; }
    set vorticityInfluence(v: number) { this.curl.vorticityInfluence.value = v; }

    get swirlIntensity() { return this.vorticity.curl.value; }
    set swirlIntensity(v: number) { this.vorticity.curl.value = v; }

    get pressureDecay() { return this.clear.decay.value; }
    set pressureDecay(v: number) { this.clear.decay.value = v; }

    get bumpDisplacmentScale() { return this._bumpDisplacmentScale.value; }
    set bumpDisplacmentScale(v: number) { this._bumpDisplacmentScale.value = v; }

    /** The dye (colour) texture. */
    get colorTexture() { return this.dye.read; }

    /** The simulation data texture (holds the surface velocities). */
    get dataTexture() { return this.velocity.read; }

    velocityDissipation = DEFAULT_VELOCITY_DISSIPATION;
    densityDissipation = DEFAULT_DENSITY_DISSIPATION;
    pressureIterations = DEFAULT_PRESSURE_ITERATIONS;
    actAsSmoke = true;

    private scroll: ScrollShader;
    private splat: SplatShader;
    private curl: CurlShader;
    private vorticity: VorticityShader;
    private divergence: DivergenceShader;
    private clear: ClearShader;
    private pressure: PressureShader;
    private gradient: GradientSubtractShader;
    private advect: AdvectShader;

    /** Max speed at which the liquid can move, in UV units. */
    private uMaxSpeed: NumberUniform;
    private t = 0;

    constructor(renderer: WebGPURenderer, textureWidth: number, textureHeight: number, objectCount = 1, settings?: Partial<FluidMaterialSettings>) {
        super({
            roughness: 0.5,
            color: new Color(0xcccccc),
            transparent: true,
        });

        this.uMaxSpeed = uniform(settings?.maxSpeed ?? DEFAULT_MAX_SPEED);

        this.raycaster = new Raycaster();

        this.velocity = new PingPongTexture(textureWidth, textureHeight);
        this.dye = new PingPongTexture(textureWidth, textureHeight);

        this.objectPositionsArray = new Float32Array(objectCount * 4);
        this.objectDataArray = new Float32Array(objectCount * 4);

        this.objectPositionAttribute = new StorageBufferAttribute(this.objectPositionsArray, 4);
        this.objectDataAttribute = new StorageBufferAttribute(this.objectDataArray, 4);

        this.tracking = new Array(objectCount).fill(0).map((_, index) => new TrackedObject(index));

        const texel = uniform(new Vector2(1 / textureWidth, 1 / textureHeight));

        this.uTarget = texture(placeholderTexture);

        const w = textureWidth;
        const h = textureHeight;
        const velocity = this.velocity.textures;
        const dye = this.dye.textures;

        this.scroll = new ScrollShader(this.uTarget).createBinds(w, h, ...velocity, ...dye);

        this.splat = new SplatShader(
            this.uTarget,
            this.objectPositionAttribute,
            this.objectDataAttribute,
            objectCount,
            this.uMaxSpeed,
        ).createBinds(w, h, ...velocity, ...dye);

        this.curl = new CurlShader(this.uTarget, this.uMaxSpeed).createBinds(w, h, ...velocity);
        this.vorticity = new VorticityShader(this.uTarget, this.uMaxSpeed).createBinds(w, h, ...velocity);
        this.divergence = new DivergenceShader(this.uTarget, this.uMaxSpeed).createBinds(w, h, ...velocity);
        this.clear = new ClearShader(this.uTarget).createBinds(w, h, ...velocity);
        this.pressure = new PressureShader(this.uTarget, this.uMaxSpeed).createBinds(w, h, ...velocity);
        this.gradient = new GradientSubtractShader(this.uTarget, this.uMaxSpeed).createBinds(w, h, ...velocity);
        this.advect = new AdvectShader(this.uTarget, this.uMaxSpeed).createBinds(w, h, ...velocity, ...dye);

        this.renderMaterial = (material, target) => {
            material.renderBind(renderer, target);
        };

        // --- displacement: lift the surface by the brightest dye channel ---
        const maxChannel = this.uTarget.sample(uv());
        const maxValue = max(maxChannel.r.clamp(0, 1), max(maxChannel.g.clamp(0, 1), maxChannel.b.clamp(0, 1)));

        this.positionNode = positionLocal.add(normalGeometry.mul(maxValue.mul(this._bumpDisplacmentScale)));

        if (settings?.transparent) {
            this.opacityNode = maxValue;
        }

        this.colorNode = this.uTarget;

        // --- recompute normals so lighting respects the displacement ---
        const height = Fn<[ShaderNodeObject<Node>]>(([uvOffset]) =>
            dot(this.uTarget.sample(uv().add(uvOffset)).rgb, vec3(LUMINANCE_WEIGHTS[0], LUMINANCE_WEIGHTS[1], LUMINANCE_WEIGHTS[2])));

        this.normalNode = Fn(() => {
            const scale = this._bumpDisplacmentScale;

            const hL = height(vec2(texel.x.negate(), 0.0));
            const hR = height(vec2(texel.x, 0.0));
            const hD = height(vec2(0.0, texel.y.negate()));
            const hU = height(vec2(0.0, texel.y));

            const dx = vec3(texel.x.mul(2), hR.sub(hL).mul(scale), 0.0);
            const dy = vec3(0.0, hU.sub(hD).mul(scale), texel.y.mul(2));

            const normal = normalize(cross(dy, dx));
            return normalize(modelNormalMatrix.mul(normal));
        })();

        if (settings?.emitColor) {
            this.emissiveNode = maxChannel.pow(EMISSIVE_ACCENT_POWER);
        }
    }

    /**
     * Add an object to be tracked so it affects the liquid. Its current and past positions
     * give its directional speed.
     */
    track(object: Object3D, ratio = 1, color: ColorRepresentation = Color.NAMES.black) {
        const freeSlot = this.tracking.find(slot => !slot.target);
        if (!freeSlot) {
            throw new Error(`No room for tracking, all slots taken!`);
        }

        // Raycast from the object down onto the surface to find the UV beneath it (done in updatePositions).
        freeSlot.target = object;

        freeSlot.onChange = (source) => {
            const i = source.index;

            this.objectDataArray[i * 4] = source.color?.r ?? 0;
            this.objectDataArray[i * 4 + 1] = source.color?.g ?? 0;
            this.objectDataArray[i * 4 + 2] = source.color?.b ?? 0;
            this.objectDataArray[i * 4 + 3] = source.ratio;

            if (!source.target) {
                this.objectPositionsArray[i * 4] = 0;
                this.objectPositionsArray[i * 4 + 1] = 0;
                this.objectPositionsArray[i * 4 + 2] = 0;
                this.objectPositionsArray[i * 4 + 3] = 0;

                this.objectPositionAttribute.needsUpdate = true;
            }

            this.objectDataAttribute.needsUpdate = true;
        };

        freeSlot.ratio = ratio;
        freeSlot.color = new Color(color);

        return freeSlot;
    }

    untrack(object: Object3D) {
        let removed = false;

        this.tracking.forEach((t, i) => {
            if (t.target == object) {
                removed = true;
                t.target = undefined;
                t.ratio = 0;
                t.color = undefined;

                this.objectPositionsArray[i * 4] = 0;
                this.objectPositionsArray[i * 4 + 1] = 0;
                this.objectPositionsArray[i * 4 + 2] = 0;
                this.objectPositionsArray[i * 4 + 3] = 0;
                this.objectPositionAttribute.needsUpdate = true;
            }
        });

        if (!removed) {
            console.warn("FluidMaterialGPU.untrack: the given object was not being tracked; nothing to remove.", object);
        }
    }

    /**
     * Renders the material into the velocity write target, then swaps so the freshly generated
     * texture becomes the new read target.
     */
    private blit(material: ComputeShader) {
        this.renderMaterial(material, this.velocity.write);
        this.velocity.swap();
        this.uTarget.value = this.velocity.read;
    }

    private blitDye(material: ComputeShader) {
        this.renderMaterial(material, this.dye.write);
        this.dye.swap();
        this.uTarget.value = this.velocity.read;
    }

    private scrollTextures(uvStep: Vector2) {
        this.scroll.uvScroll.value = uvStep;
        this.uTarget.value = this.velocity.read;
        this.blit(this.scroll);

        this.uTarget.value = this.dye.read;
        this.blitDye(this.scroll);
    }

    /**
     * Update the positions. We use UVs as positions: a ray is cast from each object onto the
     * surface, and the UV beneath the object is used as its position.
     */
    private updatePositions(mesh: Mesh) {

        if (this.follow) { // assumes Y is up and we follow only in the XZ plane
            this.follow.getWorldPosition(this.tmp);

            this.followOffset.copy(this.tmp).sub(this.lastFollowPos);
            this.followOffset.y = 0; // ignore the Y axis

            this.lastFollowPos.copy(this.tmp);

            if (mesh.parent) {
                mesh.parent.worldToLocal(this.tmp);
            }

            mesh.position.x = this.tmp.x;
            mesh.position.z = this.tmp.z;
        }

        let offset: Vector2 | undefined; // UV offset

        for (const obj of this.tracking) {

            if (!obj.target) continue;

            this.tmp.set(0, 1, 0); // assumes the object's origin is at the bottom of the model
            const wpos = obj.target.localToWorld(this.tmp);
            const followingObj = obj.target == this.follow;

            if (followingObj) {
                // sample the UV at the last position, since the followed object stays at UV 0.5,0.5
                wpos.sub(this.followOffset);
            }

            this.tmp2.copy(wpos);

            const rpos = mesh.worldToLocal(this.tmp2);
            rpos.y = 0; // put the position at the surface of the mesh

            mesh.localToWorld(rpos); // point at the surface of the mesh

            this.raycaster.set(wpos, rpos.sub(wpos).normalize());

            const hit = this.raycaster.intersectObject(mesh, true);

            if (hit.length) {
                const uv = hit[0].uv; // UV under the object

                if (uv) {
                    const i = obj.index;

                    if (followingObj) {
                        // old positions
                        this.objectPositionsArray[i * 4 + 2] = uv.x;
                        this.objectPositionsArray[i * 4 + 3] = uv.y;

                        // new positions
                        this.objectPositionsArray[i * 4 + 0] = 0.5;
                        this.objectPositionsArray[i * 4 + 1] = 0.5;

                        offset = new Vector2(0.5 - uv.x, 0.5 - uv.y);

                        this.scrollTextures(offset);
                    } else {
                        // old positions
                        this.objectPositionsArray[i * 4 + 2] = this.objectPositionsArray[i * 4];
                        this.objectPositionsArray[i * 4 + 3] = this.objectPositionsArray[i * 4 + 1];

                        // new positions
                        this.objectPositionsArray[i * 4] = uv.x;
                        this.objectPositionsArray[i * 4 + 1] = uv.y;
                    }
                }
            }
        }

        if (this.follow && offset != null) {
            // the UV was scrolled, so subtract this offset from all positions except the follow target
            this.tracking.forEach(obj => {
                if (obj.target && obj.target != this.follow) {
                    const i = obj.index;
                    this.objectPositionsArray[i * 4 + 2] -= offset!.x;
                    this.objectPositionsArray[i * 4 + 3] -= offset!.y;
                }
            });
        }

        this.objectPositionAttribute.needsUpdate = true;
    }

    update(delta: number, mesh: Mesh) {
        this.t += delta;

        this.uTarget.value = this.velocity.read;

        this.updatePositions(mesh);

        // Splat velocity
        this.splat.splatVelocity.value = 1;
        this.blit(this.splat);

        // Splat colours
        this.splat.splatVelocity.value = 0;
        this.uTarget.value = this.dye.read;
        this.blitDye(this.splat);

        // 2. curl into the alpha channel
        this.blit(this.curl);

        // 3. apply vorticity forces
        this.vorticity.delta.value = delta;
        this.blit(this.vorticity);

        // 4. divergence
        this.blit(this.divergence);

        // 5. clear pressure
        this.blit(this.clear);

        // 6. solve pressure
        for (let i = 0; i < this.pressureIterations; i++) {
            this.blit(this.pressure);
        }

        // 7. subtract pressure gradient
        this.blit(this.gradient);

        // 8. advect velocity
        this.advect.delta.value = delta;
        this.advect.uSource.value = this.velocity.read;
        this.advect.sourceIsVelocity.value = 1;
        this.advect.dissipation.value = this.velocityDissipation;
        this.blit(this.advect);

        // 9. advect dye / colour
        this.advect.uSource.value = this.dye.read;
        this.advect.sourceIsVelocity.value = 0;
        this.advect.dissipation.value = this.densityDissipation;
        this.blitDye(this.advect);

        this.uTarget.value = this.dye.read;
    }

    addDebugPanelFolder(gui: GUI, name = "Fluid Material") {

        const panel = gui.addFolder(name);

        panel.add(this as FluidMaterialGPU, "splatForce", -.5, .5);
        panel.add(this as FluidMaterialGPU, "splatThickness", 0.001, 1);
        panel.add(this as FluidMaterialGPU, "vorticityInfluence", 0.1, 1);
        panel.add(this as FluidMaterialGPU, "swirlIntensity", 1, 100);
        panel.add(this as FluidMaterialGPU, "pressureDecay", 0, 1);
        panel.add(this as FluidMaterialGPU, "velocityDissipation", 0, 1);
        panel.add(this as FluidMaterialGPU, "densityDissipation", 0, 1);
        panel.add(this as FluidMaterialGPU, "bumpDisplacmentScale", -1, 1);
        panel.add(this as FluidMaterialGPU, "pressureIterations", 1, 100, 1);

        panel.add({
            copySettings: () => {
                const settings: FluidSettings = {
                    splatForce: this.splatForce,
                    splatThickness: this.splatThickness,
                    vorticityInfluence: this.vorticityInfluence,
                    swirlIntensity: this.swirlIntensity,
                    pressureDecay: this.pressureDecay,
                    velocityDissipation: this.velocityDissipation,
                    densityDissipation: this.densityDissipation,
                    bumpDisplacmentScale: this.bumpDisplacmentScale,
                    pressureIterations: this.pressureIterations,
                };

                navigator.clipboard.writeText(JSON.stringify(settings, null, 2));
            },
        }, "copySettings");

        return panel;
    }

    /**
     * Restore values previously copied from the debug panel.
     * @see `addDebugPanelFolder`
     */
    setSettings(s: FluidSettings) {
        this.splatForce = s.splatForce;
        this.splatThickness = s.splatThickness;
        this.vorticityInfluence = s.vorticityInfluence;
        this.swirlIntensity = s.swirlIntensity;
        this.pressureDecay = s.pressureDecay;
        this.velocityDissipation = s.velocityDissipation;
        this.densityDissipation = s.densityDissipation;
        this.bumpDisplacmentScale = s.bumpDisplacmentScale;
        this.pressureIterations = s.pressureIterations;
    }
}
