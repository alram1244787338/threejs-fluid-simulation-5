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
    Color, DataTexture, FloatType, LinearFilter, MeshPhysicalMaterial, NearestFilter, Raycaster,
    RGBAFormat, Vector2, Vector3,
    type ColorRepresentation, type Mesh, type Object3D, type ShaderMaterial, type WebGLProgramParametersWithUniforms,
    type WebGLRenderer,
} from "three";
import { FullScreenQuad } from "three/examples/jsm/Addons.js";
import {
    DEFAULT_DENSITY_DISSIPATION, DEFAULT_DISPLACEMENT_SCALE, DEFAULT_PRESSURE_ITERATIONS,
    DEFAULT_VELOCITY_DISSIPATION,
} from "./webgl/constants";
import { applyDisplacementShader } from "./webgl/displacement";
import { PingPongTarget } from "./webgl/PingPongTarget";
import {
    AdvectVelocityShader, ClearShader, CurlShader, DivergenceShader, GradientSubtractShader,
    PressureShader, ScrollShader, SplatShader, VorticityShader,
} from "./webgl/shaders";
import type { FluidV3Settings, TargetObject } from "./webgl/types";

/**
 * R - Pressure
 * G - X dir
 * B - Y dir
 * A - wildcard, used to pass values from shader to shader. Not persisted.
 */
export class FluidV3Material extends MeshPhysicalMaterial {

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

    private tracking: TargetObject[];

    // Ping-pong targets: `velocity` holds pressure + velocity, `dye` holds colour + elevation.
    private velocity: PingPongTarget;
    private dye: PingPongTarget;

    /** The dye (colour) texture. */
    get colorTexture() {
        return this.dye.texture;
    }

    /** The simulation data texture (holds the surface velocities). */
    get dataTexture() {
        return this.velocity.texture;
    }

    private quad: FullScreenQuad;
    private raycaster: Raycaster;
    private tmp: Vector3 = new Vector3();
    private tmp2: Vector3 = new Vector3();

    private objectPositionTexture: DataTexture;
    private objectPositionArray: Float32Array;

    private objectDataTexture: DataTexture;
    private objectDataArray: Float32Array;

    // shaders involved in the simulation
    private scroll: ScrollShader;
    private splat: SplatShader;
    private curl: CurlShader;
    private vorticity: VorticityShader;
    private divergenceShader: DivergenceShader;
    private clearShader: ClearShader;
    private pressureShader: PressureShader;
    private gradientShader: GradientSubtractShader;
    private advectionShader: AdvectVelocityShader;
    private supportLinearFiltering: boolean;

    private t = 0;

    /** If `true`, on every update the `alphaMap` is set to the colour map. */
    private actAsSmoke = false;

    constructor(private renderer: WebGLRenderer, textureWidth: number, textureHeight: number, objectCount = 1) {
        const aspect = textureWidth / textureHeight;

        super({
            roughness: 1,
            color: new Color(0xffffff),
            displacementScale: DEFAULT_DISPLACEMENT_SCALE,
            transparent: true,
        });

        // Devices that can't linearly filter float textures use point sampling + the shader's
        // software bilerp; everything else uses hardware linear filtering.
        const gl = renderer.getContext();
        this.supportLinearFiltering = !!gl.getExtension('OES_texture_half_float_linear');
        const filter = this.supportLinearFiltering ? LinearFilter : NearestFilter;

        // ping pong render textures
        this.velocity = new PingPongTarget(textureWidth, textureHeight, filter);
        this.dye = new PingPongTarget(textureWidth, textureHeight, filter);

        this.objectDataArray = new Float32Array(objectCount * 4); // color + ratio: R, G, B, ratio
        this.objectPositionArray = new Float32Array(objectCount * 4); // current + old UV positions

        this.objectDataTexture = new DataTexture(this.objectDataArray, objectCount, 1, RGBAFormat, FloatType);
        this.objectPositionTexture = new DataTexture(this.objectPositionArray, objectCount, 1, RGBAFormat, FloatType);

        this.tracking = new Array(objectCount).fill(0).map((_, index) => ({ target: undefined, index, ratio: 1 }));

        this.quad = new FullScreenQuad();
        this.raycaster = new Raycaster();

        const texel = new Vector2(1 / textureWidth, 1 / textureHeight);

        // ----- shaders used to simulate the liquid -----
        this.scroll = new ScrollShader(texel);
        this.splat = new SplatShader(texel, objectCount, aspect);
        this.curl = new CurlShader(texel);
        this.vorticity = new VorticityShader(texel);
        this.divergenceShader = new DivergenceShader(texel);
        this.clearShader = new ClearShader(texel);
        this.pressureShader = new PressureShader(texel);
        this.gradientShader = new GradientSubtractShader(texel);
        this.advectionShader = new AdvectVelocityShader(texel, texel, !this.supportLinearFiltering);
    }

    get splatForce() { return this.splat.uniforms.splatForce.value; }
    set splatForce(v: number) { this.splat.uniforms.splatForce.value = v; }

    get splatThickness() { return this.splat.uniforms.thickness.value; }
    set splatThickness(v: number) { this.splat.uniforms.thickness.value = v; }
    get vorticityInfluence() { return this.curl.uniforms.vorticityInfluence.value; }
    set vorticityInfluence(v: number) { this.curl.uniforms.vorticityInfluence.value = v; }

    get swirlIntensity() { return this.vorticity.uniforms.curl.value; }
    set swirlIntensity(v: number) { this.vorticity.uniforms.curl.value = v; }

    get pressure() { return this.clearShader.uniforms.value.value; }
    set pressure(v: number) { this.clearShader.uniforms.value.value = v; }

    velocityDissipation = DEFAULT_VELOCITY_DISSIPATION;
    densityDissipation = DEFAULT_DENSITY_DISSIPATION;
    pressureIterations = DEFAULT_PRESSURE_ITERATIONS;

    /**
     * Make normals respect the displacement. Delegates to the displacement patcher, which fails
     * loudly (rather than silently) if a future three.js renames a shader chunk.
     */
    override onBeforeCompile(shader: WebGLProgramParametersWithUniforms): void {
        applyDisplacementShader(shader);
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

        freeSlot.target = object;
        freeSlot.ratio = ratio;
        freeSlot.color = new Color(color);

        const i = freeSlot.index;

        // NOTE: the R channel is written at index `i` (rather than `i * 4`) to preserve the
        // original behaviour; the other channels use `i * 4 + n`.
        this.objectDataArray[i] = freeSlot.color.r;
        this.objectDataArray[i * 4 + 1] = freeSlot.color.g;
        this.objectDataArray[i * 4 + 2] = freeSlot.color.b;
        this.objectDataArray[i * 4 + 3] = ratio;

        this.objectDataTexture.needsUpdate = true;
    }

    untrack(object: Object3D) {
        let removed = false;

        this.tracking.forEach(t => {
            if (t.target == object) {
                removed = true;
                t.target = undefined;
                t.ratio = 1;
                t.color?.set(0, 0, 0);

                const i = t.index;

                // Mirrors the indexing used in track().
                this.objectDataArray[i] = 0;
                this.objectDataArray[i * 4 + 1] = 0;
                this.objectDataArray[i * 4 + 2] = 0;
                this.objectDataArray[i * 4 + 3] = 0;
                this.objectDataTexture.needsUpdate = true;
            }
        });

        if (!removed) {
            console.warn("FluidV3Material.untrack: the given object was not being tracked; nothing to remove.", object);
        }
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

        this.tracking.forEach(obj => {

            if (!obj.target) return;

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
                        this.objectPositionArray[i * 4 + 2] = uv.x;
                        this.objectPositionArray[i * 4 + 3] = uv.y;

                        // new positions
                        this.objectPositionArray[i * 4 + 0] = 0.5;
                        this.objectPositionArray[i * 4 + 1] = 0.5;

                        offset = new Vector2(0.5 - uv.x, 0.5 - uv.y);

                        this.scrollTextures(offset);
                    } else {
                        // old positions
                        this.objectPositionArray[i * 4 + 2] = this.objectPositionArray[i * 4 + 0];
                        this.objectPositionArray[i * 4 + 3] = this.objectPositionArray[i * 4 + 1];

                        // new positions
                        this.objectPositionArray[i * 4 + 0] = uv.x;
                        this.objectPositionArray[i * 4 + 1] = uv.y;
                    }
                }
            }
        });

        if (this.follow && offset != null) {
            // the UV was scrolled, so subtract this offset from all positions except the follow target
            this.tracking.forEach(obj => {
                if (obj.target && obj.target != this.follow) {
                    const i = obj.index;
                    this.objectPositionArray[i * 4 + 2] -= offset!.x;
                    this.objectPositionArray[i * 4 + 3] -= offset!.y;
                }
            });
        }

        this.objectPositionTexture.needsUpdate = true;
    }

    /**
     * Renders the material into the velocity write target, then swaps so the freshly generated
     * texture becomes the new read target.
     */
    private blit(material: ShaderMaterial) {
        this.renderer.setRenderTarget(this.velocity.write);
        this.quad.material = material;
        this.quad.render(this.renderer);
        this.velocity.swap();
    }

    private blitDye(material: ShaderMaterial) {
        this.renderer.setRenderTarget(this.dye.write);
        this.quad.material = material;
        this.quad.render(this.renderer);
        this.dye.swap();
    }

    private scrollTextures(uvStep: Vector2) {
        this.scroll.uniforms.uvScroll.value = uvStep;

        this.scroll.uniforms.uTarget.value = this.velocity.texture;
        this.blit(this.scroll);

        this.scroll.uniforms.uTarget.value = this.dye.texture;
        this.blitDye(this.scroll);
    }

    /**
     * @param delta
     * @param mesh The plane mesh used to simulate the liquid.
     */
    update(delta: number, mesh: Mesh) {
        this.t += delta;

        this.updatePositions(mesh);

        // 1. add new velocities based on object movement
        this.splat.uniforms.objectData.value = this.objectDataTexture;
        this.splat.uniforms.objectPosition.value = this.objectPositionTexture;
        this.splat.uniforms.uTarget.value = this.velocity.texture;
        this.splat.uniforms.splatVelocity.value = true;

        this.blit(this.splat);

        // add colours
        this.splat.uniforms.objectData.value = this.objectDataTexture;
        this.splat.uniforms.objectPosition.value = this.objectPositionTexture;
        this.splat.uniforms.uTarget.value = this.dye.texture;
        this.splat.uniforms.splatVelocity.value = false;

        this.blitDye(this.splat);

        // 2. curl into the alpha channel
        this.curl.uniforms.uVelocity.value = this.velocity.texture;
        this.blit(this.curl);

        // 3. apply vorticity forces
        this.vorticity.uniforms.uVelocityAndCurl.value = this.velocity.texture;
        this.vorticity.uniforms.dt.value = delta;
        this.blit(this.vorticity);

        // 4. divergence
        this.divergenceShader.uniforms.uVelocity.value = this.velocity.texture;
        this.blit(this.divergenceShader);

        // 5. clear pressure
        this.clearShader.uniforms.uTexture.value = this.velocity.texture;
        this.blit(this.clearShader);

        // 6. solve pressure
        for (let i = 0; i < this.pressureIterations; i++) {
            this.pressureShader.uniforms.uPressureWithDivergence.value = this.velocity.texture;
            this.blit(this.pressureShader);
        }

        // 7. subtract pressure gradient
        this.gradientShader.uniforms.uPressureWithVelocity.value = this.velocity.texture;
        this.blit(this.gradientShader);

        // 8. advect velocity
        this.advectionShader.uniforms.dt.value = delta;
        this.advectionShader.uniforms.uVelocity.value = this.velocity.texture;
        this.advectionShader.uniforms.uSource.value = this.velocity.texture;
        this.advectionShader.uniforms.sourceIsVelocity.value = true;
        this.advectionShader.uniforms.dissipation.value = this.velocityDissipation;
        this.blit(this.advectionShader);

        // 9. advect dye / colour
        this.advectionShader.uniforms.uVelocity.value = this.velocity.texture;
        this.advectionShader.uniforms.uSource.value = this.dye.texture;
        this.advectionShader.uniforms.sourceIsVelocity.value = false;
        this.advectionShader.uniforms.dissipation.value = this.densityDissipation;
        this.blitDye(this.advectionShader);

        this.renderer.setRenderTarget(null);

        this.displacementMap = this.dye.texture;

        if (this.actAsSmoke) {
            this.alphaMap = this.dye.texture;
        }
        this.map = this.dye.texture;
    }

    addDebugPanelFolder(gui: GUI, name = "Fluid Material") {

        const panel = gui.addFolder(name);

        panel.add(this as FluidV3Material, "splatForce", -1000, 1000);
        panel.add(this as FluidV3Material, "splatThickness", 0.001, 0.2);
        panel.add(this as FluidV3Material, "vorticityInfluence", 0.1, 1);
        panel.add(this as FluidV3Material, "swirlIntensity", 1, 100);
        panel.add(this as FluidV3Material, "pressure", 0, 1);
        panel.add(this as FluidV3Material, "velocityDissipation", 0, 1);
        panel.add(this as FluidV3Material, "densityDissipation", 0, 1);
        panel.add(this as FluidV3Material, "displacementScale", -.1, .1);
        panel.add(this as FluidV3Material, "pressureIterations", 1, 100, 1);
        panel.add({
            copySettings: () => {
                const settings: FluidV3Settings = {
                    splatForce: this.splatForce,
                    splatThickness: this.splatThickness,
                    vorticityInfluence: this.vorticityInfluence,
                    swirlIntensity: this.swirlIntensity,
                    pressure: this.pressure,
                    velocityDissipation: this.velocityDissipation,
                    densityDissipation: this.densityDissipation,
                    displacementScale: this.displacementScale,
                    pressureIterations: this.pressureIterations,
                };

                navigator.clipboard.writeText(JSON.stringify(settings, null, 2));
            },
        }, "copySettings");

        panel.add(this as FluidV3Material, "asSolid");
        panel.add(this as FluidV3Material, "asSmoke");
    }

    asSolid() {
        this.alphaMap = null;
        this.transparent = false;
        this.actAsSmoke = false;
    }

    asSmoke() {
        this.transparent = true;
        this.actAsSmoke = true;
    }

    /**
     * Restore values previously copied from the debug panel.
     * @see `addDebugPanelFolder`
     */
    setSettings(s: FluidV3Settings) {
        this.splatForce = s.splatForce;
        this.splatThickness = s.splatThickness;
        this.vorticityInfluence = s.vorticityInfluence;
        this.swirlIntensity = s.swirlIntensity;
        this.pressure = s.pressure;
        this.velocityDissipation = s.velocityDissipation;
        this.densityDissipation = s.densityDissipation;
        this.displacementScale = s.displacementScale;
        this.pressureIterations = s.pressureIterations;
    }
}
