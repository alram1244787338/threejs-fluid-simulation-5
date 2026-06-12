
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
import { NodeRepresentation, storage, abs, add, clamp, Continue, cross, distance, dot, Fn, If, instanceIndex, length, Loop, max, mix, modelNormalMatrix, mul, normalGeometry, normalize, positionLocal, smoothstep, texture, textureStore, uniform, uv, vec2, vec3, vec4, type ShaderNodeObject } from "three/tsl";
import { Color, ComputeNode, FloatType, Mesh, MeshPhysicalNodeMaterial, Node, Object3D, StorageBufferAttribute, StorageTexture, Texture, TextureNode, UniformNode, Vector2, WebGPURenderer, type ColorRepresentation } from "three/webgpu";
import { FluidSettingRanges, FluidSettings, FluidSimulation, FluidSimulationBackend, installSettingsAccessors } from "./fluid/FluidSimulation";

type Sampler2D = ShaderNodeObject<TextureNode>;
type NumberUniform = ShaderNodeObject<UniformNode<number>>;

const placeholderTexture = new Texture();
placeholderTexture.flipY = false;

const offsetSample = Fn<[Sampler2D, ShaderNodeObject<Node>, ShaderNodeObject<Node>, number, number]>(([sampler, uv, texel, x, y]) => {

    return sampler.sample(uv.add(texel.mul(vec2(x, y))));
});
//----------------------------------------------------
const encode = Fn<[ShaderNodeObject<Node>, ShaderNodeObject<Node>]>(([ _maxValue, value ])=>{
    return value; //value.div(maxValue).add(1).div(2);
});
const decode = Fn<[ShaderNodeObject<Node>, ShaderNodeObject<Node>]>(([ _maxValue, value ])=>{
    return value; //.mul(2).sub(1).mul(maxValue);
});


class ComputeShader {

    private textureToShader: Map<Texture, ShaderNodeObject<ComputeNode>>;

    constructor(private fn: (pixelPos: ShaderNodeObject<Node>, uvPos: ShaderNodeObject<Node>, texelSize: ShaderNodeObject<Node>) => NodeRepresentation) {
        this.textureToShader = new Map<Texture, ShaderNodeObject<ComputeNode>>()
    }

    private create(outTo: Texture, width: number, height: number) {

        return Fn(() => {

            const resolution = vec2(width, height);
            const posX = instanceIndex.mod(width);
            const posY = instanceIndex.div(width);
            const pixelPosition = vec2(posX, posY);
            const uvCoord = vec2(pixelPosition.add(vec2(0.5, 0.5))).div(resolution);
            const textelSize = vec2(1, 1).div(resolution);

            return textureStore(outTo, pixelPosition, this.fn(pixelPosition, uvCoord, textelSize)).toWriteOnly();

        })().compute(width * height)
    }

    createBinds(width: number, height: number, ...targets: Texture[]) {
        for (const target of targets)
            this.textureToShader.set(target, this.create(target, width, height));
        return this;
    }

    renderBind(renderer: WebGPURenderer, bindTarget: Texture) {
        if (!this.textureToShader.has(bindTarget)) {
            throw new Error("You are trying to render to a texture that this shader doesn't have. Did you forgot to call createBindTo?")
        }

        renderer.compute(this.textureToShader.get(bindTarget)!);

        return bindTarget;
    }

}

class ScrollShader extends ComputeShader {
    readonly uvScroll = uniform(new Vector2())
    constructor(uTarget: Sampler2D) {
        super((_pixelPos, uvPos) => {
            return uTarget.sample(uvPos.add(this.uvScroll))
        });
    }
}

class SplatShader extends ComputeShader {
    readonly splatVelocity = uniform(1);

    /**
     * % of the max speed at which we can move
     */
    readonly splatForce = uniform(-.1);
    readonly thickness = uniform(1);

    constructor(uTarget: Sampler2D, positionAttr: StorageBufferAttribute, colorAttr: StorageBufferAttribute, count: number, maxVelocity:NumberUniform ) {

        super((_pixelPos, vUv, textelSize) => {

            const pixel = uTarget.sample(vUv).toVar('pixel');

            Loop(count, ({ i }) => {

                const pos = storage(positionAttr, "vec4", count).element(i);
                const curr = pos.xy;
                const prev = pos.zw;
                const data = storage(colorAttr, "vec4", count).element(i);
                const color = data.rgb;
                const ratio = data.a.mul(this.thickness);

                const diff = curr.sub(prev);

                If(length(diff).equal(0.0), () => {

                    Continue();

                });

                const toFrag = vUv.sub(prev);
                const t = clamp(dot(toFrag, diff).div(dot(diff, diff)), 0, 1)
                const proj = prev.add(t.mul(diff));


                const d = distance(vUv, proj);

                If(d.lessThan(textelSize.x.mul(ratio)), () => {

                    const influence = smoothstep(ratio, 0.0, d);

                    If(this.splatVelocity, () => {

                        const vel = normalize(diff).mul(this.splatForce.negate());

                        // Diff is in UV units... the diference between the new and the old UV positions.
                        //const vel = diff.normalize().mul( clamp( diff.length(), maxVelocity.negate(), maxVelocity ) );

                        // vel will be a number between -1 and 1...
                        pixel.assign(vec4(pixel.r, encode(maxVelocity, vel), 0));

                    })
                        .Else(() => {

                            pixel.assign(mix(pixel, vec4(color, 1.0), influence));


                        });

                });


            });
            ;

            return pixel;
        });


    }
}

class CurlShader extends ComputeShader {
    readonly vorticityInfluence = uniform(1);

    constructor( uVelocity: Sampler2D, maxVelocity:NumberUniform ) {
        super((_, vUv, textelSize) => {
            const L = decode( maxVelocity, offsetSample(uVelocity, vUv, textelSize, -1, 0).b );
            const R = decode( maxVelocity, offsetSample(uVelocity, vUv, textelSize, 1, 0).b);
            const T = decode( maxVelocity, offsetSample(uVelocity, vUv, textelSize, 0, 1).g);
            const B = decode( maxVelocity, offsetSample(uVelocity, vUv, textelSize, 0, -1).g);

            const vorticity = R.sub(L).sub(T).add(B);
            const pixel = uVelocity.sample(vUv).toVar("pixel");
            const curl = this.vorticityInfluence.mul(vorticity);

            return vec4( pixel.xyz, encode( maxVelocity, curl ) );
        });
    }
}

class VorticityShader extends ComputeShader {
    readonly curl = uniform(2);
    readonly delta = uniform(0)

    constructor( uTarget: Sampler2D, maxVelocity:NumberUniform ) {
        super((_, vUv, textelSize) => {
            const L = decode( maxVelocity, offsetSample(uTarget, vUv, textelSize, -1, 0).a );
            const R = decode( maxVelocity, offsetSample(uTarget, vUv, textelSize, 1, 0).a );
            const T = decode( maxVelocity, offsetSample(uTarget, vUv, textelSize, 0, 1).a );
            const B = decode( maxVelocity, offsetSample(uTarget, vUv, textelSize, 0, -1).a );

            const pixel = uTarget.sample(vUv).toVar("pixel");
            const C = decode( maxVelocity, pixel.a );

            const force = mul(0.5, vec2(abs(T).sub(abs(B)), abs(R).sub(abs(L)))).toVar("force");

            force.divAssign(length(force).add(0.0001));
            force.mulAssign(this.curl.mul(C));
            force.mulAssign(vec2(0, -1.0));

            const velocity = decode( maxVelocity, pixel.gb.add(force.mul(this.delta)));

            return vec4( pixel.r, encode( maxVelocity, velocity ), 0);
        });
    }
}

class DivergenceShader extends ComputeShader {
    constructor(uVelocity: Sampler2D, maxVelocity:NumberUniform ) {
        super((_, vUv, textelSize) => {

            const L = decode( maxVelocity, offsetSample(uVelocity, vUv, textelSize, -1, 0).g).toVar("L");
            const R = decode( maxVelocity, offsetSample(uVelocity, vUv, textelSize, 1, 0).g).toVar("R");
            const T = decode( maxVelocity, offsetSample(uVelocity, vUv, textelSize, 0, 1).b).toVar("T");
            const B = decode( maxVelocity, offsetSample(uVelocity, vUv, textelSize, 0, -1).b).toVar("B");

            const pixel = uVelocity.sample(vUv);
            const C = decode( maxVelocity, pixel.gb ); // velocity info...

            If(vUv.x.sub(textelSize.x).lessThan(0), () => L.assign(C.x.negate()));
            If(vUv.x.add(textelSize.x).greaterThan(1), () => R.assign(C.x.negate()));
            If(vUv.y.add(textelSize.y).greaterThan(1), () => T.assign(C.y.negate()));
            If(vUv.y.sub(textelSize.y).lessThan(0), () => B.assign(C.y.negate()));

            const div = mul(0.5, R.sub(L).add(T.sub(B)));

            return vec4(pixel.r, pixel.gb, decode(2,div));
        });
    }
}

class ClearShader extends ComputeShader {
    readonly decay = uniform(0.317);

    constructor(uTarget: Sampler2D) {
        super((_, vUv) => {

            const pixel = uTarget.sample(vUv);
            return vec4(pixel.r.mul(this.decay), pixel.gba);

        });
    }
}

class PressureShader extends ComputeShader {
    constructor(uTarget: Sampler2D, maxVelocity:NumberUniform ) {
        super((_, vUv, textelSize) => {

            const L = decode( maxVelocity, offsetSample(uTarget, vUv, textelSize, -1, 0).x);
            const R = decode( maxVelocity, offsetSample(uTarget, vUv, textelSize, 1, 0).x);
            const T = decode( maxVelocity, offsetSample(uTarget, vUv, textelSize, 0, 1).x);
            const B = decode( maxVelocity, offsetSample(uTarget, vUv, textelSize, 0, -1).x);
            const pixel = uTarget.sample(vUv).toVar();

            const divergence = decode( maxVelocity, pixel.a );

            const pressure = mul(L.add(R).add(B).add(T).sub(divergence), .25);

            return vec4( encode( maxVelocity, pressure ), pixel.gba);

        });
    }
}

class GradientSubtractShader extends ComputeShader {
    constructor( uTarget: Sampler2D, maxVelocity:NumberUniform ) {
        super((_, vUv, textelSize) => {

            const L = decode( maxVelocity, offsetSample(uTarget, vUv, textelSize, -1, 0).x);
            const R = decode( maxVelocity, offsetSample(uTarget, vUv, textelSize, 1, 0).x);
            const T = decode( maxVelocity, offsetSample(uTarget, vUv, textelSize, 0, 1).x);
            const B = decode( maxVelocity, offsetSample(uTarget, vUv, textelSize, 0, -1).x);

            const pixel = uTarget.sample(vUv).toVar();
            const velocity = decode( maxVelocity, pixel.gb ).toVar("velocity");

            velocity.subAssign(vec2(R.sub(L), T.sub(B)));

            return vec4(pixel.r, encode( maxVelocity, velocity), 0.0);//
        });
    }
}

class AdvectShader extends ComputeShader {
    readonly sourceIsVelocity = uniform(0);
    readonly delta = uniform(0);
    readonly dissipation = uniform(0.1);
    readonly uSource: Sampler2D = texture(placeholderTexture);

    constructor( uVelocity: Sampler2D, maxVelocity:NumberUniform ) {
        super((_, vUv, _textelSize) => {

            const original = uVelocity.sample(vUv);
            const velocity = decode( maxVelocity, original.yz );
            const coord = vUv.sub(this.delta.mul(velocity)) ;
            const result = this.uSource.sample(coord).toVar("pixel");
            const decay = add(1.0, this.dissipation.mul(this.delta));
            result.divAssign(decay);

            If(this.sourceIsVelocity, () => {
                result.assign(vec4(
                    original.r,
                    result.gb,
                    original.w
                ));

            })

            return result;
        });
    }
}


type FluidMaterialSettings = {

    /**
     * Will use the color as source of emision doing pow to create accents
     */
    emitColor?: boolean

    /**
     * if true, it will have a transparent background
     */
    transparent?: boolean

    /**
     * max absolute speed in UV units
     */
    maxSpeed:number
}

/** Default settings for the WebGPU backend (units are UV-space velocities). */
const WEBGPU_DEFAULTS: FluidSettings = {
    splatForce: -0.1,
    splatThickness: 1,
    vorticityInfluence: 1,
    swirlIntensity: 2,
    pressure: 0.317,
    velocityDissipation: 0.283,
    densityDissipation: 0.2,
    displacementScale: 0.1,
    pressureIterations: 39,
};

/** Debug-panel slider ranges tuned for the WebGPU backend. */
const WEBGPU_RANGES: FluidSettingRanges = {
    splatForce: [-0.5, 0.5],
    splatThickness: [0.001, 1],
    vorticityInfluence: [0.1, 1],
    swirlIntensity: [1, 100],
    pressure: [0, 1],
    velocityDissipation: [0, 1],
    densityDissipation: [0, 1],
    displacementScale: [-1, 1],
    pressureIterations: [1, 100],
};

/* eslint-disable @typescript-eslint/no-empty-object-type, @typescript-eslint/no-unsafe-declaration-merging --
 * Canonical settings getters/setters (splatForce, pressure, ...) are installed at
 * runtime by installSettingsAccessors(); this empty interface merges their types
 * onto the class so callers get e.g. `material.splatForce` without each backend
 * re-declaring all nine accessors. */
export interface FluidMaterialGPU extends FluidSettings {}

/**
 * WebGPU fluid material. It is the WebGPU/TSL adapter for the shared
 * {@link FluidSimulation}: it owns the compute shaders, storage textures and
 * object buffers and implements {@link FluidSimulationBackend}, while the
 * simulation pipeline, object tracking, follow logic and settings all live in the
 * shared layer.
 */
export class FluidMaterialGPU extends MeshPhysicalNodeMaterial implements FluidSimulationBackend {

    /** Shared, backend-agnostic simulation pipeline. */
    private readonly simulation: FluidSimulation;

    private _bumpDisplacmentScale = uniform(0.1);

    private currentRT: Texture;
    private nextRT: Texture;

    private uTarget: Sampler2D;

    private dyeRT: Texture;
    private nextDyeRT: Texture;

    /** Per-object data buffer (FluidSimulationBackend contract). */
    readonly objectDataArray: Float32Array;
    /** Per-object positions buffer (FluidSimulationBackend contract). */
    readonly objectPositionArray: Float32Array;
    private objectDataAttribute: StorageBufferAttribute;
    private objectPositionAttribute: StorageBufferAttribute;

    private renderMaterial: (material: ComputeShader, target: Texture) => void;

    /**
     * Legacy alias for the canonical `displacementScale` setting. Kept so older
     * callers keep working; both names read/write the same shared value.
     */
    get bumpDisplacmentScale() {
        return this.simulation.settings.displacementScale;
    }

    set bumpDisplacmentScale(v: number) {
        this.simulation.settings.displacementScale = v;
    }

    /**
     * Color
     */
    get colorTexture() {
        return this.dyeRT;
    }

    /**
     * Idk why you would need this but maybe you'll find this useful since it contains the velocities of the surface...
     */
    get dataTexture() {
        return this.currentRT;
    }

    private scrollShader: ScrollShader;
    private splatShader: SplatShader;
    private curlShader: CurlShader;
    private vorticityShader: VorticityShader;
    private divergenceShader: DivergenceShader;
    private clearShader: ClearShader;
    private pressureShader: PressureShader;
    private gradientShader: GradientSubtractShader;
    private advectShader: AdvectShader;

    /**
     * Max speed at which the liquid can move inUV units.
     */
    private uMaxSpeed: ShaderNodeObject<UniformNode<number>>;

    constructor(renderer: WebGPURenderer, textureWidth: number, textureHeight: number, objectCount = 1, settings?: Partial<FluidMaterialSettings>) {
        super({
            roughness: 0.5,
            color: new Color(0xcccccc),
            transparent: true,
        });

        this.uMaxSpeed = uniform(settings?.maxSpeed ?? (1/10));

        const rt = () => {
            const txt = new StorageTexture(textureWidth, textureHeight);
            txt.type = FloatType;
            return txt;
        };

        this.currentRT = rt();
        this.nextRT = rt();
        this.dyeRT = rt();
        this.nextDyeRT = rt();



        this.objectPositionArray = new Float32Array(objectCount * 4);
        this.objectDataArray = new Float32Array(objectCount * 4);

        this.objectPositionAttribute = new StorageBufferAttribute(this.objectPositionArray, 4);
        this.objectDataAttribute = new StorageBufferAttribute(this.objectDataArray, 4);

        const texel = uniform(new Vector2(1 / textureWidth, 1 / textureHeight));


        this.uTarget = texture(placeholderTexture);

        const w = textureWidth;
        const h = textureHeight;
        const velA = this.currentRT;
        const velB = this.nextRT;
        const dyeA = this.dyeRT;
        const dyeB = this.nextDyeRT;


        this.scrollShader = new ScrollShader(this.uTarget).createBinds(w, h, velA, velB, dyeA, dyeB);

        this.splatShader = new SplatShader(   this.uTarget,
                                        this.objectPositionAttribute,
                                        this.objectDataAttribute,
                                        objectCount,
                                        this.uMaxSpeed ).createBinds(w, h, velA, velB, dyeA, dyeB);

        this.curlShader = new CurlShader( this.uTarget, this.uMaxSpeed ).createBinds(w, h, velA, velB);
        this.vorticityShader = new VorticityShader(this.uTarget, this.uMaxSpeed ).createBinds(w, h, velA, velB);
        this.divergenceShader = new DivergenceShader(this.uTarget, this.uMaxSpeed ).createBinds(w, h, velA, velB);
        this.clearShader = new ClearShader(this.uTarget).createBinds(w, h, velA, velB);
        this.pressureShader = new PressureShader(this.uTarget, this.uMaxSpeed).createBinds(w, h, velA, velB);
        this.gradientShader = new GradientSubtractShader(this.uTarget, this.uMaxSpeed).createBinds(w, h, velA, velB);
        this.advectShader = new AdvectShader(this.uTarget, this.uMaxSpeed).createBinds(w, h, velA, velB, dyeA, dyeB);


        this.renderMaterial = (material, target) => {
            material.renderBind(renderer, target);
        }
        //#endregion
        //--------------------------------------------------------------------------------------------------------------------------------------


        /////// displacement
        const maxChannel = this.uTarget.sample(uv());
        const maxValue = max(maxChannel.r.clamp(0, 1), max(maxChannel.g.clamp(0, 1), maxChannel.b.clamp(0, 1)));

        this.positionNode = positionLocal.add(normalGeometry.mul(maxValue.mul(this._bumpDisplacmentScale)));

        // alpha
        if (settings?.transparent) {
            this.opacityNode = maxValue;
        }

        this.colorNode = this.uTarget ;


        ///// fix shading...
        const height = Fn<[ShaderNodeObject<Node>]>(([uvOffset]) =>
            dot(this.uTarget.sample(uv().add(uvOffset)).rgb, vec3(0.299, 0.587, 0.114)));

        this.normalNode = Fn(() => {
            const scale = this._bumpDisplacmentScale;

            const hL = height(vec2(texel.x.negate(), 0.0));
            const hR = height(vec2(texel.x, 0.0));
            const hD = height(vec2(0.0, texel.y.negate()));
            const hU = height(vec2(0.0, texel.y));

            const dx = vec3(texel.x.mul(2), (hR.sub(hL)).mul(scale), 0.0);
            const dy = vec3(0.0, (hU.sub(hD)).mul(scale), texel.y.mul(2));

            const normal = normalize(cross(dy, dx));
            return normalize(modelNormalMatrix.mul(normal));
        })();

        if (settings?.emitColor) {
            this.emissiveNode = maxChannel.pow(3);
        }

        // shared pipeline + tracking + settings; `this` is the WebGPU backend.
        this.simulation = new FluidSimulation(this, objectCount, WEBGPU_DEFAULTS);
        installSettingsAccessors(this, this.simulation);
    }

    // ---------------------------------------------------------------------------
    // Public, backend-agnostic API (delegated to the shared simulation)
    // ---------------------------------------------------------------------------

    /** @see FluidSimulation.follow */
    get follow() { return this.simulation.follow }
    set follow(obj: Object3D | undefined) { this.simulation.follow = obj }

    /** @see FluidSimulation.track */
    track(object: Object3D, ratio = 1, color: ColorRepresentation = Color.NAMES.black) {
        this.simulation.track(object, ratio, color);
    }

    /** @see FluidSimulation.untrack */
    untrack(object: Object3D) {
        this.simulation.untrack(object);
    }

    update(delta: number, mesh: Mesh) {
        this.simulation.update(delta, mesh);
    }

    /** @see FluidSimulation.setSettings */
    setSettings(s: Partial<FluidSettings> & { pressureDecay?: number; bumpDisplacmentScale?: number }) {
        this.simulation.setSettings(s);
    }

    addDebugPanelFolder(gui: GUI, name = "Fluid Material") {
        return this.simulation.addDebugPanelFolder(gui, WEBGPU_RANGES, name);
    }

    // ---------------------------------------------------------------------------
    // FluidSimulationBackend implementation (WebGPU / TSL specifics)
    // ---------------------------------------------------------------------------

    markObjectDataDirty(): void {
        this.objectDataAttribute.needsUpdate = true;
    }

    markObjectPositionDirty(): void {
        this.objectPositionAttribute.needsUpdate = true;
    }

    applySettings(s: FluidSettings): void {
        this.splatShader.splatForce.value = s.splatForce;
        this.splatShader.thickness.value = s.splatThickness;
        this.curlShader.vorticityInfluence.value = s.vorticityInfluence;
        this.vorticityShader.curl.value = s.swirlIntensity;
        this.clearShader.decay.value = s.pressure;
        this._bumpDisplacmentScale.value = s.displacementScale;
    }

    scroll(uvStep: Vector2): void {
        this.scrollShader.uvScroll.value = uvStep;

        this.uTarget.value = this.currentRT;
        this.blit(this.scrollShader);

        this.uTarget.value = this.dyeRT;
        this.blitDye(this.scrollShader);
    }

    splatVelocity(): void {
        this.uTarget.value = this.currentRT;
        this.splatShader.splatVelocity.value = 1;
        this.blit(this.splatShader);
    }

    splatColor(): void {
        this.uTarget.value = this.dyeRT;
        this.splatShader.splatVelocity.value = 0;
        this.blitDye(this.splatShader);
    }

    curl(): void {
        this.uTarget.value = this.currentRT;
        this.blit(this.curlShader);
    }

    vorticity(delta: number): void {
        this.uTarget.value = this.currentRT;
        this.vorticityShader.delta.value = delta;
        this.blit(this.vorticityShader);
    }

    divergence(): void {
        this.uTarget.value = this.currentRT;
        this.blit(this.divergenceShader);
    }

    clearPressure(): void {
        this.uTarget.value = this.currentRT;
        this.blit(this.clearShader);
    }

    pressureStep(): void {
        this.uTarget.value = this.currentRT;
        this.blit(this.pressureShader);
    }

    gradientSubtract(): void {
        this.uTarget.value = this.currentRT;
        this.blit(this.gradientShader);
    }

    advectVelocity(delta: number, dissipation: number): void {
        this.uTarget.value = this.currentRT;
        this.advectShader.delta.value = delta;
        this.advectShader.uSource.value = this.currentRT;
        this.advectShader.sourceIsVelocity.value = 1;
        this.advectShader.dissipation.value = dissipation;
        this.blit(this.advectShader);
    }

    advectColor(delta: number, dissipation: number): void {
        // advect reads the velocity field from uTarget, so it must point at the
        // velocity texture even though the dye is the source being advected.
        this.uTarget.value = this.currentRT;
        this.advectShader.delta.value = delta;
        this.advectShader.uSource.value = this.dyeRT;
        this.advectShader.sourceIsVelocity.value = 0;
        this.advectShader.dissipation.value = dissipation;
        this.blitDye(this.advectShader);
    }

    present(): void {
        this.uTarget.value = this.dyeRT;
    }

    /**
     * Renders the material into the next render texture and then swaps them so the new currentRT is the one that was generated by the material.
     */
    private blit(material: ComputeShader) {
        this.renderMaterial(material, this.nextRT);
        //swap
        [this.currentRT, this.nextRT] = [this.nextRT, this.currentRT];

        this.uTarget.value = this.currentRT;

    }

    private blitDye(material: ComputeShader) {
        this.renderMaterial(material, this.nextDyeRT);
        //swap
        [this.dyeRT, this.nextDyeRT] = [this.nextDyeRT, this.dyeRT];

        this.uTarget.value = this.currentRT;

    }
}
