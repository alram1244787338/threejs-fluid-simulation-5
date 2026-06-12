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
import { Color, ColorRepresentation, DataTexture, FloatType, Mesh, MeshPhysicalMaterial, Object3D, RGBAFormat, ShaderMaterial, Vector2, WebGLRenderer, WebGLRenderTarget, type WebGLProgramParametersWithUniforms } from "three";
import { FullScreenQuad } from "three/examples/jsm/Addons.js";
import { FluidSettingRanges, FluidSettings, FluidSimulation, FluidSimulationBackend, installSettingsAccessors } from "./fluid/FluidSimulation";

/**
 * R - Pressure
 * G - X dir
 * B - Y dir
 * A - wildcard, used to pass values from shader to shader. Not persisted.
 */

const vertexShader = `
                varying vec2 vUv;
                varying vec2 vL;
                varying vec2 vR;
                varying vec2 vT;
                varying vec2 vB;
                uniform vec2 texelSize;

                void main() {
                    vUv = uv;

                    vL = uv - vec2(texelSize.x, 0.0);
                    vR = uv + vec2(texelSize.x, 0.0);
                    vT = uv + vec2(0.0, texelSize.y);
                    vB = uv - vec2(0.0, texelSize.y);

                    gl_Position = vec4(position, 1.0);
                }
            `;

class ScrollShader extends ShaderMaterial {
    constructor( texelSize:Vector2 ) {
        super({
            uniforms: {
                uTarget: { value: null },
                texelSize: { value: texelSize },
                uvScroll: { value: null },
            },
            vertexShader,
            fragmentShader: `
                precision mediump float;
                varying highp vec2 vUv;
                uniform sampler2D uTarget;
                uniform vec2 uvScroll;

                void main() {
                    gl_FragColor = texture2D(uTarget, vUv + uvScroll);
                }
            `
        })
    }
}

/**
 * Introduces either velocity or color into target. Depending on `splatVelocity` flag.
 */
class SplatShader extends ShaderMaterial {
    constructor( texelSize:Vector2, objectCount:number, aspectRatio:number ) {
        super({
            uniforms: {
                uTarget: { value: null },
                splatVelocity: { value:false },
                color: { value: new Color(0xffffff) },
                texelSize: { value: texelSize },
                objectData: { value: null }, // Contains current and previous object positions
                objectPosition: { value: null }, // Contains current and previous object positions
                count: { value: objectCount },
                thickness: { value: 1 }, // in UV units
                aspectRatio: { value:aspectRatio } // in UV units
                , splatForce: { value: -196 }
            },

            vertexShader,
            fragmentShader:`
                precision mediump float;
                precision mediump sampler2D;

                varying highp vec2 vUv;
                uniform sampler2D uTarget;
                uniform sampler2D objectData; //color + ratio
                uniform sampler2D objectPosition; // current + old uv positions
                uniform int count;
                uniform float thickness; //TODO: this shold be individual per object to allow diferent types of bodies affecting the liquid
                uniform float aspectRatio;
                uniform highp vec2 texelSize;
                uniform bool splatVelocity;
                uniform vec3 color;
                uniform float splatForce;

                void main () {

                    vec4 pixel = texture2D(uTarget, vUv);

                    // Add External Forces (from objects)
                    // IMPROVEMENT: This loop is much more efficient as it reads from a texture.
                    for (int i = 0; i < count; i++) {
                        // Read object data from the texture.
                        // texelFetch is used for direct, un-interpolated pixel reads.
                        vec4 data = texelFetch(objectData, ivec2(i, 0), 0);
                        vec4 positions = texelFetch(objectPosition, ivec2(i, 0), 0);
                        vec2 curr = positions.xy; // Current position in .xy
                        vec2 prev = positions.zw; // Previous position in .zw
                        float ratio = data.a * thickness;

                        vec2 diff = curr - prev;
                        if (length(diff) == 0.0) continue; // Skip if the object hasn't moved

                        vec2 toFrag = vUv - prev;
                        float t = clamp(dot(toFrag, diff) / dot(diff, diff), 0.0, 1.0);
                        vec2 proj = prev + t * diff;

                        vec2 aspect = vec2(aspectRatio, 1.0);

                        // Calculate distance in a way that respects the screen's aspect ratio
                        float d = distance(vUv * aspect, proj * aspect);

                        if (d < ratio) {
                            // IMPROVEMENT: Correct influence logic.
                            // Influence is strongest when distance 'd' is 0.
                            float influence = smoothstep(ratio, 0.0, d);

                            if( splatVelocity )
                            {

                                vec2 vel = normalize( ( diff )/texelSize ) * -splatForce;


                                //vel = mix( pixel.gb, vel, influence );

                                pixel.g = vel.x;
                                pixel.b = vel.y;
                            }
                            else
                            {

                                pixel = mix( pixel, vec4( data.rgb, 1.0 ), influence );
                            }

                        }
                    }

                    gl_FragColor = pixel;
                }
            `
        })
    }
}


/**
 * sets vorticity inthe alpha channel of uVelocity image
 */
class CurlShader extends ShaderMaterial {
    constructor( texelSize:Vector2 ) {
        super({
            uniforms: {
                uVelocity: { value: null },
                texelSize: { value: texelSize },
                vorticityInfluence: { value:1 }
            },
            vertexShader,
            fragmentShader:`
                precision mediump float;
                precision mediump sampler2D;

                varying highp vec2 vUv;
                varying highp vec2 vL;
                varying highp vec2 vR;
                varying highp vec2 vT;
                varying highp vec2 vB;
                uniform sampler2D uVelocity;
                uniform float vorticityInfluence;

                void main () {
                    float L = texture2D(uVelocity, vL).b;
                    float R = texture2D(uVelocity, vR).b;
                    float T = texture2D(uVelocity, vT).g;
                    float B = texture2D(uVelocity, vB).g;
                    float vorticity = R - L - T + B;

                    vec4 pixel = texture2D(uVelocity, vUv);

                    pixel.a = vorticityInfluence * vorticity; // set in the 4th component...

                    gl_FragColor = pixel;
                }
            `
        })
    }
}

/**
 * updates the velocity image
 */
class VorticityShader extends ShaderMaterial {
    constructor( texelSize:Vector2 ) {
        super({
            uniforms: {
                uVelocityAndCurl: { value: null },
                texelSize: { value: texelSize },
                curl: { value: 1 },
                dt: { value: 0 },
            },
            vertexShader,
            fragmentShader:`
                precision highp float;
                precision highp sampler2D;

                varying vec2 vUv;
                varying vec2 vL;
                varying vec2 vR;
                varying vec2 vT;
                varying vec2 vB;
                uniform sampler2D uVelocityAndCurl;
                uniform float curl;
                uniform float dt;

                void main () {
                    float L = texture2D(uVelocityAndCurl, vL).a;
                    float R = texture2D(uVelocityAndCurl, vR).a;
                    float T = texture2D(uVelocityAndCurl, vT).a;
                    float B = texture2D(uVelocityAndCurl, vB).a;
                    float C = texture2D(uVelocityAndCurl, vUv).a;

                    vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
                    force /= length(force) + 0.0001;
                    force *= curl * C;
                    force.y *= -1.0;

                    vec4 pixel = texture2D(uVelocityAndCurl, vUv);

                    vec2 velocity = pixel.gb;
                    velocity += force * dt;
                    velocity = min(max(velocity, -1000.0), 1000.0);

                    gl_FragColor = vec4( pixel.r, velocity, 0.0 );
                }
            `
        })
    }
}

/**
 * Adds divergence in the alpha channel of the velocity image
 */
class DivergenceShader extends ShaderMaterial {
    constructor( texelSize:Vector2 ) {
        super({
            uniforms: {
                uVelocity: { value: null },
                texelSize: { value: texelSize },
            },
            vertexShader,
            fragmentShader:`
                precision mediump float;
                precision mediump sampler2D;

                varying highp vec2 vUv;
                varying highp vec2 vL;
                varying highp vec2 vR;
                varying highp vec2 vT;
                varying highp vec2 vB;
                uniform sampler2D uVelocity;

                void main () {
                    float L = texture2D(uVelocity, vL).g;
                    float R = texture2D(uVelocity, vR).g;
                    float T = texture2D(uVelocity, vT).b;
                    float B = texture2D(uVelocity, vB).b;

                    vec4 pixel = texture2D(uVelocity, vUv);

                    vec2 C = pixel.gb;
                    if (vL.x < 0.0) { L = -C.x; }
                    if (vR.x > 1.0) { R = -C.x; }
                    if (vT.y > 1.0) { T = -C.y; }
                    if (vB.y < 0.0) { B = -C.y; }

                    float div = 0.5 * (R - L + T - B);

                    gl_FragColor = vec4( pixel.r, C, div );
                }
            `
        })
    }
}

/**
 *  Multiplies the pressure by `value` uniform
 */
class ClearShader extends ShaderMaterial {
    constructor( texelSize:Vector2 ) {
        super({
            uniforms: {
                uTexture: { value: null },
                value: { value: 0.317 }, //PRESSURE
                texelSize: { value: texelSize },
            },
            vertexShader,
            fragmentShader:`
                precision mediump float;
                precision mediump sampler2D;

                varying highp vec2 vUv;
                uniform sampler2D uTexture;
                uniform float value;

                void main () {
                    vec4 pixel = texture2D(uTexture, vUv);

                    pixel.r *= value;

                    gl_FragColor = pixel ;
                }
            `
        })
    }
}

/**
 * updates the pressure of the image
 */
class PressureShader extends ShaderMaterial {
    constructor( texelSize:Vector2 ) {
        super({
            uniforms: {
                uPressureWithDivergence: { value: null },
                texelSize: { value: texelSize },
            },
            vertexShader,
            fragmentShader:`
                precision mediump float;
                precision mediump sampler2D;

                varying highp vec2 vUv;
                varying highp vec2 vL;
                varying highp vec2 vR;
                varying highp vec2 vT;
                varying highp vec2 vB;
                uniform sampler2D uPressureWithDivergence;

                void main () {
                    float L = texture2D(uPressureWithDivergence, vL).x;
                    float R = texture2D(uPressureWithDivergence, vR).x;
                    float T = texture2D(uPressureWithDivergence, vT).x;
                    float B = texture2D(uPressureWithDivergence, vB).x;
                    float C = texture2D(uPressureWithDivergence, vUv).x;

                    vec4 pixel = texture2D(uPressureWithDivergence, vUv);
                    float divergence = pixel.a;
                    float pressure = (L + R + B + T - divergence) * 0.25;

                    pixel.x = pressure;

                    gl_FragColor = pixel;
                }
            `
        })
    }
}


class GradientSubtractShader extends ShaderMaterial {
    constructor( texelSize:Vector2 ) {
        super({
            uniforms: {
                uPressureWithVelocity: { value: null },
                texelSize: { value: texelSize },
            },
            vertexShader,
            fragmentShader:`
                precision mediump float;
                precision mediump sampler2D;

                varying highp vec2 vUv;
                varying highp vec2 vL;
                varying highp vec2 vR;
                varying highp vec2 vT;
                varying highp vec2 vB;
                uniform sampler2D uPressureWithVelocity;

                void main () {
                    float L = texture2D(uPressureWithVelocity, vL).x;
                    float R = texture2D(uPressureWithVelocity, vR).x;
                    float T = texture2D(uPressureWithVelocity, vT).x;
                    float B = texture2D(uPressureWithVelocity, vB).x;

                    vec4 pixel = texture2D(uPressureWithVelocity, vUv);

                    vec2 velocity = pixel.gb;
                    velocity.xy -= vec2(R - L, T - B);

                    gl_FragColor = vec4( pixel.r, velocity, 0.0 );
                }
            `
        })
    }
}


class AdvectVelocityShader extends ShaderMaterial {
    constructor( texelSize:Vector2, dyeTexelSize:Vector2, manualFiltering = false ) {
        super({
            uniforms: {
                uVelocity: { value: null },
                uSource: { value: null },
                sourceIsVelocity: { value: null },
                texelSize: { value: texelSize },
                dt: { value: 0 },
                dyeTexelSize: { value: dyeTexelSize },
                dissipation: { value: 0.2 },
            },
            defines: {
                MANUAL_FILTERING: manualFiltering
            },
            vertexShader,
            fragmentShader:`
                precision highp float;
                precision highp sampler2D;

                varying vec2 vUv;
                uniform sampler2D uVelocity;
                uniform sampler2D uSource;
                uniform vec2 texelSize;
                uniform vec2 dyeTexelSize;
                uniform float dt;
                uniform float dissipation;
                uniform bool sourceIsVelocity;

                vec4 bilerp (sampler2D sam, vec2 uv, vec2 tsize) {
                    vec2 st = uv / tsize - 0.5;

                    vec2 iuv = floor(st);
                    vec2 fuv = fract(st);

                    vec4 a = texture2D(sam, (iuv + vec2(0.5, 0.5)) * tsize);
                    vec4 b = texture2D(sam, (iuv + vec2(1.5, 0.5)) * tsize);
                    vec4 c = texture2D(sam, (iuv + vec2(0.5, 1.5)) * tsize);
                    vec4 d = texture2D(sam, (iuv + vec2(1.5, 1.5)) * tsize);

                    return mix(mix(a, b, fuv.x), mix(c, d, fuv.x), fuv.y);
                }

                void main () {

                    #ifdef MANUAL_FILTERING
                        vec2 coord = vUv - dt * bilerp(uVelocity, vUv, texelSize).gb * texelSize;
                        vec4 result = bilerp(uSource, coord, dyeTexelSize);
                    #else
                        vec2 coord = vUv - dt * texture2D(uVelocity, vUv).gb * texelSize;
                        vec4 result = texture2D(uSource, coord);
                    #endif
                        float decay = 1.0 + dissipation * dt;
                        result /= decay;

                        if( sourceIsVelocity )
                        {
                            vec4 data = texture2D(uVelocity, vUv);
                            gl_FragColor = vec4( data.r, result.g, result.b, data.a);
                        }
                        else
                        {
                            gl_FragColor = result;
                        }
                }
            `
        })
    }
}

/** Default settings for the WebGL backend (units are pixel-space velocities). */
const WEBGL_DEFAULTS: FluidSettings = {
    splatForce: -196,
    splatThickness: 1,
    vorticityInfluence: 1,
    swirlIntensity: 1,
    pressure: 0.317,
    velocityDissipation: 0.283,
    densityDissipation: 0.138,
    displacementScale: 0.0078,
    pressureIterations: 39,
};

/** Debug-panel slider ranges tuned for the WebGL backend. */
const WEBGL_RANGES: FluidSettingRanges = {
    splatForce: [-1000, 1000],
    splatThickness: [0.001, 0.2],
    vorticityInfluence: [0.1, 1],
    swirlIntensity: [1, 100],
    pressure: [0, 1],
    velocityDissipation: [0, 1],
    densityDissipation: [0, 1],
    displacementScale: [-0.1, 0.1],
    pressureIterations: [1, 100],
};

/* eslint-disable @typescript-eslint/no-empty-object-type, @typescript-eslint/no-unsafe-declaration-merging --
 * Canonical settings getters/setters (splatForce, pressure, ...) are installed at
 * runtime by installSettingsAccessors(); this empty interface merges their types
 * onto the class so callers get e.g. `material.splatForce` without each backend
 * re-declaring all nine accessors. */
export interface FluidV3Material extends FluidSettings {}

/**
 * WebGL fluid material. It is the WebGL adapter for the shared
 * {@link FluidSimulation}: it owns the GLSL ping-pong render targets and shaders
 * and implements {@link FluidSimulationBackend}, while the simulation pipeline,
 * object tracking, follow logic and settings all live in the shared layer.
 */
export class FluidV3Material extends MeshPhysicalMaterial implements FluidSimulationBackend {

    /** Shared, backend-agnostic simulation pipeline. */
    private readonly simulation: FluidSimulation;

    private currentRT:WebGLRenderTarget;
    private nextRT:WebGLRenderTarget;

    // the "color" + elevation (the alpha...)
    private dyeRT:WebGLRenderTarget;
    private nextDyeRT:WebGLRenderTarget;

    /**
     * Color
     */
    get colorTexture() {
         return this.dyeRT.texture;
    }

    /**
     * Idk why you would need this but maybe you'll find this useful since it contains the velocities of the surface...
     */
    get dataTexture() {
        return this.currentRT.texture;
    }

    private quad:FullScreenQuad;

    private objectPositionTexture:DataTexture; // R
    /** Per-object positions buffer (FluidSimulationBackend contract). */
    readonly objectPositionArray:Float32Array;

    private objectDataTexture:DataTexture; // R
    /** Per-object data buffer (FluidSimulationBackend contract). */
    readonly objectDataArray:Float32Array;

    // shaders involved in the simulation
    private scrollShader:ScrollShader;
    private splatShader:SplatShader;
    private curlShader:CurlShader;
    private vorticityShader:VorticityShader;
    private divergenceShader:DivergenceShader;
    private clearShader:ClearShader;
    private pressureShader:PressureShader;
    private gradientShader:GradientSubtractShader;
    private advectionShader:AdvectVelocityShader;
    private supportLinearFiltering:boolean;

    /**
     * If `true` on every update the `alphaMap` will be set to the `colorMap`
     */
    private actAsSmoke = false;

    constructor( private renderer:WebGLRenderer, textureWidth:number, textureHeight:number, objectCount=1 )
    {
        const aspect = textureWidth / textureHeight;

        super({
            roughness: 1,
            color: new Color( 0xffffff ),
            displacementScale:0.0078,
            transparent:true
        });

        // ping pong render textures...
        this.currentRT = new WebGLRenderTarget(textureWidth, textureHeight, { type: FloatType });
        this.nextRT = new WebGLRenderTarget(textureWidth, textureHeight, { type: FloatType });


        // color textures.
        this.dyeRT = new WebGLRenderTarget(textureWidth, textureHeight, { type: FloatType });
        this.nextDyeRT = new WebGLRenderTarget(textureWidth, textureHeight, { type: FloatType });


        // 2. Create a Float32Array to hold the data

        this.objectDataArray = new Float32Array(objectCount * 4);// color + ratio: R, G, B, ratio
        this.objectPositionArray = new Float32Array(objectCount * 4);// current + old UV positions: current.x, current.y, prev.x, prev.y

        // 3. Create the DataTexture
        this.objectDataTexture = new DataTexture(
            this.objectDataArray,
            objectCount, // width
            1,           // height
            RGBAFormat,
            FloatType
        );

        this.objectPositionTexture = new DataTexture(
            this.objectPositionArray,
            objectCount, // width
            1,           // height
            RGBAFormat,
            FloatType
        );

        this.quad = new FullScreenQuad();

        const texel = new Vector2( 1/textureWidth, 1/textureHeight );

        // ----- shaders used to simulate the liquid -----

        const gl = renderer.getContext();
        this.supportLinearFiltering = !!gl.getExtension('OES_texture_half_float_linear');

        this.scrollShader = new ScrollShader( texel );
        this.splatShader = new SplatShader( texel, objectCount, aspect );
        this.curlShader = new CurlShader(texel);
        this.vorticityShader = new VorticityShader( texel );
        this.divergenceShader = new DivergenceShader( texel );
        this.clearShader = new ClearShader( texel );
        this.pressureShader = new PressureShader(texel);
        this.gradientShader = new GradientSubtractShader(texel);
        this.advectionShader = new AdvectVelocityShader(texel, texel, this.supportLinearFiltering? false : true );

        // shared pipeline + tracking + settings; `this` is the WebGL backend.
        this.simulation = new FluidSimulation(this, objectCount, WEBGL_DEFAULTS);
        installSettingsAccessors(this, this.simulation);
    }

    // ---------------------------------------------------------------------------
    // Public, backend-agnostic API (delegated to the shared simulation)
    // ---------------------------------------------------------------------------

    /** @see FluidSimulation.follow */
    get follow() { return this.simulation.follow }
    set follow( obj:Object3D|undefined ) { this.simulation.follow = obj }

    /** @see FluidSimulation.track */
    track( object:Object3D, ratio = 1, color:ColorRepresentation = Color.NAMES.black ) {
        this.simulation.track( object, ratio, color );
    }

    /** @see FluidSimulation.untrack */
    untrack( object:Object3D ) {
        this.simulation.untrack( object );
    }

    /**
     * @param delta time step in seconds
     * @param mesh The plane mesh used to simulate the liquid.
     */
    update( delta:number, mesh:Mesh ) {
        this.simulation.update( delta, mesh );
    }

    /** @see FluidSimulation.setSettings */
    setSettings( s:Partial<FluidSettings> & { pressureDecay?:number; bumpDisplacmentScale?:number } ) {
        this.simulation.setSettings( s );
    }

    addDebugPanelFolder( gui:GUI, name = "Fluid Material" ) {
        const panel = this.simulation.addDebugPanelFolder( gui, WEBGL_RANGES, name );
        panel.add(this as { asSolid(): void; asSmoke(): void }, "asSolid");
        panel.add(this as { asSolid(): void; asSmoke(): void }, "asSmoke");
        return panel;
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

    // ---------------------------------------------------------------------------
    // FluidSimulationBackend implementation (WebGL / GLSL specifics)
    // ---------------------------------------------------------------------------

    markObjectDataDirty(): void {
        this.objectDataTexture.needsUpdate = true;
    }

    markObjectPositionDirty(): void {
        this.objectPositionTexture.needsUpdate = true;
    }

    applySettings(s: FluidSettings): void {
        this.splatShader.uniforms.splatForce.value = s.splatForce;
        this.splatShader.uniforms.thickness.value = s.splatThickness;
        this.curlShader.uniforms.vorticityInfluence.value = s.vorticityInfluence;
        this.vorticityShader.uniforms.curl.value = s.swirlIntensity;
        this.clearShader.uniforms.value.value = s.pressure;
        // `displacementScale` is read directly off this material by three.js (the
        // installed accessor returns settings.displacementScale), so there is
        // nothing extra to push here.
    }

    scroll(uvStep: Vector2): void {
        this.scrollShader.uniforms.uvScroll.value = uvStep;

        this.scrollShader.uniforms.uTarget.value = this.currentRT.texture;
        this.blit(this.scrollShader);

        this.scrollShader.uniforms.uTarget.value = this.dyeRT.texture;
        this.blitDye(this.scrollShader);
    }

    splatVelocity(): void {
        this.splatShader.uniforms.objectData.value = this.objectDataTexture;
        this.splatShader.uniforms.objectPosition.value = this.objectPositionTexture;
        this.splatShader.uniforms.uTarget.value = this.currentRT.texture;
        this.splatShader.uniforms.splatVelocity.value = true;
        this.blit(this.splatShader);
    }

    splatColor(): void {
        this.splatShader.uniforms.objectData.value = this.objectDataTexture;
        this.splatShader.uniforms.objectPosition.value = this.objectPositionTexture;
        this.splatShader.uniforms.uTarget.value = this.dyeRT.texture;
        this.splatShader.uniforms.splatVelocity.value = false;
        this.blitDye(this.splatShader);
    }

    curl(): void {
        this.curlShader.uniforms.uVelocity.value = this.currentRT.texture;
        this.blit(this.curlShader);
    }

    vorticity(delta: number): void {
        this.vorticityShader.uniforms.uVelocityAndCurl.value = this.currentRT.texture;
        this.vorticityShader.uniforms.dt.value = delta;
        this.blit(this.vorticityShader);
    }

    divergence(): void {
        this.divergenceShader.uniforms.uVelocity.value = this.currentRT.texture;
        this.blit(this.divergenceShader);
    }

    clearPressure(): void {
        this.clearShader.uniforms.uTexture.value = this.currentRT.texture;
        this.blit(this.clearShader);
    }

    pressureStep(): void {
        this.pressureShader.uniforms.uPressureWithDivergence.value = this.currentRT.texture;
        this.blit(this.pressureShader);
    }

    gradientSubtract(): void {
        this.gradientShader.uniforms.uPressureWithVelocity.value = this.currentRT.texture;
        this.blit(this.gradientShader);
    }

    advectVelocity(delta: number, dissipation: number): void {
        this.advectionShader.uniforms.dt.value = delta;
        this.advectionShader.uniforms.uVelocity.value = this.currentRT.texture;
        this.advectionShader.uniforms.uSource.value = this.currentRT.texture;
        this.advectionShader.uniforms.sourceIsVelocity.value = true;
        this.advectionShader.uniforms.dissipation.value = dissipation;
        this.blit(this.advectionShader);
    }

    advectColor(delta: number, dissipation: number): void {
        this.advectionShader.uniforms.dt.value = delta;
        this.advectionShader.uniforms.uVelocity.value = this.currentRT.texture;
        this.advectionShader.uniforms.uSource.value = this.dyeRT.texture;
        this.advectionShader.uniforms.sourceIsVelocity.value = false;
        this.advectionShader.uniforms.dissipation.value = dissipation;
        this.blitDye(this.advectionShader);
    }

    present(): void {
        this.renderer.setRenderTarget(null);

        this.displacementMap = this.dyeRT.texture;

        if (this.actAsSmoke) {
            this.alphaMap = this.dyeRT.texture;
        }
        this.map = this.dyeRT.texture;
    }

    /**
     * Make normals respect the displacement...
     */
    override onBeforeCompile( shader: WebGLProgramParametersWithUniforms ): void {
         // Pass UV and world position to fragment shader
            shader.vertexShader = shader.vertexShader
                .replace(
                    '#include <common>',
                    `#include <common>
                    varying vec2 vUv;
                    varying vec3 vWorldPos;`
                )
                .replace(
                    '#include <uv_vertex>',
                    `#include <uv_vertex>
                    vUv = uv;`
                )
                .replace(
                    '#include <project_vertex>',
                    `#include <project_vertex>
                    vWorldPos = position; // (modelMatrix * vec4(position, 1.0)).xyz;`
                ).replace(
                    '#include <displacementmap_vertex>',
                    `
                    #ifdef USE_DISPLACEMENTMAP

                        vec3 dispColor = texture2D( displacementMap, vUv ).rgb;
                        float displacement = max( max(dispColor.r, dispColor.g),  dispColor.b );

                        transformed += normalize( objectNormal ) * ( displacement * displacementScale + displacementBias );

                    #endif
                `
    );

            // Displace in fragment and recompute normals from that
            shader.fragmentShader = shader.fragmentShader
                .replace(
                '#include <common>',
                `#include <common>
                uniform sampler2D displacementMap;
                uniform float displacementScale;
                uniform mat3 normalMatrix;
                varying vec2 vUv;
                varying vec3 vWorldPos;`
                )
                .replace(
                '#include <normal_fragment_begin>',
                `
                    vec3 color = texture2D(displacementMap, vUv).rgb;
                    float luminance = dot(color, vec3(0.299, 0.587, 0.114));

                    float d = luminance-0.5; //texture2D(displacementMap, vUv).r - 0.5;
                    vec3 displacedWorld = vWorldPos + vec3(0.0, d * displacementScale, 0.0);

                    vec3 dx = dFdx(displacedWorld);
                    vec3 dy = dFdy(displacedWorld);
                    vec3 displacedNormal = normalize(cross(dx, dy));

                    vec3 normalView = normalize(normalMatrix * displacedNormal);
                    vec3 normal = normalView;
                    vec3 nonPerturbedNormal = normalView;
                `
                );
    }

    /**
     * Renders the material into the next render texture and then swaps them so the new currentRT is the one that was generated by the material.
     */
    private blit( material:ShaderMaterial )
    {
        this.renderer.setRenderTarget( this.nextRT );
        this.quad.material = material;
        this.quad.render(this.renderer);

        //swap
        [this.currentRT, this.nextRT] = [this.nextRT, this.currentRT];
    }

    private blitDye( material:ShaderMaterial ) {
        this.renderer.setRenderTarget( this.nextDyeRT );
        this.quad.material = material;
        this.quad.render(this.renderer);

        //swap
        [this.dyeRT, this.nextDyeRT] = [this.nextDyeRT, this.dyeRT];
    }
}
