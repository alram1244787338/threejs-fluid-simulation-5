/**
 * WebGPU compute-shader classes for the fluid simulation.
 *
 * Extracted from the former monolithic `FluidMaterialGPU.ts` so the
 * shader logic can be read, tested and maintained independently from
 * the material that drives it.
 */

import {
    type NodeRepresentation,
    storage, abs, add, clamp, Continue, cross, distance, dot, Fn, If,
    instanceIndex, length, Loop, max, mix, modelNormalMatrix, mul,
    normalGeometry, normalize, positionLocal, smoothstep, texture,
    textureStore, uniform, uv, vec2, vec3, vec4,
    type ShaderNodeObject,
} from "three/tsl";
import {
    type Node, type StorageBufferAttribute, type Texture,
    StorageTexture, TextureNode, UniformNode, Vector2, ComputeNode,
    type WebGPURenderer,
} from "three/webgpu";
import {
    DEFAULT_SPLAT_FORCE_GPU,
    DEFAULT_SPLAT_THICKNESS,
    DEFAULT_VORTICITY_INFLUENCE,
    DEFAULT_SWIRL_INTENSITY,
    DEFAULT_PRESSURE_DECAY,
    VORTICITY_FORCE_EPSILON,
    VELOCITY_ALPHA_CLEAR,
} from "./FluidConstants";
import type { TSLTextureSampler, TSLNumberUniform } from "./FluidTypes";

// ──────────────────────────────────────────────────────────────
//  Placeholder texture used before any real texture is bound
// ──────────────────────────────────────────────────────────────

const placeholderTexture = new TextureNode(new StorageTexture(1, 1));
// We never actually sample real data from this; it just keeps the
// TSL graph valid until a real texture is wired in.

// ──────────────────────────────────────────────────────────────
//  Helper: offset texture sample
// ──────────────────────────────────────────────────────────────

const offsetSample = Fn<[TSLTextureSampler, ShaderNodeObject<Node>, ShaderNodeObject<Node>, number, number]>(
    ([sampler, sampleUv, texel, x, y]) => {
        return sampler.sample(sampleUv.add(texel.mul(vec2(x, y))));
    },
);

// ──────────────────────────────────────────────────────────────
//  Encode / decode helpers
//  (Currently pass-through; kept as a seam so an R11F/RG16F packing
//   optimisation can be slotted in later without touching shaders.)
// ──────────────────────────────────────────────────────────────

const encode = Fn<[ShaderNodeObject<Node>, ShaderNodeObject<Node>]>(
    ([_maxValue, value]) => value,
);

const decode = Fn<[ShaderNodeObject<Node>, ShaderNodeObject<Node>]>(
    ([_maxValue, value]) => value,
);

// ════════════════════════════════════════════════════════════════
//  Base compute shader
// ════════════════════════════════════════════════════════════════

export type ComputeFn = (
    pixelPos: ShaderNodeObject<Node>,
    uvPos: ShaderNodeObject<Node>,
    texelSize: ShaderNodeObject<Node>,
) => NodeRepresentation;

export class ComputeShader {

    private textureToShader: Map<Texture, ShaderNodeObject<ComputeNode>>;

    constructor(protected fn: ComputeFn) {
        this.textureToShader = new Map();
    }

    private create(outTo: Texture, width: number, height: number) {
        return Fn(() => {
            const resolution = vec2(width, height);
            const posX = instanceIndex.mod(width);
            const posY = instanceIndex.div(width);
            const pixelPosition = vec2(posX, posY);
            const uvCoord = vec2(pixelPosition.add(vec2(0.5, 0.5))).div(resolution);
            const texelSize = vec2(1, 1).div(resolution);

            return textureStore(
                outTo,
                pixelPosition,
                this.fn(pixelPosition, uvCoord, texelSize),
            ).toWriteOnly();
        })().compute(width * height);
    }

    createBinds(width: number, height: number, ...targets: Texture[]) {
        for (const target of targets)
            this.textureToShader.set(target, this.create(target, width, height));
        return this;
    }

    renderBind(renderer: WebGPURenderer, bindTarget: Texture) {
        if (!this.textureToShader.has(bindTarget)) {
            throw new Error(
                "You are trying to render to a texture that this shader doesn't have. " +
                "Did you forget to call createBindTo?",
            );
        }
        renderer.compute(this.textureToShader.get(bindTarget)!);
        return bindTarget;
    }
}

// ════════════════════════════════════════════════════════════════
//  Concrete shaders
// ════════════════════════════════════════════════════════════════

export class ScrollShader extends ComputeShader {
    readonly uvScroll = uniform(new Vector2());

    constructor(uTarget: TSLTextureSampler) {
        super((_pixelPos, uvPos) => {
            return uTarget.sample(uvPos.add(this.uvScroll));
        });
    }
}

export class SplatShader extends ComputeShader {
    readonly splatVelocity = uniform(1);
    readonly splatForce = uniform(DEFAULT_SPLAT_FORCE_GPU);
    readonly thickness = uniform(DEFAULT_SPLAT_THICKNESS);

    constructor(
        uTarget: TSLTextureSampler,
        positionAttr: StorageBufferAttribute,
        colorAttr: StorageBufferAttribute,
        count: number,
        maxVelocity: TSLNumberUniform,
    ) {
        super((_pixelPos, vUv, texelSize) => {
            const pixel = uTarget.sample(vUv).toVar("pixel");

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
                const t = clamp(dot(toFrag, diff).div(dot(diff, diff)), 0, 1);
                const proj = prev.add(t.mul(diff));

                const d = distance(vUv, proj);

                If(d.lessThan(texelSize.x.mul(ratio)), () => {
                    const influence = smoothstep(ratio, 0.0, d);

                    If(this.splatVelocity, () => {
                        const vel = normalize(diff).mul(this.splatForce.negate());
                        pixel.assign(vec4(pixel.r, encode(maxVelocity, vel), 0));
                    }).Else(() => {
                        pixel.assign(mix(pixel, vec4(color, 1.0), influence));
                    });
                });
            });

            return pixel;
        });
    }
}

export class CurlShader extends ComputeShader {
    readonly vorticityInfluence = uniform(DEFAULT_VORTICITY_INFLUENCE);

    constructor(uVelocity: TSLTextureSampler, maxVelocity: TSLNumberUniform) {
        super((_, vUv, texelSize) => {
            const L = decode(maxVelocity, offsetSample(uVelocity, vUv, texelSize, -1, 0).b);
            const R = decode(maxVelocity, offsetSample(uVelocity, vUv, texelSize, 1, 0).b);
            const T = decode(maxVelocity, offsetSample(uVelocity, vUv, texelSize, 0, 1).g);
            const B = decode(maxVelocity, offsetSample(uVelocity, vUv, texelSize, 0, -1).g);

            const vorticity = R.sub(L).sub(T).add(B);
            const pixel = uVelocity.sample(vUv).toVar("pixel");
            const curl = this.vorticityInfluence.mul(vorticity);

            return vec4(pixel.xyz, encode(maxVelocity, curl));
        });
    }
}

export class VorticityShader extends ComputeShader {
    readonly curl = uniform(DEFAULT_SWIRL_INTENSITY);
    readonly delta = uniform(0);

    constructor(uTarget: TSLTextureSampler, maxVelocity: TSLNumberUniform) {
        super((_, vUv, texelSize) => {
            const L = decode(maxVelocity, offsetSample(uTarget, vUv, texelSize, -1, 0).a);
            const R = decode(maxVelocity, offsetSample(uTarget, vUv, texelSize, 1, 0).a);
            const T = decode(maxVelocity, offsetSample(uTarget, vUv, texelSize, 0, 1).a);
            const B = decode(maxVelocity, offsetSample(uTarget, vUv, texelSize, 0, -1).a);

            const pixel = uTarget.sample(vUv).toVar("pixel");
            const C = decode(maxVelocity, pixel.a);

            const force = mul(0.5, vec2(abs(T).sub(abs(B)), abs(R).sub(abs(L)))).toVar("force");

            // Normalise with an epsilon guard to prevent division by zero
            force.divAssign(length(force).add(VORTICITY_FORCE_EPSILON));
            force.mulAssign(this.curl.mul(C));
            force.mulAssign(vec2(0, -1.0));

            const velocity = decode(maxVelocity, pixel.gb.add(force.mul(this.delta)));

            return vec4(pixel.r, encode(maxVelocity, velocity), VELOCITY_ALPHA_CLEAR);
        });
    }
}

export class DivergenceShader extends ComputeShader {
    constructor(uVelocity: TSLTextureSampler, maxVelocity: TSLNumberUniform) {
        super((_, vUv, texelSize) => {
            const L = decode(maxVelocity, offsetSample(uVelocity, vUv, texelSize, -1, 0).g).toVar("L");
            const R = decode(maxVelocity, offsetSample(uVelocity, vUv, texelSize, 1, 0).g).toVar("R");
            const T = decode(maxVelocity, offsetSample(uVelocity, vUv, texelSize, 0, 1).b).toVar("T");
            const B = decode(maxVelocity, offsetSample(uVelocity, vUv, texelSize, 0, -1).b).toVar("B");

            const pixel = uVelocity.sample(vUv);
            const C = decode(maxVelocity, pixel.gb);

            If(vUv.x.sub(texelSize.x).lessThan(0), () => L.assign(C.x.negate()));
            If(vUv.x.add(texelSize.x).greaterThan(1), () => R.assign(C.x.negate()));
            If(vUv.y.add(texelSize.y).greaterThan(1), () => T.assign(C.y.negate()));
            If(vUv.y.sub(texelSize.y).lessThan(0), () => B.assign(C.y.negate()));

            const div = mul(0.5, R.sub(L).add(T.sub(B)));

            return vec4(pixel.r, pixel.gb, decode(2, div));
        });
    }
}

export class ClearShader extends ComputeShader {
    readonly decay = uniform(DEFAULT_PRESSURE_DECAY);

    constructor(uTarget: TSLTextureSampler) {
        super((_, vUv) => {
            const pixel = uTarget.sample(vUv);
            return vec4(pixel.r.mul(this.decay), pixel.gba);
        });
    }
}

export class PressureShader extends ComputeShader {
    constructor(uTarget: TSLTextureSampler, maxVelocity: TSLNumberUniform) {
        super((_, vUv, texelSize) => {
            const L = decode(maxVelocity, offsetSample(uTarget, vUv, texelSize, -1, 0).x);
            const R = decode(maxVelocity, offsetSample(uTarget, vUv, texelSize, 1, 0).x);
            const T = decode(maxVelocity, offsetSample(uTarget, vUv, texelSize, 0, 1).x);
            const B = decode(maxVelocity, offsetSample(uTarget, vUv, texelSize, 0, -1).x);
            const pixel = uTarget.sample(vUv).toVar();

            const divergence = decode(maxVelocity, pixel.a);
            const pressure = mul(L.add(R).add(B).add(T).sub(divergence), 0.25);

            return vec4(encode(maxVelocity, pressure), pixel.gba);
        });
    }
}

export class GradientSubtractShader extends ComputeShader {
    constructor(uTarget: TSLTextureSampler, maxVelocity: TSLNumberUniform) {
        super((_, vUv, texelSize) => {
            const L = decode(maxVelocity, offsetSample(uTarget, vUv, texelSize, -1, 0).x);
            const R = decode(maxVelocity, offsetSample(uTarget, vUv, texelSize, 1, 0).x);
            const T = decode(maxVelocity, offsetSample(uTarget, vUv, texelSize, 0, 1).x);
            const B = decode(maxVelocity, offsetSample(uTarget, vUv, texelSize, 0, -1).x);

            const pixel = uTarget.sample(vUv).toVar();
            const velocity = decode(maxVelocity, pixel.gb).toVar("velocity");

            velocity.subAssign(vec2(R.sub(L), T.sub(B)));

            return vec4(pixel.r, encode(maxVelocity, velocity), VELOCITY_ALPHA_CLEAR);
        });
    }
}

export class AdvectShader extends ComputeShader {
    readonly sourceIsVelocity = uniform(0);
    readonly delta = uniform(0);
    readonly dissipation = uniform(0.1);
    readonly uSource: TSLTextureSampler = texture(placeholderTexture);

    constructor(uVelocity: TSLTextureSampler, maxVelocity: TSLNumberUniform) {
        super((_, vUv, _texelSize) => {
            const original = uVelocity.sample(vUv);
            const velocity = decode(maxVelocity, original.yz);
            const coord = vUv.sub(this.delta.mul(velocity));
            const result = this.uSource.sample(coord).toVar("pixel");
            const decay = add(1.0, this.dissipation.mul(this.delta));
            result.divAssign(decay);

            If(this.sourceIsVelocity, () => {
                result.assign(vec4(original.r, result.gb, original.w));
            });

            return result;
        });
    }
}
