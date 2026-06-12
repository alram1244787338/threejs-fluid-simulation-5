/*
 * WebGPU (TSL) compute shaders for the fluid simulation.
 *
 * Each pass is a small `ComputeShader` subclass. They were split out of
 * FluidMaterialGPU.ts so the GPU kernels live apart from the material/orchestration code.
 *
 * Channel layout of the simulation texture:
 *   R - pressure
 *   G - velocity X
 *   B - velocity Y
 *   A - "wildcard" scratch passed between passes; not persisted (see WILDCARD_CHANNEL_RESET).
 *
 * Original WebGL shader code (c) 2017 Pavel Dobryakov, adapted for ThreeJs (c) 2025 Pablo Bandinopla. MIT.
 */
import {
    abs, add, clamp, Continue, distance, dot, Fn, If, instanceIndex, length, Loop, mix, mul,
    normalize, smoothstep, storage, texture, textureStore, uniform, vec2, vec4,
    type NodeRepresentation, type ShaderNodeObject,
} from "three/tsl";
import {
    Texture, Vector2,
    type ComputeNode, type Node, type StorageBufferAttribute, type StorageTexture, type WebGPURenderer,
} from "three/webgpu";
import {
    CENTRAL_DIFFERENCE_SCALE, DEFAULT_ADVECTION_DISSIPATION, DEFAULT_PRESSURE_DECAY,
    DEFAULT_SPLAT_FORCE, DEFAULT_SPLAT_THICKNESS, DEFAULT_SWIRL_INTENSITY,
    DEFAULT_VORTICITY_INFLUENCE, DIVISION_EPSILON, PIXEL_CENTER_OFFSET, PRESSURE_JACOBI_SCALE,
    WILDCARD_CHANNEL_RESET,
} from "./constants";
import type { NumberUniform, TextureSampleNode } from "./types";

/** Empty texture used as a placeholder until a real target is bound at render time. */
export const placeholderTexture = new Texture();
placeholderTexture.flipY = false;

/** Sample `sampler` at `uv` offset by `(x, y)` texels. */
const offsetSample = Fn<[TextureSampleNode, ShaderNodeObject<Node>, ShaderNodeObject<Node>, number, number]>(
    ([sampler, uv, texel, x, y]) => {
        return sampler.sample(uv.add(texel.mul(vec2(x, y))));
    },
);

// `encode`/`decode` are no-op hooks kept in place for a future range-remapping scheme.
const encode = Fn<[ShaderNodeObject<Node>, ShaderNodeObject<Node>]>(([_maxValue, value]) => {
    return value; // value.div(maxValue).add(1).div(2);
});
const decode = Fn<[ShaderNodeObject<Node>, ShaderNodeObject<Node>]>(([_maxValue, value]) => {
    return value; // value.mul(2).sub(1).mul(maxValue);
});

/**
 * Base class for a compute pass. The `fn` produces the output colour for a pixel; calling
 * `createBinds` compiles a kernel per target texture, and `renderBind` dispatches it.
 */
export class ComputeShader {
    private textureToShader: Map<StorageTexture, ShaderNodeObject<ComputeNode>>;

    constructor(
        private fn: (
            pixelPos: ShaderNodeObject<Node>,
            uvPos: ShaderNodeObject<Node>,
            texelSize: ShaderNodeObject<Node>,
        ) => NodeRepresentation,
    ) {
        this.textureToShader = new Map<StorageTexture, ShaderNodeObject<ComputeNode>>();
    }

    private create(outTo: StorageTexture, width: number, height: number) {
        return Fn(() => {
            const resolution = vec2(width, height);
            const posX = instanceIndex.mod(width);
            const posY = instanceIndex.div(width);
            const pixelPosition = vec2(posX, posY);
            const uvCoord = vec2(pixelPosition.add(vec2(PIXEL_CENTER_OFFSET, PIXEL_CENTER_OFFSET))).div(resolution);
            const textelSize = vec2(1, 1).div(resolution);

            return textureStore(outTo, pixelPosition, this.fn(pixelPosition, uvCoord, textelSize)).toWriteOnly();
        })().compute(width * height);
    }

    createBinds(width: number, height: number, ...targets: StorageTexture[]) {
        for (const target of targets) {
            this.textureToShader.set(target, this.create(target, width, height));
        }
        return this;
    }

    renderBind(renderer: WebGPURenderer, bindTarget: StorageTexture) {
        if (!this.textureToShader.has(bindTarget)) {
            throw new Error(
                "You are trying to render to a texture that this shader doesn't have. Did you forget to call createBinds?",
            );
        }

        renderer.compute(this.textureToShader.get(bindTarget)!);

        return bindTarget;
    }
}

/** Scrolls a texture by a UV offset (used when "following" a target). */
export class ScrollShader extends ComputeShader {
    readonly uvScroll = uniform(new Vector2());

    constructor(uTarget: TextureSampleNode) {
        super((_pixelPos, uvPos) => {
            return uTarget.sample(uvPos.add(this.uvScroll));
        });
    }
}

/** Introduces either velocity or colour into the target, depending on the `splatVelocity` flag. */
export class SplatShader extends ComputeShader {
    readonly splatVelocity = uniform(1);

    /** Splat strength, as a fraction of the max speed. */
    readonly splatForce = uniform(DEFAULT_SPLAT_FORCE);
    readonly thickness = uniform(DEFAULT_SPLAT_THICKNESS);

    constructor(
        uTarget: TextureSampleNode,
        positionAttr: StorageBufferAttribute,
        colorAttr: StorageBufferAttribute,
        count: number,
        maxVelocity: NumberUniform,
    ) {
        super((_pixelPos, vUv, textelSize) => {
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

                If(d.lessThan(textelSize.x.mul(ratio)), () => {
                    const influence = smoothstep(ratio, 0.0, d);

                    If(this.splatVelocity, () => {
                        const vel = normalize(diff).mul(this.splatForce.negate());

                        // `vel` ends up in [-1, 1] (UV-space velocity).
                        pixel.assign(vec4(pixel.r, encode(maxVelocity, vel), WILDCARD_CHANNEL_RESET));
                    }).Else(() => {
                        pixel.assign(mix(pixel, vec4(color, 1.0), influence));
                    });
                });
            });

            return pixel;
        });
    }
}

/** Writes the curl of the velocity field into the alpha channel. */
export class CurlShader extends ComputeShader {
    readonly vorticityInfluence = uniform(DEFAULT_VORTICITY_INFLUENCE);

    constructor(uVelocity: TextureSampleNode, maxVelocity: NumberUniform) {
        super((_, vUv, textelSize) => {
            const L = decode(maxVelocity, offsetSample(uVelocity, vUv, textelSize, -1, 0).b);
            const R = decode(maxVelocity, offsetSample(uVelocity, vUv, textelSize, 1, 0).b);
            const T = decode(maxVelocity, offsetSample(uVelocity, vUv, textelSize, 0, 1).g);
            const B = decode(maxVelocity, offsetSample(uVelocity, vUv, textelSize, 0, -1).g);

            const vorticity = R.sub(L).sub(T).add(B);
            const pixel = uVelocity.sample(vUv).toVar("pixel");
            const curl = this.vorticityInfluence.mul(vorticity);

            return vec4(pixel.xyz, encode(maxVelocity, curl));
        });
    }
}

/** Applies vorticity-confinement forces to the velocity field. */
export class VorticityShader extends ComputeShader {
    readonly curl = uniform(DEFAULT_SWIRL_INTENSITY);
    readonly delta = uniform(0);

    constructor(uTarget: TextureSampleNode, maxVelocity: NumberUniform) {
        super((_, vUv, textelSize) => {
            const L = decode(maxVelocity, offsetSample(uTarget, vUv, textelSize, -1, 0).a);
            const R = decode(maxVelocity, offsetSample(uTarget, vUv, textelSize, 1, 0).a);
            const T = decode(maxVelocity, offsetSample(uTarget, vUv, textelSize, 0, 1).a);
            const B = decode(maxVelocity, offsetSample(uTarget, vUv, textelSize, 0, -1).a);

            const pixel = uTarget.sample(vUv).toVar("pixel");
            const C = decode(maxVelocity, pixel.a);

            const force = mul(CENTRAL_DIFFERENCE_SCALE, vec2(abs(T).sub(abs(B)), abs(R).sub(abs(L)))).toVar("force");

            force.divAssign(length(force).add(DIVISION_EPSILON));
            force.mulAssign(this.curl.mul(C));
            force.mulAssign(vec2(0, -1.0));

            const velocity = decode(maxVelocity, pixel.gb.add(force.mul(this.delta)));

            return vec4(pixel.r, encode(maxVelocity, velocity), WILDCARD_CHANNEL_RESET);
        });
    }
}

/** Writes the divergence of the velocity field into the alpha channel. */
export class DivergenceShader extends ComputeShader {
    constructor(uVelocity: TextureSampleNode, maxVelocity: NumberUniform) {
        super((_, vUv, textelSize) => {
            const L = decode(maxVelocity, offsetSample(uVelocity, vUv, textelSize, -1, 0).g).toVar("L");
            const R = decode(maxVelocity, offsetSample(uVelocity, vUv, textelSize, 1, 0).g).toVar("R");
            const T = decode(maxVelocity, offsetSample(uVelocity, vUv, textelSize, 0, 1).b).toVar("T");
            const B = decode(maxVelocity, offsetSample(uVelocity, vUv, textelSize, 0, -1).b).toVar("B");

            const pixel = uVelocity.sample(vUv);
            const C = decode(maxVelocity, pixel.gb); // velocity info

            If(vUv.x.sub(textelSize.x).lessThan(0), () => L.assign(C.x.negate()));
            If(vUv.x.add(textelSize.x).greaterThan(1), () => R.assign(C.x.negate()));
            If(vUv.y.add(textelSize.y).greaterThan(1), () => T.assign(C.y.negate()));
            If(vUv.y.sub(textelSize.y).lessThan(0), () => B.assign(C.y.negate()));

            const div = mul(CENTRAL_DIFFERENCE_SCALE, R.sub(L).add(T.sub(B)));

            return vec4(pixel.r, pixel.gb, decode(2, div));
        });
    }
}

/** Decays the pressure field by `decay` each call. */
export class ClearShader extends ComputeShader {
    readonly decay = uniform(DEFAULT_PRESSURE_DECAY);

    constructor(uTarget: TextureSampleNode) {
        super((_, vUv) => {
            const pixel = uTarget.sample(vUv);
            return vec4(pixel.r.mul(this.decay), pixel.gba);
        });
    }
}

/** One Jacobi iteration of the pressure solve. */
export class PressureShader extends ComputeShader {
    constructor(uTarget: TextureSampleNode, maxVelocity: NumberUniform) {
        super((_, vUv, textelSize) => {
            const L = decode(maxVelocity, offsetSample(uTarget, vUv, textelSize, -1, 0).x);
            const R = decode(maxVelocity, offsetSample(uTarget, vUv, textelSize, 1, 0).x);
            const T = decode(maxVelocity, offsetSample(uTarget, vUv, textelSize, 0, 1).x);
            const B = decode(maxVelocity, offsetSample(uTarget, vUv, textelSize, 0, -1).x);
            const pixel = uTarget.sample(vUv).toVar();

            const divergence = decode(maxVelocity, pixel.a);

            const pressure = mul(L.add(R).add(B).add(T).sub(divergence), PRESSURE_JACOBI_SCALE);

            return vec4(encode(maxVelocity, pressure), pixel.gba);
        });
    }
}

/** Subtracts the pressure gradient from the velocity field to enforce incompressibility. */
export class GradientSubtractShader extends ComputeShader {
    constructor(uTarget: TextureSampleNode, maxVelocity: NumberUniform) {
        super((_, vUv, textelSize) => {
            const L = decode(maxVelocity, offsetSample(uTarget, vUv, textelSize, -1, 0).x);
            const R = decode(maxVelocity, offsetSample(uTarget, vUv, textelSize, 1, 0).x);
            const T = decode(maxVelocity, offsetSample(uTarget, vUv, textelSize, 0, 1).x);
            const B = decode(maxVelocity, offsetSample(uTarget, vUv, textelSize, 0, -1).x);

            const pixel = uTarget.sample(vUv).toVar();
            const velocity = decode(maxVelocity, pixel.gb).toVar("velocity");

            velocity.subAssign(vec2(R.sub(L), T.sub(B)));

            return vec4(pixel.r, encode(maxVelocity, velocity), WILDCARD_CHANNEL_RESET);
        });
    }
}

/** Advects either the velocity field or the dye along the velocity field. */
export class AdvectShader extends ComputeShader {
    readonly sourceIsVelocity = uniform(0);
    readonly delta = uniform(0);
    readonly dissipation = uniform(DEFAULT_ADVECTION_DISSIPATION);
    readonly uSource: TextureSampleNode = texture(placeholderTexture);

    constructor(uVelocity: TextureSampleNode, maxVelocity: NumberUniform) {
        super((_, vUv, _textelSize) => {
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
