/*
 * Displacement / normal injection for the WebGL MeshPhysicalMaterial.
 *
 * A plain (non-node) MeshPhysicalMaterial has no shader graph, so the supported way to customise
 * it is to patch the built-in shader chunks in `onBeforeCompile`. The risk with raw string
 * `.replace()` is that it fails *silently* if a future three.js renames a chunk. {@link replaceInclude}
 * removes that risk: it asserts the chunk exists and throws a precise, actionable error if not, so
 * an upgrade surfaces as a clear message pointing at the exact chunk to fix — instead of a mesh
 * that quietly renders without displacement.
 */
import type { WebGLProgramParametersWithUniforms } from "three";
import { glslFloat, LUMINANCE_MIDPOINT, LUMINANCE_WEIGHTS } from "./constants";

/**
 * Replace a `#include <name>` chunk in a shader, failing loudly if the chunk is missing.
 * The replacement should re-include the original chunk itself when you mean to *augment* it.
 */
function replaceInclude(source: string, includeName: string, replacement: string): string {
    const token = `#include <${includeName}>`;

    if (!source.includes(token)) {
        throw new Error(
            `[FluidV3Material] expected shader chunk "${token}" was not found while patching the ` +
            `displacement shader. The installed three.js version may have renamed it; ` +
            `update src/webgl/displacement.ts to match.`,
        );
    }

    return source.replace(token, replacement);
}

const luminanceVec3 =
    `vec3(${glslFloat(LUMINANCE_WEIGHTS[0])}, ${glslFloat(LUMINANCE_WEIGHTS[1])}, ${glslFloat(LUMINANCE_WEIGHTS[2])})`;

/** Vertex: lift the surface along its normal by the brightest displacement-map channel. */
const vertexDisplacementChunk = `
    #ifdef USE_DISPLACEMENTMAP

        vec3 dispColor = texture2D( displacementMap, vUv ).rgb;
        float displacement = max( max(dispColor.r, dispColor.g), dispColor.b );

        transformed += normalize( objectNormal ) * ( displacement * displacementScale + displacementBias );

    #endif
`;

/** Fragment: recompute the normal from the displaced surface so lighting follows the bumps. */
const fragmentNormalChunk = `
    vec3 dColor = texture2D(displacementMap, vUv).rgb;
    float luminance = dot(dColor, ${luminanceVec3});

    float d = luminance - ${glslFloat(LUMINANCE_MIDPOINT)};
    vec3 displacedWorld = vWorldPos + vec3(0.0, d * displacementScale, 0.0);

    vec3 dx = dFdx(displacedWorld);
    vec3 dy = dFdy(displacedWorld);
    vec3 displacedNormal = normalize(cross(dx, dy));

    vec3 normalView = normalize(normalMatrix * displacedNormal);
    vec3 normal = normalView;
    vec3 nonPerturbedNormal = normalView;
`;

/**
 * Patch a MeshPhysicalMaterial's compiled shader so the surface is displaced by the dye height
 * and its normals are recomputed to match. Call from the material's `onBeforeCompile`.
 */
export function applyDisplacementShader(shader: WebGLProgramParametersWithUniforms): void {
    // Pass UV and (object-space) position through to the fragment shader.
    shader.vertexShader = replaceInclude(shader.vertexShader, "common", `#include <common>
                varying vec2 vUv;
                varying vec3 vWorldPos;`);
    shader.vertexShader = replaceInclude(shader.vertexShader, "uv_vertex", `#include <uv_vertex>
                vUv = uv;`);
    shader.vertexShader = replaceInclude(shader.vertexShader, "project_vertex", `#include <project_vertex>
                vWorldPos = position;`);
    shader.vertexShader = replaceInclude(shader.vertexShader, "displacementmap_vertex", vertexDisplacementChunk);

    // Displace in the fragment shader and recompute normals from that.
    shader.fragmentShader = replaceInclude(shader.fragmentShader, "common", `#include <common>
                uniform sampler2D displacementMap;
                uniform float displacementScale;
                uniform mat3 normalMatrix;
                varying vec2 vUv;
                varying vec3 vWorldPos;`);
    shader.fragmentShader = replaceInclude(shader.fragmentShader, "normal_fragment_begin", fragmentNormalChunk);
}
