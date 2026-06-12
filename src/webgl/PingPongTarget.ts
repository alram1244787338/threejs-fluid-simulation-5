/*
 * Ping-pong pair of WebGL render targets for the simulation passes.
 *
 * Encapsulates the read/write swap so the material no longer juggles four bare
 * `currentRT` / `nextRT` / `dyeRT` / `nextDyeRT` references by hand.
 */
import {
    FloatType, LinearFilter, WebGLRenderTarget,
    type MagnificationTextureFilter, type MinificationTextureFilter,
} from "three";

export class PingPongTarget {
    private _read: WebGLRenderTarget;
    private _write: WebGLRenderTarget;

    /**
     * @param filter Texture filtering for both targets. Defaults to `LinearFilter` (Three's own
     *   default). Pass `NearestFilter` when the hardware can't linearly filter float textures and
     *   the shader does its own (manual) bilinear filtering instead.
     */
    constructor(
        width: number,
        height: number,
        filter: MinificationTextureFilter & MagnificationTextureFilter = LinearFilter,
    ) {
        const options = { type: FloatType, minFilter: filter, magFilter: filter };
        this._read = new WebGLRenderTarget(width, height, options);
        this._write = new WebGLRenderTarget(width, height, options);
    }

    /** The render target the passes should sample from this frame. */
    get read(): WebGLRenderTarget {
        return this._read;
    }

    /** The render target the next pass should render into. */
    get write(): WebGLRenderTarget {
        return this._write;
    }

    /** Convenience accessor for the readable texture. */
    get texture() {
        return this._read.texture;
    }

    /** Promote the freshly written target to be the new read target. */
    swap(): void {
        [this._read, this._write] = [this._write, this._read];
    }
}
