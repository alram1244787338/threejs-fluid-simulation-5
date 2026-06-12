/*
 * Ping-pong pair of storage textures for the WebGPU compute passes.
 *
 * Encapsulates the read/write swap so the material no longer juggles four bare
 * `currentRT` / `nextRT` / `dyeRT` / `nextDyeRT` references by hand.
 */
import { FloatType, StorageTexture } from "three/webgpu";

export class PingPongTexture {
    private _read: StorageTexture;
    private _write: StorageTexture;

    constructor(width: number, height: number) {
        this._read = PingPongTexture.createTarget(width, height);
        this._write = PingPongTexture.createTarget(width, height);
    }

    private static createTarget(width: number, height: number): StorageTexture {
        const texture = new StorageTexture(width, height);
        texture.type = FloatType;
        return texture;
    }

    /** The texture the passes should sample from this frame. */
    get read(): StorageTexture {
        return this._read;
    }

    /** The texture the next pass should render into. */
    get write(): StorageTexture {
        return this._write;
    }

    /**
     * Both underlying textures. A compute shader must be bound to each side up front
     * (via `createBinds`) because either can become the write target after a swap.
     */
    get textures(): readonly StorageTexture[] {
        return [this._read, this._write];
    }

    /** Promote the freshly written texture to be the new read target. */
    swap(): void {
        [this._read, this._write] = [this._write, this._read];
    }
}
