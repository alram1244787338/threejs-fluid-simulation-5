/**
 * DoubleBuffer – encapsulates the "ping-pong" pair of render targets
 * used throughout the fluid simulation.
 *
 * Instead of juggling four loose variables (`currentRT`, `nextRT`,
 * `dyeRT`, `nextDyeRT`) and swapping them by hand, the material now
 * owns two `DoubleBuffer` instances — one for velocity data and one
 * for dye / colour — and calls `.swap()` after each render pass.
 *
 * The class is generic over the texture type so it works for both
 * WebGL (`WebGLRenderTarget`) and WebGPU (`StorageTexture`).
 */
export class DoubleBuffer<T> {

    /** The texture that holds the most recent result. */
    public current: T;
    /** The texture that will be written to on the next pass. */
    public next: T;

    constructor(current: T, next: T) {
        this.current = current;
        this.next = next;
    }

    /**
     * Exchange `current` ↔ `next`.  Call this after rendering into
     * `next` so that it becomes the new `current`.
     */
    swap(): void {
        [this.current, this.next] = [this.next, this.current];
    }
}
