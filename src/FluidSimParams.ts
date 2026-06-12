/**
 * FluidSimParams — 集中管理流体模拟的所有可配置参数
 *
 * - 统一定义参数名、默认值、范围、校验
 * - 自动绑定到 shader uniform 或任意 getter/setter
 * - 自动生成 lil-gui 调试面板
 * - 支持变更回调（单参数监听 / 全局监听）
 */
import GUI from "three/examples/jsm/libs/lil-gui.module.min.js";

// ─── 参数定义类型 ───────────────────────────────────────────────────────────

export interface ParamDefinition {
    /** 参数唯一标识 */
    key: string;
    /** GUI 显示名称 */
    label: string;
    /** 默认值 */
    defaultValue: number;
    /** GUI slider 最小值 */
    min: number;
    /** GUI slider 最大值 */
    max: number;
    /** GUI slider 步长（不填则由 lil-gui 自动） */
    step?: number;
    /** 自定义校验函数，返回修正后的值 */
    validate?: (v: number) => number;
}

/** 变更回调签名 */
export type ParamChangeListener = (key: string, newValue: number, oldValue: number) => void;

/** 参数绑定：实际 uniform / 属性的读写接口 */
export interface ParamBinding {
    get: () => number;
    set: (v: number) => void;
}

// ─── 统一参数定义 ───────────────────────────────────────────────────────────
//
// 所有默认值只在这里维护一处。
// validate 用于防止非法值（负数、0 等）。

export const FLUID_PARAM_DEFINITIONS: ParamDefinition[] = [
    {
        key: "splatForce",
        label: "Splat Force",
        defaultValue: -200,
        min: -1000,
        max: 1000,
    },
    {
        key: "splatThickness",
        label: "Splat Thickness",
        defaultValue: 0.01,
        min: 0.001,
        max: 1,
        validate: (v) => Math.max(0.001, v),
    },
    {
        key: "vorticityInfluence",
        label: "Vorticity Influence",
        defaultValue: 1,
        min: 0,
        max: 1,
    },
    {
        key: "swirlIntensity",
        label: "Swirl Intensity",
        defaultValue: 10,
        min: 0,
        max: 100,
    },
    {
        key: "pressureDecay",
        label: "Pressure Decay",
        defaultValue: 0.3,
        min: 0,
        max: 1,
    },
    {
        key: "velocityDissipation",
        label: "Velocity Dissipation",
        defaultValue: 0.28,
        min: 0,
        max: 1,
        validate: (v) => Math.max(0, v),
    },
    {
        key: "densityDissipation",
        label: "Density Dissipation",
        defaultValue: 0.5,
        min: 0,
        max: 1,
        validate: (v) => Math.max(0, v),
    },
    {
        key: "displacementScale",
        label: "Displacement Scale",
        defaultValue: 0.01,
        min: -1,
        max: 1,
    },
    {
        key: "pressureIterations",
        label: "Pressure Iterations",
        defaultValue: 40,
        min: 1,
        max: 100,
        step: 1,
        validate: (v) => Math.max(1, Math.round(v)),
    },
];

// ─── 参数管理器 ─────────────────────────────────────────────────────────────

/**
 * 管理所有模拟参数的中心存储。
 *
 * 用法：
 * ```ts
 * const params = new FluidSimParams();
 *
 * // 绑定到 shader uniform
 * params.bind("splatForce", {
 *     get: () => splat.uniforms.splatForce.value,
 *     set: (v) => { splat.uniforms.splatForce.value = v; },
 * });
 *
 * // 读取 / 写入
 * params.set("splatForce", -100);
 * params.get("splatForce"); // -100
 *
 * // 监听变更
 * params.onChange("splatForce", (key, newVal, oldVal) => { ... });
 * params.onAnyChange((key, newVal, oldVal) => { ... });
 *
 * // 批量应用 / 导出
 * params.applySettings({ splatForce: -50, swirlIntensity: 20 });
 * const snapshot = params.getSettings();
 *
 * // 自动生成 GUI
 * params.addDebugPanel(gui, "Fluid");
 * ```
 */
export class FluidSimParams {
    /** 当前值 */
    private values: Map<string, number> = new Map();

    /** 参数定义 */
    private definitions: Map<string, ParamDefinition> = new Map();

    /** 参数绑定（uniform / 属性读写器） */
    private bindings: Map<string, ParamBinding> = new Map();

    /** 按参数名注册的监听器 */
    private listeners: Map<string, ParamChangeListener[]> = new Map();

    /** 全局监听器（任何参数变更都会触发） */
    private globalListeners: ParamChangeListener[] = [];

    constructor(definitions: ParamDefinition[] = FLUID_PARAM_DEFINITIONS) {
        for (const def of definitions) {
            this.definitions.set(def.key, def);
            this.values.set(def.key, def.defaultValue);
        }
    }

    // ─── 绑定 ────────────────────────────────────────────────────────────────

    /**
     * 将参数绑定到实际的 uniform / 属性。
     * 绑定后立即把当前值推送过去。
     */
    bind(key: string, binding: ParamBinding): void {
        this.bindings.set(key, binding);
        const value = this.values.get(key);
        if (value !== undefined) {
            binding.set(value);
        }
    }

    /**
     * 便捷绑定：传入一个拥有 `.value` 属性的对象（如 THREE uniform）。
     */
    bindUniform(key: string, uniform: { value: number }): void {
        this.bind(key, {
            get: () => uniform.value,
            set: (v) => { uniform.value = v; },
        });
    }

    // ─── 读写 ────────────────────────────────────────────────────────────────

    get(key: string): number {
        return this.values.get(key) ?? 0;
    }

    set(key: string, value: number): void {
        const def = this.definitions.get(key);
        if (!def) {
            console.warn(`[FluidSimParams] unknown parameter "${key}"`);
            return;
        }

        // 自定义校验
        if (def.validate) {
            value = def.validate(value);
        }

        // 范围裁剪
        value = Math.max(def.min, Math.min(def.max, value));

        const oldValue = this.values.get(key) ?? def.defaultValue;
        this.values.set(key, value);

        // 推送到绑定
        const binding = this.bindings.get(key);
        if (binding) {
            binding.set(value);
        }

        // 只在值真正变化时通知
        if (oldValue !== value) {
            const keyListeners = this.listeners.get(key);
            if (keyListeners) {
                for (const listener of keyListeners) {
                    listener(key, value, oldValue);
                }
            }
            for (const listener of this.globalListeners) {
                listener(key, value, oldValue);
            }
        }
    }

    // ─── 变更监听 ─────────────────────────────────────────────────────────────

    onChange(key: string, listener: ParamChangeListener): void {
        if (!this.listeners.has(key)) {
            this.listeners.set(key, []);
        }
        this.listeners.get(key)!.push(listener);
    }

    onAnyChange(listener: ParamChangeListener): void {
        this.globalListeners.push(listener);
    }

    removeListener(key: string, listener: ParamChangeListener): void {
        const keyListeners = this.listeners.get(key);
        if (keyListeners) {
            const idx = keyListeners.indexOf(listener);
            if (idx >= 0) keyListeners.splice(idx, 1);
        }
    }

    removeAllListeners(): void {
        this.listeners.clear();
        this.globalListeners.length = 0;
    }

    // ─── 批量操作 ─────────────────────────────────────────────────────────────

    /**
     * 批量应用参数（如从 JSON 加载预设）。
     */
    applySettings(settings: Partial<Record<string, number>>): void {
        for (const [key, value] of Object.entries(settings)) {
            if (value !== undefined) {
                this.set(key, value);
            }
        }
    }

    /**
     * 导出当前所有参数值（用于序列化 / 复制）。
     */
    getSettings(): Record<string, number> {
        const result: Record<string, number> = {};
        for (const [key, value] of this.values) {
            result[key] = value;
        }
        return result;
    }

    /**
     * 将所有参数恢复为定义时的默认值。
     */
    resetToDefaults(): void {
        for (const def of this.definitions.values()) {
            this.set(def.key, def.defaultValue);
        }
    }

    // ─── Debug GUI ───────────────────────────────────────────────────────────

    /**
     * 自动生成 lil-gui 调试面板。
     * 参数的 GUI 控件直接从定义生成，不用手写 panel.add()。
     */
    addDebugPanel(gui: GUI, name = "Fluid Simulation"): GUI {
        const panel = gui.addFolder(name);

        // lil-gui 需要一个可写对象
        const proxy: Record<string, number> = {};
        for (const def of this.definitions.values()) {
            proxy[def.key] = this.values.get(def.key) ?? def.defaultValue;
        }

        for (const def of this.definitions.values()) {
            const controller = panel.add(proxy, def.key, def.min, def.max, def.step);
            controller.name(def.label);
            controller.onChange((v: number) => {
                this.set(def.key, v);
            });

            // 当参数被外部 set() 修改时，同步回 proxy（以便 GUI 刷新）
            this.onChange(def.key, (_key, newVal) => {
                proxy[def.key] = newVal;
                controller.updateDisplay();
            });
        }

        panel.add({
            copySettings: () => {
                navigator.clipboard.writeText(JSON.stringify(this.getSettings(), null, 2));
            },
        }, "copySettings").name("📋 Copy Settings");

        panel.add({
            resetDefaults: () => this.resetToDefaults(),
        }, "resetDefaults").name("↺ Reset Defaults");

        return panel;
    }

    /**
     * 获取参数定义列表（供外部工具使用）。
     */
    getDefinitions(): ParamDefinition[] {
        return Array.from(this.definitions.values());
    }
}
