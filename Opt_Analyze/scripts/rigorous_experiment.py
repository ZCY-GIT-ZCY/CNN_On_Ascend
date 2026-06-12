"""
科学严谨的两阶段对比实验。

Phase 1 - 噪声基底标定：
  将 reexport_plain（统一编译基线）独立跑 N 次，量化环境噪声。
  输出：噪声的均值、标准差、95%置信区间、最小可检测效应量。

Phase 2 - 优化效果测量：
  所有变体与 reexport_plain 在随机顺序下各跑 M 轮。
  每轮记录：mean/std/p50/p95/p99 + 系统负载。
  输出：每个变体相对基线的差异、t检验p值、效应量。

两个模型都做。
"""
import json
import math
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
OPT_ANALYZE = SCRIPT_DIR.parent
RESULTS_DIR = OPT_ANALYZE / "data"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_BASE = OPT_ANALYZE.parent
OPTIM_DIR = MODELS_BASE / "Optimization" / "models"

# 模型配置
MODELS = {
    "mobilenet_v3_small": {
        "baseline_om": "/home/HwHiAiUser/Desktop/CNN/Optimization/models/mobilenet_v3_small_reexport_plain.om",
        "variants": {
            "fp16":          f"{OPTIM_DIR}/mobilenet_v3_small_fp16.om",
            "shape_inferred": f"{OPTIM_DIR}/mobilenet_v3_small_shape_inferred.om",
            "opset13":        f"{OPTIM_DIR}/mobilenet_v3_small_opset13.om",
            "bn_folded":      f"{OPTIM_DIR}/mobilenet_v3_small_bn_folded.om",
            "channels_last":  f"{OPTIM_DIR}/mobilenet_v3_small_channels_last.om",
            "onnxsim":        f"{OPTIM_DIR}/mobilenet_v3_small_onnxsim.om",
            "onnxoptimizer":  f"{OPTIM_DIR}/mobilenet_v3_small_onnxoptimizer.om",
        },
    },
    "resnet50": {
        "baseline_om": "/home/HwHiAiUser/Desktop/CNN/Optimization/models/resnet50_reexport_plain.om",
        "variants": {
            "fp16":          f"{OPTIM_DIR}/resnet50_fp16.om",
            "shape_inferred": f"{OPTIM_DIR}/resnet50_shape_inferred.om",
            "opset13":        f"{OPTIM_DIR}/resnet50_opset13.om",
            "bn_folded":      f"{OPTIM_DIR}/resnet50_bn_folded.om",
            "channels_last":  f"{OPTIM_DIR}/resnet50_channels_last.om",
            "onnxsim":        f"{OPTIM_DIR}/resnet50_onnxsim.om",
            "onnxoptimizer":  f"{OPTIM_DIR}/resnet50_onnxoptimizer.om",
        },
    },
}

WARMUP = 30
ITERATIONS = 180
NOISE_ROUNDS = 10     # 噪声基底：10轮
VARIANT_ROUNDS = 6    # 优化效果：每变体6轮


def run_child(om_path: str, warmup: int = WARMUP, iterations: int = ITERATIONS) -> Dict:
    """在子进程中执行一轮 benchmark。"""
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "controlled_benchmark.py"),
        "--mode", "child",
        "--model-path", om_path,
        "--warmup", str(warmup),
        "--iterations", str(iterations),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        return {"error": proc.stderr[:2000]}
    for line in reversed(proc.stdout.splitlines()):
        if line.strip().startswith("{"):
            return json.loads(line)
    return {"error": f"no JSON in output: {proc.stdout[:500]}"}


def get_load() -> float:
    try:
        with open("/proc/loadavg") as f:
            return float(f.read().split()[0])
    except:
        return 0.0


def phase1_noise_floor(model_key: str, model_cfg: dict, n: int) -> Tuple[List[Dict], Dict]:
    """Phase 1: 跑同OM N次，标定噪声基底。"""
    baseline_om = model_cfg["baseline_om"]
    results = []

    print(f"\n{'='*70}")
    print(f"[{model_key}] Phase 1: 噪声基底标定 — {baseline_om}")
    print(f"  重复 {n} 次")
    print(f"{'='*70}")

    for i in range(n):
        load_before = get_load()
        t0 = time.time()
        result = run_child(baseline_om)
        elapsed = time.time() - t0
        load_after = get_load()

        entry = {"round": i, "load_before": load_before, "load_after": load_after}

        if "error" in result:
            entry["error"] = result["error"]
            print(f"  [{i+1}/{n}] ❌ {result['error'][:80]}")
        else:
            entry["mean_ms"] = result["mean_ms"]
            entry["std_ms"] = result["std_ms"]
            entry["p50_ms"] = result["p50_ms"]
            entry["p95_ms"] = result["p95_ms"]
            entry["p99_ms"] = result["p99_ms"]
            entry["latencies_ms"] = result["latencies_ms"]
            print(f"  [{i+1}/{n}] mean={result['mean_ms']:.4f}ms  "
                  f"std={result['std_ms']:.4f}  load={load_before:.1f}→{load_after:.1f}  "
                  f"t={elapsed:.1f}s")

        results.append(entry)
        if i < n - 1:
            time.sleep(2)

    # 汇总噪声统计
    means = [r["mean_ms"] for r in results if "mean_ms" in r]
    if len(means) < 2:
        return results, {"error": "insufficient data"}

    noise_stats = {
        "n": len(means),
        "mean_of_means": float(np.mean(means)),
        "std_of_means": float(np.std(means, ddof=1)),
        "min_mean": float(min(means)),
        "max_mean": float(max(means)),
        "range_pct": (max(means) - min(means)) / np.mean(means) * 100,
        "cv_pct": float(np.std(means, ddof=1) / np.mean(means) * 100),
        # 95% 置信区间（t分布）
        "ci_95_half": float(np.std(means, ddof=1) / np.sqrt(len(means)) * 2.262),  # t(9, 0.975)
        "mean_within_run_std": float(np.mean([r.get("std_ms", 0) for r in results if "std_ms" in r])),
    }
    # 最小可检测效应量（效应量 = 噪声标准差 / mean * 100）
    noise_stats["min_detectable_effect_pct"] = noise_stats["ci_95_half"] / noise_stats["mean_of_means"] * 100

    print(f"\n  >>> 噪声基底:")
    print(f"      mean={noise_stats['mean_of_means']:.4f}ms")
    print(f"      between-run std={noise_stats['std_of_means']:.4f}ms")
    print(f"      range={noise_stats['min_mean']:.4f}~{noise_stats['max_mean']:.4f}ms "
          f"({noise_stats['range_pct']:.2f}%)")
    print(f"      95%CI half-width={noise_stats['ci_95_half']:.4f}ms "
          f"({noise_stats['min_detectable_effect_pct']:.2f}%)")
    print(f"      Within-run mean std={noise_stats['mean_within_run_std']:.4f}ms")

    return results, noise_stats


def phase2_optimization_effect(model_key: str, model_cfg: dict, m: int,
                                noise_stats: Dict) -> List[Dict]:
    """Phase 2: 所有变体与基线随机交错跑M轮。"""
    baseline_om = model_cfg["baseline_om"]
    variants = model_cfg["variants"]
    all_oms = {"reexport_plain": baseline_om, **variants}
    all_names = list(all_oms.keys())

    print(f"\n{'='*70}")
    print(f"[{model_key}] Phase 2: 优化效果测量 — {len(variants)} 变体 × {m} 轮")
    print(f"{'='*70}")

    results = []

    for round_idx in range(m):
        order = list(all_names)
        random.shuffle(order)
        print(f"\n  Round {round_idx+1}/{m} — Order: {' → '.join(order)}")

        for variant in order:
            om_path = all_oms[variant]
            load_before = get_load()
            t0 = time.time()
            result = run_child(om_path)
            elapsed = time.time() - t0
            load_after = get_load()

            entry = {
                "variant": variant,
                "round": round_idx,
                "load_before": load_before,
                "load_after": load_after,
            }

            if "error" in result:
                entry["error"] = result["error"]
                print(f"    {variant:20s} ❌ {result['error'][:60]}")
            else:
                entry["mean_ms"] = result["mean_ms"]
                entry["std_ms"] = result["std_ms"]
                entry["p50_ms"] = result["p50_ms"]
                entry["p95_ms"] = result["p95_ms"]
                entry["p99_ms"] = result["p99_ms"]
                entry["latencies_ms"] = result["latencies_ms"]
                print(f"    {variant:20s}: {result['mean_ms']:.4f}ms  "
                      f"std={result['std_ms']:.4f}  "
                      f"load={load_before:.1f}→{load_after:.1f}")

            results.append(entry)
            time.sleep(1.5)

        if round_idx < m - 1:
            print(f"    -- 轮间冷却 5s --")
            time.sleep(5)

    return results


def analyze_phase2(model_key: str, results: List[Dict], noise_stats: Dict) -> Dict:
    """统计分析 Phase 2 结果。"""
    by_variant = defaultdict(list)
    for r in results:
        if "mean_ms" in r:
            by_variant[r["variant"]].append(r)

    ref_key = "reexport_plain"
    ref_data = by_variant[ref_key]
    if not ref_data:
        return {"error": "no baseline data"}

    ref_means = np.array([r["mean_ms"] for r in ref_data])
    ref_mean = np.mean(ref_means)

    analysis = {
        "baseline": ref_key,
        "baseline_mean_ms": round(ref_mean, 4),
        "noise_baseline": {
            "between_run_std": noise_stats.get("std_of_means", 0),
            "min_detectable_effect_pct": noise_stats.get("min_detectable_effect_pct", 0),
            "n_noise_rounds": noise_stats.get("n", 0),
        },
        "variants": {},
    }

    print(f"\n{'='*70}")
    print(f"[{model_key}] Phase 2 统计结果")
    print(f"  基线 (reexport_plain): {ref_mean:.4f}ms ({len(ref_data)} runs)")
    print(f"  噪声基底 (between-run std): {noise_stats.get('std_of_means', 0):.4f}ms")
    print(f"  最小可检测效应量 (95%CI): {noise_stats.get('min_detectable_effect_pct', 0):.2f}%")
    print()
    print(f"{'变体':20s} {'Mean':>8s} {'Diff(%)':>9s} {'Std':>7s} {'CV':>5s} {'p95':>8s} {'p99':>8s} {'p值':>9s} {'显著?':>5s}")
    print("-" * 80)

    for variant in sorted(by_variant.keys()):
        if variant == ref_key:
            continue
        runs = by_variant[variant]
        means = np.array([r["mean_ms"] for r in runs])
        n = len(means)
        avg = np.mean(means)
        std = np.std(means, ddof=1)
        diff_pct = (avg - ref_mean) / ref_mean * 100
        p50 = np.mean([r.get("p50_ms", 0) for r in runs])
        p95 = np.mean([r.get("p95_ms", 0) for r in runs])
        p99 = np.mean([r.get("p99_ms", 0) for r in runs])
        cv = std / avg * 100

        # t检验：变体 vs 基线（各轮均值的独立样本）
        from scipy import stats as sp_stats
        try:
            t_stat, p_value = sp_stats.ttest_ind(means, ref_means, equal_var=False)
        except Exception:
            t_stat, p_value = 0, 1.0

        # 使用噪声基底而非ref_means的std来判断显著性
        noise_std = noise_stats.get("std_of_means", 0)
        # 如果p值是在0.05以下且效应量>噪声基底
        is_significant = p_value < 0.05
        # 额外检查：效应量是否超过最小可检测效应量
        effect_exceeds_noise = abs(diff_pct) > noise_stats.get("min_detectable_effect_pct", 999)

        marker = ""
        if is_significant:
            marker = " ✅" if diff_pct < 0 else " ❌"

        print(f"{variant:20s} {avg:8.4f} {diff_pct:+8.2f}% {std:6.4f} {cv:4.2f}% "
              f"{p95:7.4f} {p99:7.4f} {p_value:8.4f} {'Y*' if is_significant and effect_exceeds_noise else 'Y' if is_significant else 'N'}{marker}")

        analysis["variants"][variant] = {
            "mean_ms": round(avg, 4),
            "std_ms": round(std, 4),
            "diff_vs_baseline_pct": round(diff_pct, 2),
            "p95_ms": round(p95, 4),
            "p99_ms": round(p99, 4),
            "cv_pct": round(cv, 2),
            "n_runs": n,
            "p_value": round(p_value, 4),
            "significant_p05": is_significant,
            "exceeds_noise_floor": effect_exceeds_noise,
            "effect_size_sigma": round(diff_pct / noise_stats.get("min_detectable_effect_pct", 1), 2)
                if noise_stats.get("min_detectable_effect_pct", 0) > 0 else 0,
        }

    return analysis


def run_model(model_key: str):
    """在单个模型上运行完整两阶段实验。"""
    cfg = MODELS[model_key]

    # Phase 1
    noise_raw, noise_stats = phase1_noise_floor(model_key, cfg, NOISE_ROUNDS)

    # Phase 2
    phase2_raw = phase2_optimization_effect(model_key, cfg, VARIANT_ROUNDS, noise_stats)

    # 分析
    analysis = analyze_phase2(model_key, phase2_raw, noise_stats)

    # 保存（将numpy类型转为Python原生类型）
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    out = {
        "model_key": model_key,
        "config": {
            "warmup": WARMUP,
            "iterations": ITERATIONS,
            "noise_rounds": NOISE_ROUNDS,
            "variant_rounds": VARIANT_ROUNDS,
        },
        "noise_floor": noise_stats,
        "noise_raw": noise_raw,
        "analysis": analysis,
        "phase2_raw": phase2_raw,
    }
    out_path = RESULTS_DIR / f"rigorous_experiment_{model_key}_{ts}.json"
    out_path.write_text(json.dumps(out, indent=2, cls=NumpyEncoder))
    print(f"\n结果已保存: {out_path}")

    return analysis


def format_explanations():
    """为每种优化生成因果解释模板（在收集数据后人工补充ONNX分析）。"""
    explanations = {}

    # 预加载 ONNX 差异分析
    diff_path = RESULTS_DIR / "onnx_diff_analysis.json"
    onnx_data = {}
    if diff_path.exists():
        onnx_data = json.loads(diff_path.read_text())

    for model_key in ["mobilenet_v3_small", "resnet50"]:
        model_onnx = onnx_data.get(model_key, {})
        original = model_onnx.get("original", {})

        print(f"\n{'='*70}")
        print(f"[{model_key}] 每种优化的因果解释")
        print(f"{'='*70}")

        for variant, info in model_onnx.get("diffs", {}).items():
            print(f"\n  --- {variant} ---")
            if not info:
                print(f"  ONNX 与原始完全相同 → 差异只能来自 ATC 编译行为")

                if "fp16" in variant:
                    print(f"  补充：fp16 的 ONNX 权重为半精度（文件大小减半）")
                    print(f"        但 ONNX 图结构不变")
                continue

            # 列出所有 ONNX 差异
            for diff_key, diff_val in info.items():
                print(f"  {diff_key}: {diff_val}")

            # 推断可能的影响
            if "total_nodes" in info:
                delta = info["total_nodes"].get("diff", 0)
                if delta < 0:
                    print(f"  → 节点减少 {abs(delta)} 个，但减少的可能是融合后的中间节点")
            if "op_count_diffs" in info:
                for op, diff in info["op_count_diffs"].items():
                    d = diff.get("diff", 0)
                    if d != 0:
                        print(f"  → {op}: {d:+d} 个")

    return explanations


def main():
    print(f"科学严谨的两阶段优化效果评估")
    print(f"时间: {datetime.now().isoformat()}")
    print(f"系统负载基线: {get_load():.2f}")
    print(f"Noise rounds: {NOISE_ROUNDS}, Variant rounds: {VARIANT_ROUNDS}")
    print(f"Warmup: {WARMUP}, Iterations: {ITERATIONS}")

    # 先跑 MobileNet（更快）
    print(f"\n\n{'#'*70}")
    print(f"{'#'*70}")
    print(f"{'#'}  第一阶段: MobileNetV3-Small")
    print(f"{'#'*70}")
    print(f"{'#'*70}")
    mob_analysis = run_model("mobilenet_v3_small")

    # 再跑 ResNet
    print(f"\n\n{'#'*70}")
    print(f"{'#'*70}")
    print(f"{'#'}  第二阶段: ResNet50")
    print(f"{'#'*70}")
    print(f"{'#'*70}")
    res_analysis = run_model("resnet50")

    # 输出最终对比
    print(f"\n\n{'='*70}")
    print(f"最终汇总")
    print(f"{'='*70}")

    for name, analysis in [("MobileNetV3-Small", mob_analysis), ("ResNet50", res_analysis)]:
        print(f"\n--- {name} ---")
        print(f"  基线 (reexport_plain): {analysis['baseline_mean_ms']:.4f}ms")
        print(f"  噪声基底 (between-run): {analysis.get('noise_baseline', {}).get('between_run_std', '?'):.4f}ms")
        print(f"  最小可检测效应量: {analysis.get('noise_baseline', {}).get('min_detectable_effect_pct', '?'):.2f}%")
        print(f"\n  变体排序:")
        var_data = analysis.get("variants", {})
        sorted_vars = sorted(var_data.items(), key=lambda x: x[1]["mean_ms"])
        for v, s in sorted_vars:
            sig = "✅" if s.get("significant_p05") and s.get("exceeds_noise_floor") else \
                  "⚠️" if s.get("significant_p05") else "—"
            print(f"    {v:20s}: {s['mean_ms']:.4f}ms ({s['diff_vs_baseline_pct']:+.2f}%) {sig}")

    # 生成解释
    format_explanations()

    print(f"\n{'='*70}")
    print(f"实验完成")


if __name__ == "__main__":
    main()
