"""
公平对比实验：以 reexport_plain（当前ATC重新编译）为统一基线。

所有变体共享同一编译链路，排除"历史OM vs 当前OM"的偏差。
"""
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
OPT_ANALYZE = SCRIPT_DIR.parent

RESULTS_DIR = OPT_ANALYZE / "data"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# MobileNet 所有变体的 OM
OMS = {
    "reexport_plain": "/home/HwHiAiUser/Desktop/CNN/Optimization/models/mobilenet_v3_small_reexport_plain.om",
    "fp16":           "/home/HwHiAiUser/Desktop/CNN/Optimization/models/mobilenet_v3_small_fp16.om",
    "shape_inferred": "/home/HwHiAiUser/Desktop/CNN/Optimization/models/mobilenet_v3_small_shape_inferred.om",
    "opset13":        "/home/HwHiAiUser/Desktop/CNN/Optimization/models/mobilenet_v3_small_opset13.om",
    "bn_folded":      "/home/HwHiAiUser/Desktop/CNN/Optimization/models/mobilenet_v3_small_bn_folded.om",
    "channels_last":  "/home/HwHiAiUser/Desktop/CNN/Optimization/models/mobilenet_v3_small_channels_last.om",
    "onnxsim":        "/home/HwHiAiUser/Desktop/CNN/Optimization/models/mobilenet_v3_small_onnxsim.om",
    "onnxoptimizer":  "/home/HwHiAiUser/Desktop/CNN/Optimization/models/mobilenet_v3_small_onnxoptimizer.om",
}

WARMUP = 30
ITERATIONS = 180
ROUNDS = 5  # 5 rounds × 8 variants ≈ 30分钟


def run_child(om_path: str) -> Dict:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "controlled_benchmark.py"),
        "--mode", "child",
        "--model-path", om_path,
        "--warmup", str(WARMUP),
        "--iterations", str(ITERATIONS),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        return {"error": proc.stderr[:2000]}
    for line in reversed(proc.stdout.splitlines()):
        if line.strip().startswith("{"):
            return json.loads(line)
    return {"error": "no JSON"}


def get_load() -> float:
    try:
        with open("/proc/loadavg") as f:
            return float(f.read().split()[0])
    except:
        return 0.0


def main():
    print("=" * 70)
    print("公平对比实验：全部变体 vs reexport_plain（统一ATC编译基线）")
    print("=" * 70)
    print(f"Variants: {len(OMS)}")
    print(f"Rounds: {ROUNDS}, Iterations: {ITERATIONS}, Warmup: {WARMUP}")
    print()

    all_data = []

    for round_idx in range(ROUNDS):
        order = list(OMS.keys())
        random.shuffle(order)
        print(f"\n--- Round {round_idx+1}/{ROUNDS} ---")
        print(f"    Order: {' → '.join(order)}")

        round_results = []
        for variant in order:
            om_path = OMS[variant]
            load_before = get_load()
            t0 = time.time()
            result = run_child(om_path)
            elapsed = time.time() - t0
            load_after = get_load()

            entry = {
                "variant": variant, "round": round_idx,
                "load_before": load_before, "load_after": load_after,
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

            round_results.append(entry)
            time.sleep(1.5)  # 变体间冷却

        all_data.extend(round_results)

        if round_idx < ROUNDS - 1:
            print(f"    冷却 5s...")
            time.sleep(5)

    # ===== 汇总分析 =====
    print("\n" + "=" * 70)
    print("汇总分析")
    print("=" * 70)

    by_variant = defaultdict(list)
    for d in all_data:
        if "mean_ms" in d:
            by_variant[d["variant"]].append(d)

    # 以 reexport_plain 为基线
    ref_key = "reexport_plain"
    ref_runs = by_variant[ref_key]
    if not ref_runs:
        print("ERROR: no reexport_plain data")
        return
    ref_mean = np.mean([r["mean_ms"] for r in ref_runs])

    print(f"\n基准: {ref_key} = {ref_mean:.4f} ms ({len(ref_runs)} rounds)")
    print(f"{'变体':20s} {'Mean(ms)':>10s} {'vs基准':>10s} {'Std(ms)':>8s} {'CV':>6s} {'P50':>8s} {'P95':>8s}")
    print("-" * 72)

    results_summary = {}
    for variant in sorted(OMS.keys()):
        runs = by_variant[variant]
        if not runs:
            continue
        means = [r["mean_ms"] for r in runs]
        avg = np.mean(means)
        std = np.std(means)
        diff_pct = (avg - ref_mean) / ref_mean * 100
        p50 = np.mean([r["p50_ms"] for r in runs])
        p95 = np.mean([r["p95_ms"] for r in runs])
        cv = std / avg * 100 if avg > 0 else 0

        # 显著性
        if len(runs) >= 2:
            se = np.std(means) / np.sqrt(len(means))
            sigma = (avg - ref_mean) / se if se > 0 else 0
        else:
            sigma = 0

        marker = "" if variant == ref_key else (" ✅" if diff_pct < 0 else "")
        print(f"{variant:20s} {avg:8.4f}  {diff_pct:+8.2f}%  {std:6.4f}  {cv:5.2f}% {p50:7.4f} {p95:7.4f}{marker}")

        results_summary[variant] = {
            "mean_ms": round(avg, 4),
            "vs_reexport_plain_pct": round(diff_pct, 2),
            "std_ms": round(std, 4),
            "cv_pct": round(cv, 2),
            "sigma": round(sigma, 1),
            "n_runs": len(runs),
        }

    print("\n" + "=" * 70)
    # 哪个变体真正比"公平基线"好？
    print("\n对比 reexport_plain（统一编译链路基线）：")
    sorted_variants = sorted(results_summary.items(), key=lambda x: x[1]["mean_ms"])
    for v, s in sorted_variants:
        if v == ref_key:
            continue
        if s["sigma"] >= 2:
            print(f"  {v:20s}: {s['mean_ms']:.4f}ms ({s['vs_reexport_plain_pct']:+.2f}%, {s['sigma']:.0f}σ) "
                  f"{'✅ 显著不同' if s['vs_reexport_plain_pct'] < 0 else '❌ 显著更慢'}")
        else:
            print(f"  {v:20s}: {s['mean_ms']:.4f}ms ({s['vs_reexport_plain_pct']:+.2f}%, {s['sigma']:.0f}σ) "
                  f"⚠️ 差异不显著")

    # 保存
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"fair_comparison_{ts}.json"
    out_path.write_text(json.dumps({
        "config": {"rounds": ROUNDS, "iterations": ITERATIONS, "warmup": WARMUP},
        "baseline": ref_key,
        "baseline_mean_ms": ref_mean,
        "results": results_summary,
        "raw_data": all_data,
    }, indent=2))
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
