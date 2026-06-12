"""
编译链路差异 vs 噪声：受控 A/B 实验。

比较三个 OM：
  A = baseline_copy（原始 AIPP OM，历史编译）
  B = reexport_plain（同ONNX，当前ATC重新编译）
  C = fp16（最接近基线的变体）

设计：每轮三个OM随机顺序，共N轮，子进程隔离。
"""
import json
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
OPT_ANALYZE = SCRIPT_DIR.parent
COMMON_DIR = OPT_ANALYZE.parent / "common" / "acllite_utils"

RESULTS_DIR = OPT_ANALYZE / "data"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OMS = {
    "baseline_copy": "/home/HwHiAiUser/Desktop/CNN/Optimization/models/mobilenet_v3_small_baseline.om",
    "reexport_plain": "/home/HwHiAiUser/Desktop/CNN/Optimization/models/mobilenet_v3_small_reexport_plain.om",
    "fp16": "/home/HwHiAiUser/Desktop/CNN/Optimization/models/mobilenet_v3_small_fp16.om",
}

WARMUP = 30
ITERATIONS = 180
ROUNDS = 8


def run_child(om_path: str, warmup: int, iterations: int) -> Dict:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "controlled_benchmark.py"),
        "--mode", "child",
        "--model-path", om_path,
        "--warmup", str(warmup),
        "--iterations", str(iterations),
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
    print("=" * 60)
    print("编译链路差异 vs 噪声 受控实验")
    print("=" * 60)
    print(f"OMs: {list(OMS.keys())}")
    print(f"Rounds: {ROUNDS}, Iterations: {ITERATIONS}, Warmup: {WARMUP}")
    print(f"Start: {datetime.now().isoformat()}")
    print(f"Initial load: {get_load():.2f}")
    print()

    rounds_data = []

    for round_idx in range(ROUNDS):
        order = list(OMS.keys())
        random.shuffle(order)
        print(f"\n--- Round {round_idx + 1}/{ROUNDS} ---")
        print(f"    Order: {' → '.join(order)}")
        print(f"    Load before: {get_load():.2f}")

        round_results = {"round": round_idx, "order": order, "runs": []}

        for variant in order:
            om_path = OMS[variant]
            load_before = get_load()
            t0 = time.time()

            result = run_child(om_path, warmup=WARMUP, iterations=ITERATIONS)

            elapsed = time.time() - t0
            load_after = get_load()

            if "error" in result:
                print(f"    {variant:20s}: ERROR - {result['error']}")
                round_results["runs"].append({
                    "variant": variant, "error": result["error"],
                    "load_before": load_before, "load_after": load_after
                })
            else:
                print(f"    {variant:20s}: mean={result['mean_ms']:.4f}ms  "
                      f"std={result['std_ms']:.4f}  "
                      f"p50={result['p50_ms']:.4f}  "
                      f"load={load_before:.1f}→{load_after:.1f}  "
                      f"time={elapsed:.1f}s")
                round_results["runs"].append({
                    "variant": variant,
                    "mean_ms": result["mean_ms"],
                    "std_ms": result["std_ms"],
                    "p50_ms": result["p50_ms"],
                    "p95_ms": result["p95_ms"],
                    "p99_ms": result["p99_ms"],
                    "min_ms": result["min_ms"],
                    "max_ms": result["max_ms"],
                    "latencies_ms": result["latencies_ms"],
                    "load_before": load_before,
                    "load_after": load_after,
                })

            # 每个变体间稍微冷却
            time.sleep(2)

        rounds_data.append(round_results)

        # 轮间冷却
        if round_idx < ROUNDS - 1:
            cool = 5
            print(f"    Cooling {cool}s...")
            time.sleep(cool)

    # ===== 汇总分析 =====
    print("\n" + "=" * 60)
    print("汇总分析")
    print("=" * 60)

    # 按变体分组
    from collections import defaultdict
    by_variant = defaultdict(list)
    for rd in rounds_data:
        for run in rd["runs"]:
            if "mean_ms" in run:
                by_variant[run["variant"]].append(run)

    for variant in ["baseline_copy", "reexport_plain", "fp16"]:
        runs = by_variant[variant]
        if not runs:
            continue
        means = [r["mean_ms"] for r in runs]
        stds = [r["std_ms"] for r in runs]
        loads = [r["load_before"] for r in runs]

        print(f"\n{variant}:")
        print(f"  Mean ± std:  {sum(means)/len(means):.4f} ± {__import__('numpy').std(means):.4f} ms")
        print(f"  Range:       {min(means):.4f} ~ {max(means):.4f} ms")
        print(f"  CV:          {(max(means)-min(means))/(sum(means)/len(means))*100:.2f}%")
        print(f"  System load: {min(loads):.1f} ~ {max(loads):.1f}")

    # 核心问题：reexport_plain vs baseline
    bl_means = [r["mean_ms"] for r in by_variant["baseline_copy"]]
    rp_means = [r["mean_ms"] for r in by_variant["reexport_plain"]]
    fp_means = [r["mean_ms"] for r in by_variant["fp16"]]

    if bl_means and rp_means:
        bl_avg = sum(bl_means) / len(bl_means)
        rp_avg = sum(rp_means) / len(rp_means)
        fp_avg = sum(fp_means) / len(fp_means)
        print(f"\n{'='*60}")
        print(f"核心结论：")
        print(f"  baseline_copy:    {bl_avg:.4f} ms")
        print(f"  reexport_plain:   {rp_avg:.4f} ms  ({((rp_avg-bl_avg)/bl_avg*100):+.2f}%)")
        print(f"  fp16:             {fp_avg:.4f} ms  ({((fp_avg-bl_avg)/bl_avg*100):+.2f}%)")

        # 显著性检查：2-sigma
        import numpy as np
        bl_std = np.std(bl_means)
        rp_std = np.std(rp_means)
        diff = rp_avg - bl_avg
        combined_std = (bl_std**2 + rp_std**2)**0.5 / len(bl_means)**0.5
        if combined_std > 0:
            sigma = diff / combined_std
            threshold = 2
            print(f"  差异显著性: {sigma:.1f}σ (>={threshold}σ则不太可能是噪声)")
        else:
            print(f"  差异显著性: N/A (std=0)")

    # 保存
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"compilation_comparison_{ts}.json"
    out_path.write_text(json.dumps({
        "config": {
            "rounds": ROUNDS,
            "iterations": ITERATIONS,
            "warmup": WARMUP,
        },
        "oms": OMS,
        "rounds_data": rounds_data,
    }, indent=2))
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
