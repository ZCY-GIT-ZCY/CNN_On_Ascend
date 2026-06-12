"""
Benchmark ATC parameter variants: 5 rounds each, sequential.
"""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np

SCRIPT = Path(__file__).resolve().parent
MODELS = Path("/home/HwHiAiUser/Desktop/CNN/Optimization/models")

OMS = {
    "reexport_plain": str(MODELS / "mobilenet_v3_small_reexport_plain.om"),
    "fp16_variant":   str(MODELS / "mobilenet_v3_small_fp16.om"),
    "force_fp16":     str(MODELS / "mobilenet_v3_small_atc_fp16.om"),
    "l1_optimize":    str(MODELS / "mobilenet_v3_small_atc_l1.om"),
}

def bench(om_path):
    cmd = [sys.executable, str(SCRIPT / "controlled_benchmark.py"),
           "--mode=child", f"--model-path={om_path}", "--warmup=30", "--iterations=180"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    for l in reversed(r.stdout.splitlines()):
        if l.strip().startswith("{"):
            return json.loads(l)
    return {"error": r.stderr[:200]}

print(f"{'变体':20s} {'Mean(ms)':>9s} {'Std':>7s} {'CV':>5s} {'Rounds':>7s}")
print("-"*50)

all_results = {}
for name, path in OMS.items():
    size = Path(path).stat().st_size
    means = []
    for r in range(5):
        res = bench(path)
        if "mean_ms" in res:
            means.append(res["mean_ms"])
        time.sleep(1.5)

    if means:
        avg, std = np.mean(means), np.std(means, ddof=1)
        cv = std/avg*100
        print(f"{name:20s} {avg:9.4f} {std:7.4f} {cv:4.1f}% {len(means):3d}  ({size/1e6:.2f}MB)")
        all_results[name] = {"mean": round(avg,4), "std": round(std,4), "cv": round(cv,1), "size_mb": round(size/1e6,2)}

print("\n=== 对比 (vs reexport_plain) ===")
ref = all_results.get("reexport_plain", {}).get("mean", 0)
for name, data in all_results.items():
    if name == "reexport_plain": continue
    diff = (data["mean"] - ref) / ref * 100
    print(f"  {name:20s}: {data['mean']:.4f}ms ({diff:+.2f}%)")

print("\nDone.")
