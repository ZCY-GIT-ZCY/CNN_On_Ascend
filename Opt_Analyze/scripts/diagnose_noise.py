"""
噪声诊断实验 - 测量基准噪声水平 vs 变体间差异。

核心问题：
  1. channels_last 的 ONNX 与原版是否字节相同？
  2. 同一个 OM 跑多次，测量变异系数 (CV) 是多少？
  3. 编译器 (ATC) 是否为确定性的？同 ONNX 编译的 OM 是否相同？
  4. 基线噪声 vs 变体差异：哪个更大？

这直接回答：性能差异来自真实优化效果，还是仅来自环境噪声？
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
OPT_ANALYZE = SCRIPT_DIR.parent
COMMON_DIR = OPT_ANALYZE.parent / "common" / "acllite_utils"
OPTIMIZATION_DIR = OPT_ANALYZE.parent / "Optimization"
MODELS_DIR = OPTIMIZATION_DIR / "models"
MOBILENET_DIR = OPT_ANALYZE.parent / "MobileNet"
RESNET_DIR = OPT_ANALYZE.parent / "ResNet"

RESULTS_DIR = OPT_ANALYZE / "data"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def run_benchmark(om_path: Path, warmup: int = 30, iterations: int = 120) -> Dict:
    """在子进程中执行一次 benchmark。"""
    cmd = [
        sys.executable,
        str(OPT_ANALYZE / "scripts" / "controlled_benchmark.py"),
        "--mode", "child",
        "--model-path", str(om_path),
        "--warmup", str(warmup),
        "--iterations", str(iterations),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        return {"error": proc.stderr[:2000]}

    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {"error": "no JSON output"}


def main():
    print("=" * 60)
    print("噪声诊断实验")
    print("=" * 60)

    model_key = "mobilenet_v3_small"  # 先用 MobileNet

    # 1. ONNX 字节对比
    print("\n--- 1. ONNX 字节相同性检查 ---")
    original_onnx = MOBILENET_DIR / "mobilenetv3.onnx"
    channels_last_onnx = MODELS_DIR / f"{model_key}_channels_last.onnx"
    onnxsim_onnx = MODELS_DIR / f"{model_key}_onnxsim.onnx"
    opset13_onnx = MODELS_DIR / f"{model_key}_opset13.onnx"

    checks = {}
    for name, path in [("original", original_onnx),
                       ("channels_last", channels_last_onnx),
                       ("onnxsim", onnxsim_onnx),
                       ("opset13", opset13_onnx)]:
        if path.exists():
            h = md5(path)
            checks[name] = {"path": str(path), "md5": h, "size": path.stat().st_size}
            print(f"  {name:20s}: MD5={h}, size={path.stat().st_size}")

    # 看哪些与原始 ONNX 字节相同
    orig_md5 = checks.get("original", {}).get("md5", "")
    for name, info in checks.items():
        if name != "original":
            same = "✅ 字节相同" if info["md5"] == orig_md5 else "❌ 不同"
            print(f"  >> {name} vs original: {same}")

    # 2. OM 字节对比
    print("\n--- 2. OM 字节相同性检查 ---")
    baseline_om = MOBILENET_DIR / "mobilenetv3_aipp.om"
    baseline_copy_om = MODELS_DIR / f"{model_key}_baseline.om"
    channels_last_om = MODELS_DIR / f"{model_key}_channels_last.om"
    reexport_plain_om = MODELS_DIR / f"{model_key}_reexport_plain.om"

    om_checks = {}
    for name, path in [("原始 AIPP OM", baseline_om),
                       ("baseline_copy", baseline_copy_om),
                       ("channels_last", channels_last_om),
                       ("reexport_plain", reexport_plain_om)]:
        if path.exists():
            h = md5(path)
            om_checks[name] = {"path": str(path), "md5": h, "size_mb": path.stat().st_size / 1e6}
            print(f"  {name:20s}: MD5={h}, size={path.stat().st_size/1e6:.2f} MB")

    # 3. 噪声水平测量：同一个 OM 跑 5 次
    print("\n--- 3. 噪声水平：同一 OM 重复 5 次 ---")
    om_to_test = baseline_om  # 用原始 AIPP OM
    noise_results = []
    for i in range(5):
        result = run_benchmark(om_to_test, warmup=30, iterations=120)
        if "error" in result:
            print(f"  Run {i+1}: ERROR - {result['error']}")
        else:
            noise_results.append(result)
            print(f"  Run {i+1}: mean={result['mean_ms']:.4f}ms, "
                  f"std={result['std_ms']:.4f}ms, "
                  f"min={result['min_ms']:.4f}, max={result['max_ms']:.4f}")

    if noise_results:
        means = [r["mean_ms"] for r in noise_results]
        cv = (max(means) - min(means)) / (sum(means) / len(means)) * 100
        print(f"\n  >>> 噪声水平:")
        print(f"      Mean across runs: {sum(means)/len(means):.4f}ms")
        print(f"      Range: {min(means):.4f} - {max(means):.4f}ms")
        print(f"      CV (range/mean): {cv:.2f}%")

    # 4. 如果 channels_last 的 ONNX 与原始相同
    #    则比较 baseline_om 与 channels_last_om 是否相同
    print("\n--- 4. 编译器确定性检查 ---")
    if checks.get("channels_last", {}).get("md5") == checks.get("original", {}).get("md5"):
        print("  channels_last ONNX = 原始 ONNX (字节级)")
        if om_checks.get("channels_last", {}).get("md5") != om_checks.get("原始 AIPP OM", {}).get("md5"):
            print("  但 OM 不同！说明从同ONNX编译出的OM被不同方式处理过")
            om_diff = om_checks.get("channels_last", {}).get("size_mb", 0) - om_checks.get("原始 AIPP OM", {}).get("size_mb", 0)
            print(f"  OM 大小差异: {om_diff:.2f} MB")
    else:
        print("  channels_last ONNX ≠ 原始 ONNX")

    # 检查 reexport_plain 和 baseline_copy 的 OM
    if all(k in om_checks for k in ["原始 AIPP OM", "baseline_copy"]):
        if om_checks["原始 AIPP OM"]["md5"] == om_checks["baseline_copy"]["md5"]:
            print("  baseline_copy (复制品) = 原始 AIPP OM ✅")
        else:
            print("  baseline_copy (复制品) ≠ 原始 AIPP OM ❌ (异常！)")

    # 5. 保存所有结果
    result = {
        "onnx_byte_checks": checks,
        "om_byte_checks": om_checks,
        "noise_runs": noise_results,
        "noise_summary": {
            "mean_ms": sum(r["mean_ms"] for r in noise_results) / len(noise_results) if noise_results else None,
            "min_ms": min(r["mean_ms"] for r in noise_results) if noise_results else None,
            "max_ms": max(r["mean_ms"] for r in noise_results) if noise_results else None,
            "cv_pct": cv if noise_results else None,
        } if noise_results else None,
    }

    out_path = RESULTS_DIR / "noise_diagnosis.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
