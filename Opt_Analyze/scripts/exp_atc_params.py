"""
实验A: ATC编译参数调优（利用已有OM，只编译缺少的）
"""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
OPT_ANALYZE = SCRIPT_DIR.parent
RESULTS_DIR = OPT_ANALYZE / "data"
CNN_ROOT = OPT_ANALYZE.parent
OPTIM_DIR = CNN_ROOT / "Optimization"
MODELS_DIR = OPTIM_DIR / "models"

ATC_BIN = "/usr/local/Ascend/ascend-toolkit/latest/bin/atc"
AIPP_CFGS = {
    "mobilenet_v3_small": str(CNN_ROOT / "MobileNet" / "aipp.cfg"),
    "resnet50": str(CNN_ROOT / "ResNet" / "aipp.cfg"),
}
ONNX_FILES = {
    "mobilenet_v3_small": str(CNN_ROOT / "MobileNet" / "mobilenetv3.onnx"),
    "resnet50": str(CNN_ROOT / "ResNet" / "resnet50.onnx"),
}
INPUT_SHAPE = "input_image:1,3,224,224"

WARMUP = 30
ITERATIONS = 180
ROUNDS = 4  # 每个OM 4轮

# 优先使用的已有OM（避免重新编译）
EXISTING_OMS = {
    "mobilenet_v3_small": {
        "allow_mix_precision_l2_optimize": str(MODELS_DIR / "mobilenet_v3_small_reexport_plain.om"),
        # force_fp16 可以用 fp16 ONNX 重新编译
        # cube_fp16in_fp32out 需要重新编译
        # l1_optimize / off_optimize 需要重新编译
    },
    "resnet50": {
        "allow_mix_precision_l2_optimize": str(MODELS_DIR / "resnet50_reexport_plain.om"),
    },
}

# 要测试的参数组合
PARAM_COMBO = [
    # (label, precision_mode, buffer_optimize)
    ("baseline",      "allow_mix_precision", "l2_optimize"),    # 基线
    ("fp16_only",     "force_fp16",          "l2_optimize"),    # 仅改precision
    ("cube_fp16out",  "cube_fp16in_fp32out", "l2_optimize"),    # Cube fp16算，输出fp32
    ("l1_opt",        "allow_mix_precision", "l1_optimize"),    # 仅改buffer
    ("off_opt",       "allow_mix_precision", "off_optimize"),   # 无buffer优化（对照）
    ("fp16_l1",       "force_fp16",          "l1_optimize"),    # 双重优化
]


def get_om_path(model_key: str, label: str, pm: str, bo: str) -> str:
    """获取OM路径（优先用已有OM，否则构造新路径）。"""
    tag = f"{pm}_{bo}"
    # 检查是否有已有OM可用
    existing = EXISTING_OMS.get(model_key, {}).get(tag)
    if existing and Path(existing).exists():
        return existing
    # 否则走新编译路径
    return str(MODELS_DIR / f"{model_key}_atc_{tag}.om")


def compile_if_needed(model_key: str, om_path: str, pm: str, bo: str) -> Tuple[bool, str]:
    """如果OM不存在则编译。"""
    if Path(om_path).exists():
        return True, om_path

    onnx = ONNX_FILES[model_key]
    aipp = AIPP_CFGS[model_key]
    output_prefix = om_path.replace(".om", "")

    print(f"    编译 {pm} + {bo}...", end=" ", flush=True)
    cmd = [
        ATC_BIN, f"--model={onnx}", "--framework=5",
        f"--output={output_prefix}", "--soc_version=Ascend310B1",
        "--input_format=NCHW", f"--input_shape={INPUT_SHAPE}",
        f"--precision_mode={pm}", f"--buffer_optimize={bo}",
        "--op_select_implmode=high_performance",
        "--enable_small_channel=1",
        f"--insert_op_conf={aipp}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if proc.returncode == 0 and Path(om_path).exists():
        print(f"✅ {Path(om_path).stat().st_size/1e6:.2f}MB")
        return True, om_path
    else:
        print(f"❌")
        return False, proc.stderr[-500:]


def run_benchmark(om_path: str) -> Dict:
    """子进程 benchmark。"""
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "controlled_benchmark.py"),
        "--mode", "child", "--model-path", om_path,
        "--warmup", str(WARMUP), "--iterations", str(ITERATIONS),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        return {"error": proc.stderr[:1000]}
    for line in reversed(proc.stdout.splitlines()):
        if line.strip().startswith("{"):
            return json.loads(line)
    return {"error": "no JSON"}


def run_model(model_key: str):
    """跑单个模型的所有参数组合。"""
    print(f"\n{'='*70}")
    print(f"模型: {model_key}")
    print(f"{'='*70}")

    results = []

    for label, pm, bo in PARAM_COMBO:
        om_path = get_om_path(model_key, label, pm, bo)
        print(f"\n  [{label:15s}] {pm:30s} + {bo:15s}")

        # 编译（如果需要）
        ok, msg = compile_if_needed(model_key, om_path, pm, bo)
        if not ok:
            print(f"  → ❌ 编译失败: {msg[:100]}")
            continue

        om_size = Path(om_path).stat().st_size

        # Benchmark ROUNDS次
        means = []
        all_stats = []
        for r in range(ROUNDS):
            result = run_benchmark(om_path)
            if "error" in result:
                print(f"    Round {r+1}: ❌ {result['error'][:60]}")
                continue
            means.append(result["mean_ms"])
            all_stats.append(result)

        if means:
            avg = np.mean(means)
            std = np.std(means, ddof=1)
            cv = std / avg * 100
            print(f"    → {avg:.4f}ms ±{std:.4f} (CV={cv:.2f}%, size={om_size/1e6:.2f}MB)")
            results.append({
                "label": label, "precision_mode": pm, "buffer_optimize": bo,
                "om_path": om_path, "om_size_mb": round(om_size/1e6, 4),
                "mean_ms": round(avg, 4), "std_ms": round(std, 4),
                "cv_pct": round(cv, 2),
                "round_means": [round(m, 4) for m in means],
                "n_rounds": len(means),
            })
        else:
            print(f"    → ❌ 所有轮次失败")

        if label != "baseline":
            time.sleep(1.5)  # 冷却

    # 汇总
    print(f"\n  --- 参数调优汇总 ({model_key}) ---")
    ref = [r for r in results if r["label"] == "baseline"]
    if ref:
        ref_mean = ref[0]["mean_ms"]
        print(f"  基线 (current): {ref_mean:.4f}ms")
        for r in results:
            if r["label"] == "baseline":
                continue
            diff = (r["mean_ms"] - ref_mean) / ref_mean * 100
            # 简单显著性判断（超过噪声基底2倍）
            noise = 1.2  # %, 从之前实验得知
            marker = "✅" if diff < -noise else ("❌" if diff > noise else "—")
            print(f"  {r['label']:15s}: {r['mean_ms']:.4f}ms ({diff:+.2f}%) {marker}")

    return results


def main():
    print("=" * 70)
    print("实验A: ATC编译参数调优")
    print(f"参数组合: {len(PARAM_COMBO)}种")
    print(f"每OM: {ROUNDS}轮, {WARMUP}预热, {ITERATIONS}次")
    print(f"噪声基底参考: ~1.2%")
    print("=" * 70)

    mob_results = run_model("mobilenet_v3_small")
    res_results = run_model("resnet50")

    # 保存结果
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "timestamp": ts,
        "config": {"warmup": WARMUP, "iterations": ITERATIONS, "rounds": ROUNDS},
        "param_combos": [(l, p, b) for l, p, b in PARAM_COMBO],
        "mobilenet_v3_small": mob_results,
        "resnet50": res_results,
    }
    out_path = RESULTS_DIR / f"atc_params_{ts}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n结果已保存: {out_path}")

    print("\n" + "=" * 70)
    print("最佳参数组合:")
    for model_key, results in [("MobileNet", mob_results), ("ResNet50", res_results)]:
        ref = [r for r in results if r["label"] == "baseline"]
        if not ref:
            continue
        ref_mean = ref[0]["mean_ms"]
        best = min(results, key=lambda r: r["mean_ms"])
        diff = (best["mean_ms"] - ref_mean) / ref_mean * 100
        print(f"  {model_key}: {best['label']} ({best['mean_ms']:.4f}ms, {diff:+.2f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
