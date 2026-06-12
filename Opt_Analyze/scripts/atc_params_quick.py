"""
ATC参数调优 - 精简版（只编译最有价值的组合）
"""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np

SCRIPT = Path(__file__).resolve().parent
ROOT = SCRIPT.parent.parent
MODELS_DIR = ROOT / "Optimization" / "models"
RESULTS_DIR = SCRIPT.parent / "data"
AIPP = str(ROOT / "MobileNet" / "aipp.cfg")
ONNX = str(ROOT / "MobileNet" / "mobilenetv3.onnx")
ATC = "/usr/local/Ascend/ascend-toolkit/latest/bin/atc"

VARIANTS = [
    ("reexport_plain",  str(MODELS_DIR / "mobilenet_v3_small_reexport_plain.om"), "已有"),
    ("fp16_variant",    str(MODELS_DIR / "mobilenet_v3_small_fp16.om"), "已有"),
    ("force_fp16",      str(MODELS_DIR / "mobilenet_v3_small_atc_fp16.om"), "编译"),
    ("l1_optimize",     str(MODELS_DIR / "mobilenet_v3_small_atc_l1.om"), "编译"),
]

def compile_om(om_path, pm, bo):
    print(f"  编译 {Path(om_path).name} ({pm}+{bo})...", end=" ", flush=True)
    out_pref = str(om_path).replace(".om","")
    r = subprocess.run([ATC, f"--model={ONNX}", "--framework=5", f"--output={out_pref}",
        "--soc_version=Ascend310B1", "--input_format=NCHW", "--input_shape=input_image:1,3,224,224",
        f"--precision_mode={pm}", f"--buffer_optimize={bo}",
        "--op_select_implmode=high_performance", "--enable_small_channel=1",
        f"--insert_op_conf={AIPP}"], capture_output=True, text=True, timeout=600)
    ok = r.returncode == 0 and Path(om_path).exists()
    print(f"{'✅' if ok else '❌'}")
    return ok

def bench(om_path):
    cmd = [sys.executable, str(SCRIPT / "controlled_benchmark.py"),
           "--mode=child", f"--model-path={om_path}", "--warmup=30", "--iterations=180"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    for l in reversed(r.stdout.splitlines()):
        if l.strip().startswith("{"):
            return json.loads(l)
    return {"error": r.stderr[:200]}

print("="*60)
print("ATC参数调优(精简) - MobileNet")
for name, path, st in VARIANTS:
    print(f"\n[{name}] ({st})")
    om_path = Path(path)

    # 编译（如果需要）
    if not om_path.exists():
        pm = "force_fp16" if "fp16" in name else "allow_mix_precision"
        bo = "l1_optimize" if "l1" in name else "l2_optimize"
        if name == "force_fp16": bo = "l2_optimize"
        ok = compile_om(str(om_path), pm, bo)
        if not ok: continue
    else:
        print(f"  复用: {om_path.stat().st_size/1e6:.2f}MB")

    # Benchmark 5轮
    means = []
    for r in range(5):
        res = bench(str(om_path))
        if "mean_ms" in res:
            means.append(res["mean_ms"])
        time.sleep(2)

    if means:
        avg, std = np.mean(means), np.std(means, ddof=1)
        print(f"  → {avg:.4f}ms ±{std:.4f} (CV={std/avg*100:.1f}%)")

# 汇总
print("\n" + "="*60)
print("汇总（与reexport_plain对比）:")
print("="*60)
