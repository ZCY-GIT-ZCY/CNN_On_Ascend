"""
实验B: fusion_switch 强制开启未生效的 fusion pass。

已知（从 fusion_result.json）:
  - Conv2DWinogradFusionPass:  match=49 effect=0
  - RemoveCastFusionPass:      match=111 effect=0
  - ConvWeightCompressFusionPass: match=54 effect=0
  - TransdataCastFusionPass:   match=57 effect=0
  - ConvFormatRefreshFusionPass: match=53 effect=0

通过 --fusion_switch_file 强制这些 pass 开启，看能否激活加速。
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
FUSION_DIR = OPT_ANALYZE / "data" / "fusion_configs"
FUSION_DIR.mkdir(parents=True, exist_ok=True)
CNN_ROOT = OPT_ANALYZE.parent
MOBILENET_DIR = CNN_ROOT / "MobileNet"
RESNET_DIR = CNN_ROOT / "ResNet"
MODELS_DIR = CNN_ROOT / "Optimization" / "models"

ATC_BIN = "/usr/local/Ascend/ascend-toolkit/latest/bin/atc"
AIPP_CFGS = {
    "mobilenet_v3_small": str(MOBILENET_DIR / "aipp.cfg"),
    "resnet50": str(RESNET_DIR / "aipp.cfg"),
}
ONNX_FILES = {
    "mobilenet_v3_small": str(MOBILENET_DIR / "mobilenetv3.onnx"),
    "resnet50": str(RESNET_DIR / "resnet50.onnx"),
}
INPUT_SHAPE = "input_image:1,3,224,224"

WARMUP = 30
ITERATIONS = 180
ROUNDS = 4

# 各种 fusion 配置
FUSION_CONFIGS = {
    # 全部开启（使用默认行为 - but with explicit "on" for all matched passes）
    "all_default": {
        "GraphFusion": {
            "ALL": "on",
        },
        "UBFusion": {
            "ALL": "on",
        }
    },
    # 强制Winograd
    "winograd_on": {
        "GraphFusion": {
            "Conv2DWinogradFusionPass": "on",
            "ConvFormatRefreshFusionPass": "on",
            "ConvRearrangeGfzFusionPass": "on",
        }
    },
    # 移除Cast
    "remove_cast": {
        "GraphFusion": {
            "RemoveCastFusionPass": "on",
            "TransdataCastFusionPass": "on",
        }
    },
    # 权重压缩
    "weight_compress": {
        "GraphFusion": {
            "ConvWeightCompressFusionPass": "on",
        }
    },
    # 全部强制开启
    "all_on": {
        "GraphFusion": {
            "Conv2DWinogradFusionPass": "on",
            "ConvFormatRefreshFusionPass": "on",
            "ConvRearrangeGfzFusionPass": "on",
            "RemoveCastFusionPass": "on",
            "TransdataCastFusionPass": "on",
            "ConvWeightCompressFusionPass": "on",
            "SameInputConv2dFixpipePass": "on",
            "StrideHoistingPass": "on",
            "TBEConvAddFusion": "on",
        },
        "UBFusion": {
            "ALL": "on",
        }
    },
}


def write_fusion_config(name: str, config: dict) -> str:
    """写入 fusion_switch 配置文件，返回路径。"""
    path = FUSION_DIR / f"fusion_{name}.cfg"
    # 包装成 Switch 格式
    switch_config = {"Switch": config}
    path.write_text(json.dumps(switch_config, indent=2))
    return str(path)


def compile_with_fusion(model_key: str, config_name: str, config: dict,
                         work_dir: Path) -> Tuple[bool, str, str, Path]:
    """用指定的 fusion_switch 配置编译 ONNX，返回(成功, om路径, fusion结果, 工作目录)。"""
    cfg_path = write_fusion_config(config_name, config)
    output_prefix = work_dir / f"{model_key}_{config_name}"
    onnx = ONNX_FILES[model_key]
    aipp = AIPP_CFGS[model_key]

    cmd = [
        ATC_BIN, f"--model={onnx}", "--framework=5",
        f"--output={output_prefix}", "--soc_version=Ascend310B1",
        "--input_format=NCHW", f"--input_shape={INPUT_SHAPE}",
        "--precision_mode=allow_mix_precision",
        "--buffer_optimize=l2_optimize",
        "--op_select_implmode=high_performance",
        "--enable_small_channel=1",
        f"--insert_op_conf={aipp}",
        f"--fusion_switch_file={cfg_path}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    om_path = f"{output_prefix}.om"

    # 收集 fusion_result.json
    fusion_result = {}
    fusion_file = work_dir / "fusion_result.json"
    if fusion_file.exists():
        try:
            fusion_result = json.loads(fusion_file.read_text())
        except:
            fusion_result = {"error": "parse failed"}

    success = proc.returncode == 0 and Path(om_path).exists()
    return success, om_path, fusion_result, work_dir


def extract_fusion_effects(fusion_result: dict) -> Dict:
    """从 fusion_result 中提取各个 pass 的匹配和生效次数。"""
    effects = {}
    try:
        passes = {}
        data = fusion_result
        if isinstance(data, dict):
            for key, val in data.items():
                if isinstance(val, dict):
                    for ftype in ["graph_fusion", "ub_fusion"]:
                        if ftype in val:
                            passes.update(val[ftype])
                        # 有些可能是大小写不同
                        for k2, v2 in val.items():
                            if "fusion" in k2.lower():
                                passes.update(v2)

        for name, info in passes.items():
            if isinstance(info, dict):
                effects[name] = {
                    "match": int(info.get("match_times", 0)),
                    "effect": int(info.get("effect_times", 0)),
                }
    except Exception as e:
        effects["_error"] = str(e)

    return effects


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


def main():
    print("=" * 70)
    print("实验B: fusion_switch 强制开启未生效 pass")
    print(f"配置数: {len(FUSION_CONFIGS)}")
    print(f"参考基线: reexport_plain（已在之前实验中测量）")
    print("=" * 70)

    # 创建工作目录
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = RESULTS_DIR / f"fusion_exp_{ts}"
    work_dir.mkdir(parents=True, exist_ok=True)

    for model_key in ["mobilenet_v3_small", "resnet50"]:
        print(f"\n{'='*70}")
        print(f"模型: {model_key}")
        print(f"{'='*70}")

        # 先在 baseline（默认配置）下编译一次，获取 fusion_result
        print(f"\n  编译 baseline（默认配置）...")
        ok, baseline_om, baseline_fusion, _ = compile_with_fusion(
            model_key, "baseline", {"GraphFusion": {"ALL": "on"}, "UBFusion": {"ALL": "on"}}, work_dir
        )
        if not ok:
            print(f"  ❌ baseline 编译失败")
            continue

        baseline_effects = extract_fusion_effects(baseline_fusion)
        print(f"  Baseline fusion stats:")
        for pass_name, stats in sorted(baseline_effects.items())[:15]:
            if stats["match"] > 0:
                print(f"    {pass_name:45s}: match={stats['match']:3d}, effect={stats['effect']:3d} "
                      f"({stats['effect']/stats['match']*100:.0f}% if stats['match']>0 else 0)")

        # Benchmark baseline
        print(f"\n  Benchmark baseline...")
        baseline_means = []
        for r in range(ROUNDS):
            result = run_benchmark(baseline_om)
            if "mean_ms" in result:
                baseline_means.append(result["mean_ms"])
            time.sleep(1.5)
        baseline_mean = np.mean(baseline_means) if baseline_means else 0
        print(f"  Baseline: {baseline_mean:.4f}ms")

        # 对每个 fusion 配置做编译+benchmark
        for config_name in ["winograd_on", "remove_cast", "weight_compress", "all_on"]:
            config = FUSION_CONFIGS[config_name]
            print(f"\n  --- {config_name} ---")

            ok, om_path, fusion_result, _ = compile_with_fusion(
                model_key, config_name, config, work_dir
            )
            if not ok:
                print(f"  ❌ 编译失败")
                continue

            # 检查 fusion 效果
            effects = extract_fusion_effects(fusion_result)
            print(f"  Fusion changes:")
            for pass_name, stats in sorted(effects.items()):
                base_stats = baseline_effects.get(pass_name, {})
                if stats["match"] > 0:
                    old_effect = base_stats.get("effect", 0)
                    new_effect = stats["effect"]
                    if new_effect != old_effect:
                        print(f"    ✅ {pass_name:45s}: {old_effect}→{new_effect} (生效!)")
                    elif new_effect > 0:
                        print(f"    ✓  {pass_name:45s}: effect={new_effect}")

            # Benchmark
            print(f"  Benchmark...")
            means = []
            for r in range(ROUNDS):
                result = run_benchmark(om_path)
                if "mean_ms" in result:
                    means.append(result["mean_ms"])
                time.sleep(1.5)

            if means:
                avg = np.mean(means)
                diff = (avg - baseline_mean) / baseline_mean * 100
                marker = "✅" if diff < -1 else ("❌" if diff > 1 else "—")
                print(f"  Result: {avg:.4f}ms ({diff:+.2f}% vs baseline) {marker}")
            else:
                print(f"  ❌ benchmark 失败")

    print(f"\nDone. Work dir: {work_dir}")


if __name__ == "__main__":
    main()
