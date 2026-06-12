"""生成更严格的成对交错基准测试计划与结果摘要。

目标：
1. 以 pair 为中心，而不是把很多变体串成一条长 schedule。
2. 统一支持 ABAB / BAAB 两种顺序设计。
3. 输出每个 pair 的配对差值分布、跨轮波动和显著性提示。
4. 重点覆盖 baseline_copy vs reexport_plain，以及 reexport_plain vs 各优化变体。
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
STRICT_BENCH = ROOT / "strict_benchmark.py"

PAIR_SPECS = {
    "mobilenet_v3_small": {
        "baseline_copy": ROOT / "models" / "mobilenet_v3_small_baseline.om",
        "reexport_plain": ROOT / "models" / "mobilenet_v3_small_reexport_plain.om",
        "shape_inferred": ROOT / "models" / "mobilenet_v3_small_shape_inferred.om",
        "opset13": ROOT / "models" / "mobilenet_v3_small_opset13.om",
        "bn_folded": ROOT / "models" / "mobilenet_v3_small_bn_folded.om",
        "channels_last": ROOT / "models" / "mobilenet_v3_small_channels_last.om",
        "fp16": ROOT / "models" / "mobilenet_v3_small_fp16.om",
        "onnxsim": ROOT / "models" / "mobilenet_v3_small_onnxsim.om",
        "onnxoptimizer": ROOT / "models" / "mobilenet_v3_small_onnxoptimizer.om",
    },
    "resnet50": {
        "baseline_copy": ROOT / "models" / "resnet50_baseline.om",
        "reexport_plain": ROOT / "models" / "resnet50_reexport_plain.om",
        "shape_inferred": ROOT / "models" / "resnet50_shape_inferred.om",
        "opset13": ROOT / "models" / "resnet50_opset13.om",
        "bn_folded": ROOT / "models" / "resnet50_bn_folded.om",
        "channels_last": ROOT / "models" / "resnet50_channels_last.om",
        "fp16": ROOT / "models" / "resnet50_fp16.om",
        "onnxsim": ROOT / "models" / "resnet50_onnxsim.om",
        "onnxoptimizer": ROOT / "models" / "resnet50_onnxoptimizer.om",
    },
}


def run_one(label: str, model_key: str, model_path: Path, rounds: int, iterations: int,
            warmup_first: int, warmup_full: int, cooldown_between_rounds: float) -> Dict:
    cmd = [
        sys.executable,
        str(STRICT_BENCH),
        "--model-key",
        model_key,
        "--model-path",
        str(model_path),
        "--rounds",
        str(rounds),
        "--iterations",
        str(iterations),
        "--warmup-first",
        str(warmup_first),
        "--warmup-full",
        str(warmup_full),
        "--cooldown-between-rounds",
        str(cooldown_between_rounds),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed:\n{proc.stderr or proc.stdout}")

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    payload = None
    for line in reversed(lines):
        if line.startswith("{"):
            payload = json.loads(line)
            break
    if payload is None:
        raise RuntimeError(f"{label}: no JSON output found")

    payload["label"] = label
    payload["compile_artifact"] = str(model_path)
    return payload


def build_schedule(design: str, left_label: str, right_label: str) -> List[Tuple[str, str]]:
    if design == "ABAB":
        return [("left", left_label), ("right", right_label), ("left", left_label), ("right", right_label)]
    if design == "BAAB":
        return [("right", right_label), ("left", left_label), ("left", left_label), ("right", right_label)]
    raise ValueError(f"Unsupported design: {design}")


def summarize_group(results: List[Dict]) -> Dict:
    means = [item["overall_mean_ms"] for item in results]
    return {
        "run_count": len(results),
        "means_ms": means,
        "avg_ms": float(np.mean(means)),
        "std_ms": float(np.std(means)),
        "min_ms": float(np.min(means)),
        "max_ms": float(np.max(means)),
    }


def summarize_pair(left_results: List[Dict], right_results: List[Dict], left_name: str, right_name: str) -> Dict:
    left_means = [item["overall_mean_ms"] for item in left_results]
    right_means = [item["overall_mean_ms"] for item in right_results]
    pair_count = min(len(left_means), len(right_means))
    paired_deltas = [right_means[i] - left_means[i] for i in range(pair_count)]
    left_avg = float(np.mean(left_means))
    right_avg = float(np.mean(right_means))
    delta_avg = right_avg - left_avg
    left_vol = float(np.std(left_means))
    right_vol = float(np.std(right_means))
    noise_band = max(left_vol, right_vol)
    significance = "inconclusive"
    if pair_count and all(delta > 0 for delta in paired_deltas) and abs(delta_avg) > noise_band:
        significance = f"{right_name}_slower"
    elif pair_count and all(delta < 0 for delta in paired_deltas) and abs(delta_avg) > noise_band:
        significance = f"{right_name}_faster"
    elif abs(delta_avg) <= noise_band:
        significance = "within_noise_band"

    return {
        "left": summarize_group(left_results),
        "right": summarize_group(right_results),
        "left_name": left_name,
        "right_name": right_name,
        "delta_ms": delta_avg,
        "delta_pct_vs_left": (delta_avg / left_avg * 100.0) if left_avg else 0.0,
        "paired_delta_ms": paired_deltas,
        "paired_delta_pct_vs_left": [delta / left_means[i] * 100.0 for i, delta in enumerate(paired_deltas)],
        "noise_band_ms": noise_band,
        "significance_hint": significance,
    }


def run_pair(model_key: str, left_name: str, right_name: str, design: str,
             rounds: int, iterations: int, warmup_first: int,
             warmup_full: int, cooldown_between_rounds: float,
             cooldown_between_runs: float) -> Dict:
    model_paths = PAIR_SPECS[model_key]
    left_path = model_paths[left_name]
    right_path = model_paths[right_name]
    if not left_path.exists():
        raise FileNotFoundError(f"Missing OM: {left_path}")
    if not right_path.exists():
        raise FileNotFoundError(f"Missing OM: {right_path}")

    schedule = build_schedule(design, left_name, right_name)
    ordered_results: List[Dict] = []
    grouped = {"left": [], "right": []}

    for idx, (side, variant_name) in enumerate(schedule):
        label = f"{model_key}_{left_name}_vs_{right_name}_{design.lower()}_{idx + 1}_{variant_name}"
        model_path = left_path if side == "left" else right_path
        result = run_one(
            label=label,
            model_key=model_key,
            model_path=model_path,
            rounds=rounds,
            iterations=iterations,
            warmup_first=warmup_first,
            warmup_full=warmup_full,
            cooldown_between_rounds=cooldown_between_rounds,
        )
        result["variant_name"] = variant_name
        result["design_side"] = side
        ordered_results.append(result)
        grouped[side].append(result)
        if idx < len(schedule) - 1 and cooldown_between_runs > 0:
            time.sleep(cooldown_between_runs)

    summary = summarize_pair(grouped["left"], grouped["right"], left_name, right_name)
    return {
        "model_key": model_key,
        "left_name": left_name,
        "right_name": right_name,
        "design": design,
        "schedule": [item[1] for item in schedule],
        "ordered_results": ordered_results,
        "grouped_results": grouped,
        "summary": summary,
    }


def default_pairs() -> List[Tuple[str, str, str]]:
    pairs = []
    for model_key in ["mobilenet_v3_small", "resnet50"]:
        pairs.append((model_key, "baseline_copy", "reexport_plain"))
        for variant in ["shape_inferred", "opset13", "channels_last", "onnxsim", "onnxoptimizer", "fp16", "bn_folded"]:
            pairs.append((model_key, "reexport_plain", variant))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--design", choices=["ABAB", "BAAB"], default="ABAB")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=220)
    parser.add_argument("--warmup-first", type=int, default=30)
    parser.add_argument("--warmup-full", type=int, default=20)
    parser.add_argument("--cooldown-between-rounds", type=float, default=2.0)
    parser.add_argument("--cooldown-between-runs", type=float, default=5.0)
    parser.add_argument("--pair", action="append", default=[], help="格式: model_key:left:right")
    args = parser.parse_args()

    pair_specs = []
    if args.pair:
        for item in args.pair:
            model_key, left_name, right_name = item.split(":", 2)
            pair_specs.append((model_key, left_name, right_name))
    else:
        pair_specs = default_pairs()

    pair_results = []
    for model_key, left_name, right_name in pair_specs:
        pair_results.append(
            run_pair(
                model_key=model_key,
                left_name=left_name,
                right_name=right_name,
                design=args.design,
                rounds=args.rounds,
                iterations=args.iterations,
                warmup_first=args.warmup_first,
                warmup_full=args.warmup_full,
                cooldown_between_rounds=args.cooldown_between_rounds,
                cooldown_between_runs=args.cooldown_between_runs,
            )
        )

    payload = {
        "config": {
            "design": args.design,
            "rounds": args.rounds,
            "iterations": args.iterations,
            "warmup_first": args.warmup_first,
            "warmup_full": args.warmup_full,
            "cooldown_between_rounds": args.cooldown_between_rounds,
            "cooldown_between_runs": args.cooldown_between_runs,
            "pair_specs": [
                {"model_key": model_key, "left_name": left_name, "right_name": right_name}
                for model_key, left_name, right_name in pair_specs
            ],
        },
        "pairs": pair_results,
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
