"""严格串行、交错顺序的 OM 基准重跑器。

目标：
1. 复用已经编译好的 OM，不重新编译。
2. 每个测量轮次使用 strict_benchmark.py 独立子进程运行。
3. 按预设顺序串行执行，避免并行争抢 NPU 资源。
4. 通过交错顺序（ABBA / BAAB 思路）降低时间漂移对某一模型的系统性偏置。
5. 汇总每次运行结果，输出便于后续报告分析的 JSON。
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent
STRICT_BENCH = ROOT / "strict_benchmark.py"
RESULTS_DIR = ROOT / "results"


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


def summarize_pair(a: List[Dict], b: List[Dict]) -> Dict:
    a_means = [item["overall_mean_ms"] for item in a]
    b_means = [item["overall_mean_ms"] for item in b]
    return {
        "a_avg_ms": sum(a_means) / len(a_means),
        "b_avg_ms": sum(b_means) / len(b_means),
        "delta_ms": (sum(b_means) / len(b_means)) - (sum(a_means) / len(a_means)),
        "delta_pct_vs_a": ((sum(b_means) / len(b_means)) - (sum(a_means) / len(a_means))) / (sum(a_means) / len(a_means)) * 100.0,
        "a_runs": a_means,
        "b_runs": b_means,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=220)
    parser.add_argument("--warmup-first", type=int, default=30)
    parser.add_argument("--warmup-full", type=int, default=20)
    parser.add_argument("--cooldown-between-rounds", type=float, default=2.0)
    parser.add_argument("--cooldown-between-models", type=float, default=5.0)
    args = parser.parse_args()

    schedule = [
        {
            "label": "mobilenet_baseline_run1",
            "model_key": "mobilenet_v3_small",
            "model_path": ROOT / "models" / "mobilenet_v3_small_baseline.om",
            "bucket": "mobilenet_baseline",
        },
        {
            "label": "mobilenet_fp16_run1",
            "model_key": "mobilenet_v3_small",
            "model_path": ROOT / "models" / "mobilenet_v3_small_fp16.om",
            "bucket": "mobilenet_fp16",
        },
        {
            "label": "mobilenet_fp16_run2",
            "model_key": "mobilenet_v3_small",
            "model_path": ROOT / "models" / "mobilenet_v3_small_fp16.om",
            "bucket": "mobilenet_fp16",
        },
        {
            "label": "mobilenet_baseline_run2",
            "model_key": "mobilenet_v3_small",
            "model_path": ROOT / "models" / "mobilenet_v3_small_baseline.om",
            "bucket": "mobilenet_baseline",
        },
        {
            "label": "resnet_baseline_run1",
            "model_key": "resnet50",
            "model_path": ROOT / "models" / "resnet50_baseline.om",
            "bucket": "resnet_baseline",
        },
        {
            "label": "resnet_fp16_run1",
            "model_key": "resnet50",
            "model_path": ROOT / "models" / "resnet50_fp16.om",
            "bucket": "resnet_fp16",
        },
        {
            "label": "resnet_fp16_run2",
            "model_key": "resnet50",
            "model_path": ROOT / "models" / "resnet50_fp16.om",
            "bucket": "resnet_fp16",
        },
        {
            "label": "resnet_baseline_run2",
            "model_key": "resnet50",
            "model_path": ROOT / "models" / "resnet50_baseline.om",
            "bucket": "resnet_baseline",
        },
    ]

    grouped: Dict[str, List[Dict]] = {
        "mobilenet_baseline": [],
        "mobilenet_fp16": [],
        "resnet_baseline": [],
        "resnet_fp16": [],
    }
    ordered_results: List[Dict] = []

    for idx, item in enumerate(schedule):
        result = run_one(
            label=item["label"],
            model_key=item["model_key"],
            model_path=item["model_path"],
            rounds=args.rounds,
            iterations=args.iterations,
            warmup_first=args.warmup_first,
            warmup_full=args.warmup_full,
            cooldown_between_rounds=args.cooldown_between_rounds,
        )
        grouped[item["bucket"]].append(result)
        ordered_results.append(result)
        if idx < len(schedule) - 1:
            time.sleep(args.cooldown_between_models)

    summary = {
        "mobilenet": summarize_pair(grouped["mobilenet_baseline"], grouped["mobilenet_fp16"]),
        "resnet50": summarize_pair(grouped["resnet_baseline"], grouped["resnet_fp16"]),
    }

    payload = {
        "config": {
            "rounds": args.rounds,
            "iterations": args.iterations,
            "warmup_first": args.warmup_first,
            "warmup_full": args.warmup_full,
            "cooldown_between_rounds": args.cooldown_between_rounds,
            "cooldown_between_models": args.cooldown_between_models,
            "schedule": [item["label"] for item in schedule],
        },
        "ordered_results": ordered_results,
        "grouped_results": grouped,
        "summary": summary,
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
