import json
from pathlib import Path
from typing import Dict, List, Tuple

from config import BASELINES, MODELS_DIR, RESULTS_DIR
from run_experiments import compile_onnx_to_om, benchmark_om_isolated, ExperimentResult, file_size_mb


RUNTIME_VARIANTS = [
    "shape_inferred",
    "opset13",
    "bn_folded",
    "channels_last",
    "fp16",
    "onnxsim",
    "onnxoptimizer",
]


def build_reexport_plain(model_key: str) -> Path:
    src = BASELINES[model_key]["onnx"]
    dst = MODELS_DIR / f"{model_key}_reexport_plain.onnx"
    dst.write_bytes(src.read_bytes())
    return dst


def benchmark_runtime_variant(model_key: str, variant_id: str, iterations: int, warmup: int) -> Dict:
    onnx_path = MODELS_DIR / f"{model_key}_{variant_id}.onnx"
    output_prefix = MODELS_DIR / f"{model_key}_{variant_id}"
    compile_info = compile_onnx_to_om(model_key, onnx_path, output_prefix)
    if compile_info["status"] != "ok":
        raise RuntimeError(f"compile failed for {model_key} {variant_id}: {compile_info['stderr']}")
    om_path = Path(compile_info["om_path"])
    stats = benchmark_om_isolated(model_key, om_path, iterations=iterations, warmup=warmup)
    result = ExperimentResult(
        model_key=model_key,
        experiment_id=variant_id,
        stage="runtime",
        status="ok",
        source_path=str(onnx_path),
        artifact_path=str(onnx_path),
        compile_artifact=str(om_path),
        latency_ms=stats["mean_ms"],
        fps=stats["fps"],
        std_ms=stats["std_ms"],
        min_ms=stats["min_ms"],
        max_ms=stats["max_ms"],
        p50_ms=stats["p50_ms"],
        p90_ms=stats["p90_ms"],
        p95_ms=stats["p95_ms"],
        p99_ms=stats["p99_ms"],
        model_size_mb=file_size_mb(om_path),
        note="补做 reexport_plain 对照后，用统一流程重测该变体。",
    )
    return result.to_dict()


def main() -> None:
    iterations = 120
    warmup = 20
    results: List[Dict] = []
    summary: Dict[str, Dict] = {}

    for model_key in BASELINES:
        plain_path = build_reexport_plain(model_key)
        plain_compile = compile_onnx_to_om(model_key, plain_path, MODELS_DIR / f"{model_key}_reexport_plain")
        if plain_compile["status"] != "ok":
            raise RuntimeError(f"reexport_plain compile failed for {model_key}: {plain_compile['stderr']}")
        plain_om = Path(plain_compile["om_path"])
        plain_stats = benchmark_om_isolated(model_key, plain_om, iterations=iterations, warmup=warmup)
        plain_result = ExperimentResult(
            model_key=model_key,
            experiment_id="reexport_plain",
            stage="runtime",
            status="ok",
            source_path=str(plain_path),
            artifact_path=str(plain_path),
            compile_artifact=str(plain_om),
            latency_ms=plain_stats["mean_ms"],
            fps=plain_stats["fps"],
            std_ms=plain_stats["std_ms"],
            min_ms=plain_stats["min_ms"],
            max_ms=plain_stats["max_ms"],
            p50_ms=plain_stats["p50_ms"],
            p90_ms=plain_stats["p90_ms"],
            p95_ms=plain_stats["p95_ms"],
            p99_ms=plain_stats["p99_ms"],
            model_size_mb=file_size_mb(plain_om),
            note="新增中性对照：直接复用项目现有 ONNX，经相同 ATC 流程重新编译，不启用额外优化。",
        ).to_dict()
        results.append(plain_result)

        variant_results: List[Tuple[str, Dict]] = []
        for variant_id in RUNTIME_VARIANTS:
            variant_result = benchmark_runtime_variant(model_key, variant_id, iterations=iterations, warmup=warmup)
            results.append(variant_result)
            variant_results.append((variant_id, variant_result))

        plain_latency = plain_result["latency_ms"]
        baseline_gap = None
        try:
            base_results = json.loads((RESULTS_DIR / "experiment_results.json").read_text())
            baseline_latency = next(
                item["latency_ms"]
                for item in base_results
                if item["model_key"] == model_key and item["experiment_id"] == "baseline_copy"
            )
            baseline_gap = (plain_latency - baseline_latency) / baseline_latency * 100.0
        except Exception:
            baseline_gap = None

        best_variant = min(variant_results, key=lambda x: x[1]["latency_ms"])
        summary[model_key] = {
            "reexport_plain_latency_ms": plain_latency,
            "reexport_plain_vs_baseline_pct": baseline_gap,
            "best_reexport_variant": best_variant[0],
            "best_reexport_variant_latency_ms": best_variant[1]["latency_ms"],
            "best_variant_vs_plain_pct": (best_variant[1]["latency_ms"] - plain_latency) / plain_latency * 100.0,
        }

    (RESULTS_DIR / "reexport_control_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "reexport_control_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(RESULTS_DIR / "reexport_control_results.json")
    print(RESULTS_DIR / "reexport_control_summary.json")


if __name__ == "__main__":
    main()
