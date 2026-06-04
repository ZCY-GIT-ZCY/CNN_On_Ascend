"""统一的 Ascend CNN 优化实验管线。

目标：
1. 生成多个 ONNX 变体
2. 收集图结构指标
3. 在可行时编译 OM
4. 用独立进程做基准测试，避免 ACL 重复初始化污染
5. 为后续报告生成结构化 JSON 数据
"""
import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import onnx

from config import BASELINES, MODELS_DIR, REPORTS_DIR, RESULTS_DIR, WORKSPACE


ATC_BIN = Path(os.environ.get("ASCEND_TOOLKIT_HOME", "/usr/local/Ascend/ascend-toolkit/latest")) / "atc" / "bin" / "atc"
AIC_METRICS_310B1 = [
    "PipeUtilization",
    "ArithmeticUtilization",
    "Memory",
    "MemoryL0",
    "MemoryUB",
    "ResourceConflictRatio",
]


@dataclass
class ExperimentResult:
    model_key: str
    experiment_id: str
    stage: str
    status: str
    source_path: str = ""
    artifact_path: str = ""
    compile_artifact: str = ""
    latency_ms: float = 0.0
    fps: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    model_size_mb: float = 0.0
    profiler_dir: str = ""
    profiler_metrics: Dict = None
    note: str = ""

    def to_dict(self) -> Dict:
        data = asdict(self)
        if data["profiler_metrics"] is None:
            data["profiler_metrics"] = {}
        return data


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def run_python(script: Path, args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True)


def parse_json_from_output(output: str) -> Dict:
    """从混合日志输出中提取最后一个 JSON 对象。"""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    for line in reversed(lines):
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"No JSON object found in output: {output[-1000:]}")


def benchmark_om_isolated(model_key: str, model_path: Path, iterations: int, warmup: int) -> Dict:
    benchmark_script = Path(__file__).resolve().parent / "benchmark_runner.py"
    proc = run_python(
        benchmark_script,
        [
            "--model-key",
            model_key,
            "--model-path",
            str(model_path),
            "--iterations",
            str(iterations),
            "--warmup",
            str(warmup),
        ],
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip())
    return parse_json_from_output(proc.stdout)


def profile_om(model_key: str, model_path: Path, warmup: int, iterations: int, metric: str) -> Dict:
    profile_script = Path(__file__).resolve().parent / "profile_runner.py"
    out_dir = RESULTS_DIR / "profiles" / f"{model_key}_{model_path.stem}_{metric}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "msprof",
        f"--output={out_dir}",
        "--ascendcl=on",
        "--runtime-api=on",
        "--task-time=on",
        "--aicpu=on",
        "--ai-core=on",
        f"--aic-metrics={metric}",
        sys.executable,
        str(profile_script),
        "--model-key",
        model_key,
        "--model-path",
        str(model_path),
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "status": "ok" if proc.returncode == 0 else "failed",
        "command": " ".join(cmd),
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "output_dir": str(out_dir),
    }


def compile_onnx_to_om(model_key: str, onnx_path: Path, output_prefix: Path) -> Dict:
    cfg = BASELINES[model_key]
    aipp_cfg = cfg["model_dir"] / "aipp.cfg"
    cmd = [
        str(ATC_BIN),
        f"--model={onnx_path}",
        "--framework=5",
        f"--output={output_prefix}",
        "--soc_version=Ascend310B1",
        "--input_format=NCHW",
        f"--input_shape={cfg['input_name']}:1,3,224,224",
        "--precision_mode=allow_mix_precision",
        "--op_select_implmode=high_performance",
        "--buffer_optimize=l2_optimize",
        "--enable_small_channel=1",
        f"--insert_op_conf={aipp_cfg}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    om_path = output_prefix.with_suffix(".om")
    return {
        "status": "ok" if proc.returncode == 0 and om_path.exists() else "failed",
        "command": " ".join(map(str, cmd)),
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "om_path": str(om_path),
    }


def infer_shapes_only(model_key: str) -> ExperimentResult:
    cfg = BASELINES[model_key]
    src = cfg["onnx"]
    dst = MODELS_DIR / f"{model_key}_shape_inferred.onnx"
    model = onnx.load(str(src))
    model = onnx.shape_inference.infer_shapes(model)
    onnx.save(model, str(dst))
    return ExperimentResult(
        model_key=model_key,
        experiment_id="onnx_shape_inferred",
        stage="graph",
        status="ok",
        source_path=str(src),
        artifact_path=str(dst),
        model_size_mb=file_size_mb(dst),
        note="补全 shape 信息，供后续简化与 ATC 编译使用。",
    )


def copy_baseline(model_key: str, iterations: int, warmup: int) -> ExperimentResult:
    cfg = BASELINES[model_key]
    src = cfg["om"]
    dst = MODELS_DIR / f"{model_key}_baseline.om"
    if src != dst:
        dst.write_bytes(src.read_bytes())
    stats = benchmark_om_isolated(model_key, dst, iterations=iterations, warmup=warmup)
    return ExperimentResult(
        model_key=model_key,
        experiment_id="baseline_copy",
        stage="runtime",
        status="ok",
        source_path=str(src),
        artifact_path=str(src),
        compile_artifact=str(dst),
        latency_ms=stats["mean_ms"],
        fps=stats["fps"],
        std_ms=stats["std_ms"],
        min_ms=stats["min_ms"],
        max_ms=stats["max_ms"],
        p50_ms=stats["p50_ms"],
        p90_ms=stats["p90_ms"],
        p95_ms=stats["p95_ms"],
        p99_ms=stats["p99_ms"],
        model_size_mb=file_size_mb(dst),
        note="独立进程统一重测已部署 OM，作为后续性能对照基线。",
    )


def collect_variant_candidates(model_key: str) -> List[Dict[str, Path]]:
    candidates = []
    for suffix in [
        "opset13.onnx",
        "shape_inferred.onnx",
        "bn_folded.onnx",
        "channels_last.onnx",
        "fp16.onnx",
        "onnxsim.onnx",
        "onnxoptimizer.onnx",
    ]:
        path = MODELS_DIR / f"{model_key}_{suffix}"
        if path.exists():
            candidates.append({"experiment_id": suffix.replace(".onnx", ""), "path": path})
    return candidates


def summarize_results(results: List[Dict]) -> Dict:
    summary: Dict[str, Dict] = {}
    for item in results:
        summary.setdefault(item["model_key"], {})[item["experiment_id"]] = {
            "stage": item["stage"],
            "status": item["status"],
            "latency_ms": item["latency_ms"],
            "fps": item["fps"],
            "std_ms": item["std_ms"],
            "p50_ms": item["p50_ms"],
            "p90_ms": item["p90_ms"],
            "p95_ms": item["p95_ms"],
            "p99_ms": item["p99_ms"],
            "model_size_mb": item["model_size_mb"],
            "artifact_path": item["artifact_path"],
            "compile_artifact": item["compile_artifact"],
            "profiler_dir": item["profiler_dir"],
            "note": item["note"],
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=list(BASELINES.keys()))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    if not args.skip_export:
        export_script = Path(__file__).resolve().parent / "export_variants.py"
        export_proc = run_python(export_script, [])
        (RESULTS_DIR / "export_variants.log").write_text(
            (export_proc.stdout or "") + "\n--- STDERR ---\n" + (export_proc.stderr or ""),
            encoding="utf-8",
        )

    all_results: List[Dict] = []
    for model_key in args.models:
        baseline_fn_name = "baseline_copy"
        for fn_name, fn in [(baseline_fn_name, copy_baseline), ("onnx_shape_inferred", infer_shapes_only)]:
            try:
                if fn_name == baseline_fn_name:
                    result = fn(model_key, iterations=args.iterations, warmup=args.warmup)
                else:
                    result = fn(model_key)
                all_results.append(result.to_dict())
            except Exception as exc:
                all_results.append(
                    ExperimentResult(
                        model_key=model_key,
                        experiment_id=fn_name,
                        stage="runtime" if fn_name == baseline_fn_name else "graph",
                        status="failed",
                        note=str(exc),
                    ).to_dict()
                )

        for variant in collect_variant_candidates(model_key):
            onnx_path = variant["path"]
            experiment_id = variant["experiment_id"]
            om_prefix = MODELS_DIR / f"{model_key}_{experiment_id}"
            if args.skip_compile:
                all_results.append(
                    ExperimentResult(
                        model_key=model_key,
                        experiment_id=experiment_id,
                        stage="compile",
                        status="skipped",
                        artifact_path=str(onnx_path),
                        model_size_mb=file_size_mb(onnx_path),
                        note="按参数跳过编译。",
                    ).to_dict()
                )
                continue

            try:
                compile_info = compile_onnx_to_om(model_key, onnx_path, om_prefix)
                if compile_info["status"] != "ok":
                    raise RuntimeError(compile_info["stderr"] or compile_info["stdout"] or "ATC compile failed")
                om_path = Path(compile_info["om_path"])
                stats = benchmark_om_isolated(model_key, om_path, iterations=args.iterations, warmup=args.warmup)
                profile_info: Optional[Dict] = None
                if args.profile:
                    for metric in AIC_METRICS_310B1:
                        profile_info = profile_om(model_key, om_path, warmup=5, iterations=20, metric=metric)
                        if profile_info["status"] == "ok":
                            break
                all_results.append(
                    ExperimentResult(
                        model_key=model_key,
                        experiment_id=experiment_id,
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
                        profiler_dir=(profile_info or {}).get("output_dir", ""),
                        profiler_metrics=profile_info or {},
                        note="ONNX 变体已完成编译与独立进程基准测试。",
                    ).to_dict()
                )
            except Exception as exc:
                all_results.append(
                    ExperimentResult(
                        model_key=model_key,
                        experiment_id=experiment_id,
                        stage="runtime",
                        status="failed",
                        source_path=str(onnx_path),
                        artifact_path=str(onnx_path),
                        model_size_mb=file_size_mb(onnx_path),
                        note=str(exc),
                    ).to_dict()
                )

    results_path = RESULTS_DIR / "experiment_results.json"
    summary_path = RESULTS_DIR / "experiment_summary.json"
    results_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summarize_results(all_results), ensure_ascii=False, indent=2), encoding="utf-8")
    print(results_path)
    print(summary_path)


if __name__ == "__main__":
    main()
