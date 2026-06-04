"""数值一致性验证：比较基线 ONNX 与各优化 ONNX 变体的输出偏差。"""
import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnxruntime as ort
import torch
import torchvision.models as models

from config import BASELINES, MODELS_DIR, RESULTS_DIR


INPUT_SHAPE = (1, 3, 224, 224)
SEED = 20260604
SAMPLE_COUNT = 8


@dataclass
class ValidationResult:
    model_key: str
    variant: str
    status: str
    sample_count: int
    max_abs_diff: float = 0.0
    mean_abs_diff: float = 0.0
    rmse: float = 0.0
    cosine_similarity_mean: float = 0.0
    top1_match_rate: float = 0.0
    baseline_top1_ids: List[int] = None
    variant_top1_ids: List[int] = None
    note: str = ""

    def to_dict(self) -> Dict:
        data = asdict(self)
        if data["baseline_top1_ids"] is None:
            data["baseline_top1_ids"] = []
        if data["variant_top1_ids"] is None:
            data["variant_top1_ids"] = []
        return data


def build_reference_model(model_key: str):
    if model_key == "resnet50":
        return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).eval()
    if model_key == "mobilenet_v3_small":
        return models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1).eval()
    raise KeyError(f"Unsupported model key: {model_key}")


def generate_samples(sample_count: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    samples = rng.standard_normal((sample_count, *INPUT_SHAPE[1:]), dtype=np.float32)
    return samples.astype(np.float32)


def run_pytorch_baseline(model_key: str, samples: np.ndarray) -> np.ndarray:
    model = build_reference_model(model_key)
    with torch.inference_mode():
        outputs = model(torch.from_numpy(samples)).cpu().numpy().astype(np.float32)
    return outputs


def create_session(model_path: Path) -> ort.InferenceSession:
    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(model_path), providers=providers)


def run_onnx_model(model_path: Path, samples: np.ndarray) -> np.ndarray:
    session = create_session(model_path)
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_type = input_meta.type
    if "float16" in input_type:
        cast_samples = samples.astype(np.float16)
    else:
        cast_samples = samples.astype(np.float32)
    outputs = []
    for sample in cast_samples:
        feed = {input_name: sample[None, ...]}
        pred = session.run(None, feed)[0]
        outputs.append(pred.astype(np.float32))
    return np.concatenate(outputs, axis=0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.sum(a * b, axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denom = np.clip(denom, 1e-12, None)
    return dot / denom


def evaluate_variant(model_key: str, variant: str, model_path: Path, samples: np.ndarray, baseline_outputs: np.ndarray) -> ValidationResult:
    try:
        variant_outputs = run_onnx_model(model_path, samples)
    except Exception as exc:
        return ValidationResult(
            model_key=model_key,
            variant=variant,
            status="failed",
            sample_count=len(samples),
            note=str(exc),
        )

    diff = variant_outputs - baseline_outputs
    abs_diff = np.abs(diff)
    cosine = cosine_similarity(baseline_outputs, variant_outputs)
    baseline_top1 = np.argmax(baseline_outputs, axis=1)
    variant_top1 = np.argmax(variant_outputs, axis=1)
    match_rate = float(np.mean(baseline_top1 == variant_top1))

    return ValidationResult(
        model_key=model_key,
        variant=variant,
        status="ok",
        sample_count=len(samples),
        max_abs_diff=float(abs_diff.max()),
        mean_abs_diff=float(abs_diff.mean()),
        rmse=float(np.sqrt(np.mean(diff ** 2))),
        cosine_similarity_mean=float(cosine.mean()),
        top1_match_rate=match_rate,
        baseline_top1_ids=[int(x) for x in baseline_top1.tolist()],
        variant_top1_ids=[int(x) for x in variant_top1.tolist()],
        note="baseline 为 torchvision PyTorch 权重前向输出；variant 为导出 ONNX CPU 推理输出。",
    )


def collect_variant_paths(model_key: str) -> Dict[str, Path]:
    variants: Dict[str, Path] = {}
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
            variants[suffix.replace(".onnx", "")] = path
    return variants


def summarize(results: List[Dict]) -> Dict[str, Dict[str, Dict]]:
    summary: Dict[str, Dict[str, Dict]] = {}
    for item in results:
        summary.setdefault(item["model_key"], {})[item["variant"]] = {
            "status": item["status"],
            "sample_count": item["sample_count"],
            "max_abs_diff": item["max_abs_diff"],
            "mean_abs_diff": item["mean_abs_diff"],
            "rmse": item["rmse"],
            "cosine_similarity_mean": item["cosine_similarity_mean"],
            "top1_match_rate": item["top1_match_rate"],
            "note": item["note"],
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=list(BASELINES.keys()))
    parser.add_argument("--sample-count", type=int, default=SAMPLE_COUNT)
    args = parser.parse_args()

    samples = generate_samples(args.sample_count)
    all_results: List[Dict] = []

    for model_key in args.models:
        baseline_outputs = run_pytorch_baseline(model_key, samples)
        variants = collect_variant_paths(model_key)
        for variant, path in variants.items():
            result = evaluate_variant(model_key, variant, path, samples, baseline_outputs)
            all_results.append(result.to_dict())

    results_path = RESULTS_DIR / "numerical_validation_results.json"
    summary_path = RESULTS_DIR / "numerical_validation_summary.json"
    results_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summarize(all_results), ensure_ascii=False, indent=2), encoding="utf-8")
    print(results_path)
    print(summary_path)


if __name__ == "__main__":
    main()
