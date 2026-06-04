"""收集 ONNX 图结构指标，用于量化融合与简化效果。"""
import json
from collections import Counter
from pathlib import Path

import onnx

from config import BASELINES, MODELS_DIR, RESULTS_DIR


def analyze_model(path: Path) -> dict:
    model = onnx.load(str(path))
    graph = model.graph
    op_counts = Counter(node.op_type for node in graph.node)
    total_nodes = sum(op_counts.values())
    transpose_like = sum(op_counts.get(name, 0) for name in ["Transpose", "Reshape", "Flatten", "Unsqueeze", "Squeeze"])
    norm_like = sum(op_counts.get(name, 0) for name in ["BatchNormalization", "InstanceNormalization"])
    activation_like = sum(op_counts.get(name, 0) for name in ["Relu", "HardSigmoid", "HardSwish", "Clip", "Sigmoid", "Mul", "Add"])
    conv_like = op_counts.get("Conv", 0) + op_counts.get("Gemm", 0) + op_counts.get("MatMul", 0)
    return {
        "path": str(path),
        "total_nodes": total_nodes,
        "initializer_count": len(graph.initializer),
        "conv_like_nodes": conv_like,
        "norm_like_nodes": norm_like,
        "activation_like_nodes": activation_like,
        "transpose_like_nodes": transpose_like,
        "op_counts": dict(op_counts),
        "file_size_mb": path.stat().st_size / (1024 * 1024),
    }


def main() -> None:
    outputs = {}
    for model_key, cfg in BASELINES.items():
        candidates = {
            "original": cfg["onnx"],
            "shape_inferred": MODELS_DIR / f"{model_key}_shape_inferred.onnx",
            "optimized": MODELS_DIR / f"{model_key}_optimized.onnx",
        }
        outputs[model_key] = {}
        for tag, path in candidates.items():
            if path.exists():
                outputs[model_key][tag] = analyze_model(path)

    out = RESULTS_DIR / "onnx_graph_metrics.json"
    out.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
