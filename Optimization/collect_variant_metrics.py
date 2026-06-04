"""为导出的 ONNX 变体收集图结构与体积指标。"""
import json
from collections import Counter
from pathlib import Path

import onnx

from config import MODELS_DIR, RESULTS_DIR


def analyze_model(path: Path) -> dict:
    model = onnx.load(str(path))
    graph = model.graph
    op_counts = Counter(node.op_type for node in graph.node)
    total_nodes = sum(op_counts.values())
    return {
        "path": str(path),
        "total_nodes": total_nodes,
        "initializer_count": len(graph.initializer),
        "conv": op_counts.get("Conv", 0),
        "gemm": op_counts.get("Gemm", 0),
        "batchnorm": op_counts.get("BatchNormalization", 0),
        "relu": op_counts.get("Relu", 0),
        "hardsigmoid": op_counts.get("HardSigmoid", 0),
        "mul": op_counts.get("Mul", 0),
        "add": op_counts.get("Add", 0),
        "transpose": op_counts.get("Transpose", 0),
        "flatten": op_counts.get("Flatten", 0),
        "global_avgpool": op_counts.get("GlobalAveragePool", 0),
        "maxpool": op_counts.get("MaxPool", 0),
        "file_size_mb": path.stat().st_size / (1024 * 1024),
        "op_counts": dict(op_counts),
    }


def main() -> None:
    result = {}
    for path in sorted(MODELS_DIR.glob("*.onnx")):
        result[path.name] = analyze_model(path)

    out = RESULTS_DIR / "variant_graph_metrics.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
