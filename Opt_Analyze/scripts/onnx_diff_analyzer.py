"""
ONNX 图差异分析工具 - 量化每个变体相对 baseline ONNX 的结构差异。

功能：
1. 对每个变体的 ONNX 进行详细结构统计
2. 与原始 ONNX 对比，标记差异
3. 提取语义级差异摘要（删除了什么、新增了什么、改变了什么）
4. 保存到 JSON 供后续分析
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import onnx
from onnx import helper, shape_inference

SCRIPT_DIR = Path(__file__).resolve().parent
OPT_ANALYZE = SCRIPT_DIR.parent
CNN_ROOT = OPT_ANALYZE.parent
OPTIMIZATION_DIR = CNN_ROOT / "Optimization"
MODELS_DIR = OPTIMIZATION_DIR / "models"
MOBILENET_DIR = CNN_ROOT / "MobileNet"
RESNET_DIR = CNN_ROOT / "ResNet"

RESULTS_DIR = OPT_ANALYZE / "data"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 原始 ONNX 路径
ORIGINAL_ONNX = {
    "mobilenet_v3_small": MOBILENET_DIR / "mobilenetv3.onnx",
    "resnet50": RESNET_DIR / "resnet50.onnx",
}

# 变体 ONNX 路径模板
VARIANT_ONNX_PATHS = [
    "shape_inferred", "opset13", "bn_folded",
    "channels_last", "fp16", "onnxsim", "onnxoptimizer",
]


def analyze_onnx(path: Path) -> Dict:
    """对单个 ONNX 模型进行详细结构分析。"""
    model = onnx.load(str(path))
    graph = model.graph

    # 基本信息
    opset_imports = {d.domain: d.version for d in model.opset_import}
    ir_version = model.ir_version

    # 节点统计
    total_nodes = len(graph.node)
    op_counts = {}
    for node in graph.node:
        op = node.op_type
        op_counts[op] = op_counts.get(op, 0) + 1

    # Initializer 统计
    initializer_count = len(graph.initializer)
    initializer_dtypes = {}
    for init in graph.initializer:
        dtype = init.data_type
        name = init.name
        dims = list(init.dims)
        initializer_dtypes[name] = {
            "dtype": dtype,
            "shape": dims,
            "size_elements": int(onnx.numpy_helper.to_array(init).size),
        }

    # Input/Output
    input_info = []
    for inp in graph.input:
        input_info.append({
            "name": inp.name,
            "shape": [str(d.dim_value) if d.dim_value > 0 else "?"
                      for d in inp.type.tensor_type.shape.dim],
            "dtype": inp.type.tensor_type.elem_type,
        })

    output_info = []
    for out in graph.output:
        output_info.append({
            "name": out.name,
            "shape": [str(d.dim_value) if d.dim_value > 0 else "?"
                      for d in out.type.tensor_type.shape.dim],
            "dtype": out.type.tensor_type.elem_type,
        })

    # 特殊节点统计（类型转换、转置、变形等）
    special_nodes = {
        "transpose": [],
        "cast": [],
        "reshape": [],
        "shape": [],
        "unsqueeze": [],
        "squeeze": [],
        "slice": [],
        "gather": [],
        "identity": [],
    }
    for node in graph.node:
        op = node.op_type
        if op in special_nodes:
            special_nodes[op].append({
                "name": node.name,
                "inputs": list(node.input),
                "outputs": list(node.output),
            })

    # 权重/常量折叠相关
    constant_nodes = sum(1 for n in graph.node if n.op_type == "Constant")

    file_size_mb = path.stat().st_size / (1024 * 1024)

    return {
        "path": str(path),
        "file_size_mb": round(file_size_mb, 4),
        "ir_version": ir_version,
        "opset_imports": opset_imports,
        "total_nodes": total_nodes,
        "op_counts": dict(sorted(op_counts.items(), key=lambda x: -x[1])),
        "initializer_count": initializer_count,
        "input_info": input_info,
        "output_info": output_info,
        "special_nodes": {k: v for k, v in special_nodes.items() if v},
        "constant_nodes_count": constant_nodes,
    }


def diff_two_onnx(reference: Dict, target: Dict, ref_label: str = "reference",
                  target_label: str = "target") -> Dict:
    """比较两个 ONNX 结构分析结果，提取差异。"""
    diffs = {}

    # 文件大小差异
    if reference["file_size_mb"] != target["file_size_mb"]:
        diffs["file_size_mb"] = {
            ref_label: reference["file_size_mb"],
            target_label: target["file_size_mb"],
            "diff_mb": round(target["file_size_mb"] - reference["file_size_mb"], 4),
            "diff_pct": round(
                (target["file_size_mb"] - reference["file_size_mb"])
                / reference["file_size_mb"] * 100, 2
            ),
        }

    # 节点数差异
    if reference["total_nodes"] != target["total_nodes"]:
        diffs["total_nodes"] = {
            ref_label: reference["total_nodes"],
            target_label: target["total_nodes"],
            "diff": target["total_nodes"] - reference["total_nodes"],
        }

    # 算子计数差异
    ref_ops = reference["op_counts"]
    tgt_ops = target["op_counts"]
    all_ops = set(list(ref_ops.keys()) + list(tgt_ops.keys()))
    op_diffs = {}
    for op in sorted(all_ops):
        ref_cnt = ref_ops.get(op, 0)
        tgt_cnt = tgt_ops.get(op, 0)
        if ref_cnt != tgt_cnt:
            op_diffs[op] = {
                ref_label: ref_cnt,
                target_label: tgt_cnt,
                "diff": tgt_cnt - ref_cnt,
            }
    if op_diffs:
        diffs["op_count_diffs"] = op_diffs

    # IR 版本差异
    if reference["ir_version"] != target["ir_version"]:
        diffs["ir_version"] = {
            ref_label: reference["ir_version"],
            target_label: target["ir_version"],
        }

    # Opset 差异
    ref_opsets = reference["opset_imports"]
    tgt_opsets = target["opset_imports"]
    all_domains = set(list(ref_opsets.keys()) + list(tgt_opsets.keys()))
    opset_diffs = {}
    for d in sorted(all_domains):
        if ref_opsets.get(d) != tgt_opsets.get(d):
            opset_diffs[d] = {
                ref_label: ref_opsets.get(d),
                target_label: tgt_opsets.get(d),
            }
    if opset_diffs:
        diffs["opset_diffs"] = opset_diffs

    # 特殊节点的增减
    ref_special = reference.get("special_nodes", {})
    tgt_special = target.get("special_nodes", {})
    special_diffs = {}
    for node_type in set(list(ref_special.keys()) + list(tgt_special.keys())):
        ref_cnt = len(ref_special.get(node_type, []))
        tgt_cnt = len(tgt_special.get(node_type, []))
        if ref_cnt != tgt_cnt:
            special_diffs[node_type] = {
                ref_label: ref_cnt,
                target_label: tgt_cnt,
                "diff": tgt_cnt - ref_cnt,
            }
    if special_diffs:
        diffs["special_node_diffs"] = special_diffs

    # Initializer 数量差异
    if reference["initializer_count"] != target["initializer_count"]:
        diffs["initializer_count"] = {
            ref_label: reference["initializer_count"],
            target_label: target["initializer_count"],
            "diff": target["initializer_count"] - reference["initializer_count"],
        }

    return diffs


def main():
    print("=" * 60)
    print("ONNX 图差异分析")
    print("=" * 60)

    all_results = {}

    for model_key in ["mobilenet_v3_small", "resnet50"]:
        original_path = ORIGINAL_ONNX[model_key]
        if not original_path.exists():
            print(f"[WARN] 原始 ONNX 不存在: {original_path}")
            continue

        print(f"\n--- {model_key} ---")
        print(f"原始 ONNX: {original_path}")
        ref_analysis = analyze_onnx(original_path)
        print(f"  节点数: {ref_analysis['total_nodes']}, "
              f"文件大小: {ref_analysis['file_size_mb']:.2f} MB, "
              f"算子类型: {list(ref_analysis['op_counts'].keys())}")

        all_results[model_key] = {
            "original": ref_analysis,
            "variants": {},
            "diffs": {},
        }

        for variant in VARIANT_ONNX_PATHS:
            variant_path = MODELS_DIR / f"{model_key}_{variant}.onnx"
            if not variant_path.exists():
                print(f"  [SKIP] {variant}: 文件不存在")
                continue

            var_analysis = analyze_onnx(variant_path)
            diffs = diff_two_onnx(ref_analysis, var_analysis,
                                   ref_label="original", target_label=variant)

            all_results[model_key]["variants"][variant] = var_analysis
            all_results[model_key]["diffs"][variant] = diffs

            if diffs:
                print(f"  {variant}: 节点数={var_analysis['total_nodes']}, "
                      f"大小={var_analysis['file_size_mb']:.2f}MB")
                for diff_key, diff_val in diffs.items():
                    print(f"    - {diff_key}: {diff_val}")
            else:
                print(f"  {variant}: ❌ 与原始 ONNX 无差异 (可能表示优化未生效)")

    # 保存结果
    output_path = RESULTS_DIR / "onnx_diff_analysis.json"
    output_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False)
    )
    print(f"\n结果已保存: {output_path}")
    return all_results


if __name__ == "__main__":
    main()
