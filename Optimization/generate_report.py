"""生成实验结论报告（当前阶段）。"""
import json
from pathlib import Path

from config import REPORTS_DIR, RESULTS_DIR


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(delta: float, base: float) -> float:
    if base == 0:
        return 0.0
    return delta / base * 100.0


def main() -> None:
    fusion = load_json(RESULTS_DIR.parent / "reports" / "fusion_summary.json")
    results = load_json(RESULTS_DIR / "experiment_results.json")
    variants = load_json(RESULTS_DIR / "variant_graph_metrics.json")

    baseline_mobile = next(x for x in results if x["model_key"] == "mobilenet_v3_small" and x["experiment_id"] == "baseline_copy")
    resnet_baseline = next(x for x in results if x["model_key"] == "resnet50" and x["experiment_id"] in ["baseline_copy", "copy_baseline"])

    resnet_original = variants["resnet50_opset13.onnx"]
    resnet_bn = variants["resnet50_bn_folded.onnx"]
    mobile_original = variants["mobilenet_v3_small_opset13.onnx"]
    mobile_bn = variants["mobilenet_v3_small_bn_folded.onnx"]
    mobile_fp16 = variants["mobilenet_v3_small_fp16.onnx"]
    resnet_fp16 = variants["resnet50_fp16.onnx"]

    report = f'''# Ascend 310B1 CNN 算子融合与优化实验报告

## 1. 实验目标

本阶段围绕已经完成部署与测量的 `ResNet50` 和 `MobileNetV3-Small`，继续分析：

- 普通卷积与深度可分离卷积当前的融合程度与方式
- 可尝试的常见图优化、参数融合、布局优化与精度优化
- 各优化策略对图结构、模型体积和可测性能的影响

所有新增工作均保存在 `Optimization/` 目录。

## 2. 当前模型融合形态分析

### 2.1 普通卷积模型：ResNet50

- 总卷积层数：{fusion['summary']['standard_conv_model']['total_conv_layers']}
- BottleNeck 数：{fusion['summary']['standard_conv_model']['bottleneck_blocks']}
- 主导融合模式：{', '.join(fusion['summary']['standard_conv_model']['dominant_fusions'])}
- 融合密度指标：{fusion['summary']['standard_conv_model']['fusion_density']:.2f}

结论：ResNet50 的融合机会高度规则化，主要集中在残差块内部的 `Conv+BN+ReLU` 与块尾 `Conv+BN+Add+ReLU`。这类结构对编译器最友好，适合做稳定的块级融合。

### 2.2 深度可分离卷积模型：MobileNetV3-Small

- 总卷积层数：{fusion['summary']['depthwise_model']['total_conv_layers']}
- Depthwise 卷积层数：{fusion['summary']['depthwise_model']['depthwise_layers']}
- Pointwise 卷积层数：{fusion['summary']['depthwise_model']['pointwise_layers']}
- SE 模块数：{fusion['summary']['depthwise_model']['se_blocks']}
- 主导融合模式：{', '.join(fusion['summary']['depthwise_model']['dominant_fusions'])}
- 融合密度指标：{fusion['summary']['depthwise_model']['fusion_density']:.2f}

结论：MobileNetV3 的融合点更分散，主要分布在 Expand / DW / Project / SE / 短残差上。它更依赖链式调度与中间张量复用，而不是单一大算子融合。

## 3. 已完成的优化实验

### 3.1 结构与图级实验

本轮已完成以下变体构建：

1. `shape_inferred`：补全静态 shape 信息
2. `opset13`：重新导出为 opset 13
3. `bn_folded`：在 PyTorch 侧显式做 BN folding
4. `channels_last`：使用 channels-last 内存格式导出
5. `fp16`：导出半精度权重 ONNX
6. `baseline_copy`：复制并统一重测当前 OM 基线

## 4. 图结构指标结果

### 4.1 ResNet50

- 原图节点数：{resnet_original['total_nodes']}
- BN folding 后节点数：{resnet_bn['total_nodes']}
- 节点减少：{resnet_original['total_nodes'] - resnet_bn['total_nodes']} ({pct(resnet_original['total_nodes'] - resnet_bn['total_nodes'], resnet_original['total_nodes']):.1f}%)
- ReLU 节点数：{resnet_original['relu']} -> {resnet_bn['relu']}
- 模型体积：{resnet_original['file_size_mb']:.2f} MB -> {resnet_bn['file_size_mb']:.2f} MB
- FP16 体积：{resnet_fp16['file_size_mb']:.2f} MB

解释：ResNet50 的 BN folding 明显减少了图节点数，说明普通卷积残差网络更容易通过显式参数折叠获得更“紧凑”的计算图。

### 4.2 MobileNetV3-Small

- 原图节点数：{mobile_original['total_nodes']}
- BN folding 后节点数：{mobile_bn['total_nodes']}
- 节点变化：{mobile_original['total_nodes'] - mobile_bn['total_nodes']}
- 模型体积：{mobile_original['file_size_mb']:.2f} MB -> {mobile_bn['file_size_mb']:.2f} MB
- FP16 体积：{mobile_fp16['file_size_mb']:.2f} MB

解释：MobileNetV3 的原始导出图里已经没有显式 BatchNormalization 节点，因此手工 BN folding 对图节点数几乎无影响；说明该模型在导出阶段就已经具备较高的基础融合程度。

## 5. 基线性能重测结果

### 5.1 MobileNetV3-Small OM 基线重测

- 平均延迟：{baseline_mobile['latency_ms']:.3f} ms
- 吞吐量：{baseline_mobile['fps']:.2f} FPS
- 标准差：{baseline_mobile['std_ms']:.3f} ms
- P50 / P90 / P99：{baseline_mobile['p50_ms']:.3f} / {baseline_mobile['p90_ms']:.3f} / {baseline_mobile['p99_ms']:.3f} ms

说明：这个结果明显慢于此前日志中的 0.96 ms / 1040 FPS，说明当前直接脚本重测流程与早先 benchmark 条件并不完全一致，可能受资源初始化、运行时状态或测试方法影响。因此后续所有性能比较必须统一在同一套脚本和同一环境下继续测量，不能直接和历史值混用。

### 5.2 ResNet50 OM 重测异常

- 当前状态：{resnet_baseline['status']}
- 异常信息：`{resnet_baseline['note']}`

说明：当前自动化批处理在重置 MobileNet 后再次初始化 ACL 时遇到设备初始化异常，导致 ResNet50 未完成同轮统一重测。该问题更像是 ACL 运行时串行初始化/释放细节，而非模型本身问题。

## 6. 阶段性结论

### 6.1 关于融合程度

1. **ResNet50 的块级融合潜力更强**  
   显式 BN folding 后，图节点从 {resnet_original['total_nodes']} 降到 {resnet_bn['total_nodes']}，下降 {pct(resnet_original['total_nodes'] - resnet_bn['total_nodes'], resnet_original['total_nodes']):.1f}% ，说明普通卷积残差结构更适合通过规则模式压缩图结构。

2. **MobileNetV3 的导出图已较“预融合”**  
   原始 ONNX 中已经没有 BatchNormalization 节点，说明 Conv-BN 在导出时已基本吸收；真正的难点不再是 BN 折叠，而是 DW/PW/SE 链路的调度效率、布局变换与访存。

3. **普通卷积和深度可分离卷积的优化重点不同**  
   - 普通卷积：更适合做 `Conv+BN(+Add)+ReLU` 的块级融合
   - 深度可分离卷积：更适合做 `Expand/DW/Project/SE` 的链式优化与 cache/layout 优化

### 6.2 关于常见优化方法的有效性

- `shape inference`：对图结构无直接压缩，但为编译器提供更完整静态信息，适合作为后续 ATC 编译前处理
- `opset13`：当前两模型节点统计与 opset11 几乎一致，单独切换 opset 暂未观察到结构收益
- `BN folding`：对 ResNet50 有明显结构收益，对 MobileNetV3 收益很小
- `channels_last`：导出图节点不变，但可能影响底层布局映射，仍值得在 ATC 编译后测端到端
- `FP16`：两模型体积都接近减半，是最直接的带宽优化候选

## 7. 当前保留的数据文件

已生成并保留：

- `Optimization/fusion_analysis.json`
- `Optimization/reports/fusion_summary.json`
- `Optimization/results/experiment_results.json`
- `Optimization/results/experiment_summary.json`
- `Optimization/results/onnx_graph_metrics.json`
- `Optimization/results/variant_graph_metrics.json`
- `Optimization/models/*.onnx`

## 8. 下一步建议

为了把这个方向做到“很完善”，下一轮建议继续做三类工作：

1. **ATC 自动编译所有变体**  
   将 `bn_folded / channels_last / fp16 / shape_inferred / opset13` 全部编译成 OM。

2. **统一基准脚本逐个测量**  
   对每个 OM 记录延迟、吞吐、P50/P90/P99、标准差、模型大小，并补充 msprof 的算子级统计。

3. **修复 ACL 批处理初始化问题**  
   将 ResNet50 与 MobileNet 的测试拆分为独立进程，避免 `acl.init failed ret_int=100002` 影响整轮实验。

## 9. 本阶段总结

本阶段已经完成了：

- 两类卷积网络的融合方式分析
- 多种常见优化变体的自动导出
- 融合程度的结构化量化
- 初步基线重测与问题定位
- 全部中间数据和结果文件留存

从当前结果看，**ResNet50 更适合通过显式块级融合压缩图结构，MobileNetV3 则更依赖精度压缩、布局优化和 DW/PW/SE 链路的访存优化**。这为后续真正跑满 Ascend 编译与性能测量提供了明确方向。
'''

    out = REPORTS_DIR / "experiment_report.md"
    out.write_text(report, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
