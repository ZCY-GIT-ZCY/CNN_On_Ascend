"""实验配置与公共常量。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
MOBILENET_DIR = WORKSPACE / "MobileNet"
RESNET_DIR = WORKSPACE / "ResNet"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
SCRIPTS_DIR = ROOT / "scripts"
DATA_DIR = ROOT / "data"

for path in [RESULTS_DIR, MODELS_DIR, REPORTS_DIR, SCRIPTS_DIR, DATA_DIR]:
    path.mkdir(parents=True, exist_ok=True)

BASELINES = {
    "mobilenet_v3_small": {
        "model_dir": MOBILENET_DIR,
        "onnx": MOBILENET_DIR / "mobilenetv3.onnx",
        "om": MOBILENET_DIR / "mobilenetv3_aipp.om",
        "input_name": "input_image",
        "input_shape": (1, 3, 224, 224),
        "benchmark_py": MOBILENET_DIR / "benchmark.py",
        "type": "depthwise_separable",
        "baseline_latency_ms": 0.96,
        "baseline_fps": 1040,
    },
    "resnet50": {
        "model_dir": RESNET_DIR,
        "onnx": RESNET_DIR / "resnet50.onnx",
        "om": RESNET_DIR / "resnet50_aipp.om",
        "input_name": "input_image",
        "input_shape": (1, 3, 224, 224),
        "benchmark_py": RESNET_DIR / "benchmark.py",
        "type": "standard_conv",
        "baseline_latency_ms": 3.81,
        "baseline_fps": 262,
    },
}

EXPERIMENTS = [
    {
        "id": "baseline_copy",
        "name": "Baseline AIPP OM",
        "category": "baseline",
        "description": "直接复用当前已部署的 AIPP OM 模型，作为所有优化的参考基线。",
    },
    {
        "id": "onnx_simplified",
        "name": "ONNX Simplify",
        "category": "graph_rewrite",
        "description": "使用 onnxsim 做常量折叠和冗余节点清理，观察编译后图融合是否增加。",
    },
    {
        "id": "onnx_shape_inferred",
        "name": "Shape Inference + Simplify",
        "category": "graph_rewrite",
        "description": "补全静态 shape 信息并再次简化，帮助 ATC 做更激进的图级融合。",
    },
    {
        "id": "onnx_to_opset13",
        "name": "Re-export Opset13",
        "category": "graph_rewrite",
        "description": "重新导出为 opset13，验证不同算子表达方式对 ATC 融合的影响。",
    },
    {
        "id": "manual_bn_fold",
        "name": "Manual BN Folding",
        "category": "parameter_fusion",
        "description": "在 PyTorch 侧显式做 Conv/Linear 与 BN 折叠，再导出 ONNX。",
    },
    {
        "id": "channels_last_export",
        "name": "Channels-Last Export",
        "category": "layout_optimization",
        "description": "以 channels_last 内存布局导出，尝试减少编译后的布局变换。",
    },
    {
        "id": "half_precision_export",
        "name": "FP16 Export",
        "category": "precision_optimization",
        "description": "在导出前将权重转为 float16，降低带宽压力和模型体积。",
    },
]
