"""
创建包含自定义 HardSigmoidMul 算子的 ONNX 测试模型。
然后编译并 benchmark 验证。
"""
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
import subprocess, sys, time, json
from pathlib import Path

OPT_DIR = Path("/home/HwHiAiUser/Desktop/CNN/Opt_Analyze")
DATA_DIR = OPT_DIR / "data"
SCRIPT_DIR = OPT_DIR / "scripts"
MODELS_DIR = Path("/home/HwHiAiUser/Desktop/CNN/Optimization/models")
ATC_BIN = "/usr/local/Ascend/ascend-toolkit/latest/bin/atc"
AIPP = "/home/HwHiAiUser/Desktop/CNN/MobileNet/aipp.cfg"

# ===== Step 1: 创建小型 ONNX 测试图 =====
# 一个简单的网：Input → HardSigmoidMul → Output
# HardSigmoidMul是自定义算子，在onnx域中

def create_test_model(shape=(1, 16, 112, 112)):
    batch, ch, h, w = shape

    # 输入
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT16, shape)
    scale = helper.make_tensor_value_info("scale", TensorProto.FLOAT16, shape)
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT16, shape)

    # 自定义算子节点 - 使用onnx域
    hsigmoid_mul = helper.make_node(
        "HardSigmoidMul",           # op_type - 与注册名称一致
        inputs=["input", "scale"],
        outputs=["output"],
        name="HardSigmoidMul_0",
        domain="",                  # 空域 = onnx原生域
    )

    # 构建图
    graph = helper.make_graph(
        [hsigmoid_mul],
        "test_hardsigmoid_mul",
        [x, scale],
        [y],
    )

    # 创建模型
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7

    return model

# ===== Step 2: 创建包含权重的真实测试图 =====
def create_test_model_with_weights():
    """创建一个更完整的测试图：
    input → Conv(1x1) → HardSigmoidMul(scale=Conv_out) → output

    这样 HardSigmoidMul 的输入来自前面的 Conv，
    更接近真实模型中的使用场景。
    """
    shape = (1, 8, 56, 56)
    batch, ch, h, w = shape

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT16, shape)
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT16, shape)

    # Conv权重
    w_np = np.random.randn(8, 8, 1, 1).astype(np.float16)
    w_init = numpy_helper.from_array(w_np, name="conv_weight")

    # Conv节点
    conv = helper.make_node(
        "Conv", inputs=["input", "conv_weight"], outputs=["conv_out"],
        name="conv_0", kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0],
        group=1,
    )

    # 自定义融合算子
    hsigmoid_mul = helper.make_node(
        "HardSigmoidMul",
        inputs=["input", "conv_out"],  # x, scale
        outputs=["output"],
        name="HardSigmoidMul_0",
    )

    graph = helper.make_graph(
        [conv, hsigmoid_mul],
        "test_fused", [x], [y],
        initializer=[w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    return model


def compile_model(model, name, onnx_path, om_path):
    """ATC编译"""
    onnx.save(model, str(onnx_path))
    print(f"[{name}] ONNX saved: {onnx_path}")

    cmd = [
        ATC_BIN, f"--model={onnx_path}", "--framework=5",
        f"--output={om_path.with_suffix('')}",
        "--soc_version=Ascend310B1", "--input_format=NCHW",
        "--input_shape=input:1,8,56,56",
        "--precision_mode=allow_mix_precision",
        "--op_select_implmode=high_performance",
        "--enable_small_channel=1",
        "--log=error",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if proc.returncode == 0 and om_path.exists():
        print(f"[{name}] ✅ ATC compile OK: {om_path.name} ({om_path.stat().st_size} bytes)")
        return True
    else:
        print(f"[{name}] ❌ ATC compile FAILED")
        print(f"  stderr: {proc.stderr[-500:]}")
        return False


def bench_model(om_path, n=50):
    """简单 benchmark"""
    import sys as _sys
    _sys.path.insert(0, "/home/HwHiAiUser/Desktop/CNN/common/acllite_utils")
    from acllite_resource import AclLiteResource
    from acllite_model import AclLiteModel

    r = AclLiteResource(); r.init()
    m = AclLiteModel(str(om_path))
    inp = np.zeros((56, 56, 3), dtype=np.uint8)

    for _ in range(10): m.execute([inp])

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        m.execute([inp])
        times.append((time.perf_counter() - t0) * 1000)

    del m; del r
    return np.mean(times), np.std(times)


def main():
    print("=" * 60)
    print("自定义算子 HardSigmoidMul 测试")
    print("=" * 60)

    # 创建测试模型
    model = create_test_model_with_weights()

    # 编译
    name = "hard_sigmoid_mul_test"
    onnx_path = DATA_DIR / f"{name}.onnx"
    om_path = MODELS_DIR / f"{name}.om"

    ok = compile_model(model, name, onnx_path, om_path)

    if ok:
        mean, std = bench_model(om_path)
        print(f"\n  Benchmark: {mean:.4f}ms ±{std:.4f}ms")

        # 数值验证（创建等价的纯ONNX图做对比）
        print(f"\n  ✅ 算子执行成功！")
        print(f"  数值验证需要另一个不含自定义算子的ONNX作参考")

    print("\nDone.")


if __name__ == "__main__":
    main()
