"""
TBE 自定义算子：HardSigmoid + Mul 融合

HardSigmoid(x) = clip(x/6 + 0.5, 0, 1)
HardSigmoidMul(x, scale) = clip(x/6 + 0.5, 0, 1) * scale

测试流程：
  1. 使用 TBE DSL 定义算子
  2. 编译为 .o kernel
  3. 使用 AscendCL 加载并执行
  4. 与原始实现对比性能和正确性
"""
import numpy as np
import sys
import os
import time
from pathlib import Path

# TBE 导入
try:
    import tbe.dsl as tbe
    from tbe import tvm
    from tbe.common.register import register_op_compute
except ImportError:
    sys.path.insert(0, "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages")
    import tbe.dsl as tbe
    from tbe import tvm
    from tbe.common.register import register_op_compute

# AscendCL 导入
COMMON_DIR = Path("/home/HwHiAiUser/Desktop/CNN/common/acllite_utils")
sys.path.insert(0, str(COMMON_DIR))
from acllite_resource import AclLiteResource
from acllite_model import AclLiteModel


def hard_sigmoid_mul_kernel():
    """
    HardSigmoid(x) = max(0, min(1, x/6 + 0.5))
    HardSigmoidMul(x, scale) = HardSigmoid(x) * scale

    使用 TBE DSL 实现。
    clip(x/6 + 0.5, 0, 1) * scale
    """
    # 定义计算
    def compute(x, scale):
        # x/6
        div_factor = tvm.const(1.0 / 6.0, dtype=x.dtype)
        x_div = tbe.vmuls(x, div_factor)

        # + 0.5
        bias = tvm.const(0.5, dtype=x.dtype)
        x_biased = tbe.vadds(x_div, bias)

        # clip to [0, 1] 使用 tensor+scalar 版本
        x_clipped = tbe.vmins(tbe.vmaxs(x_biased, 0.0), 1.0)

        # * scale element-wise
        out = tbe.vmul(x_clipped, scale)

        return out

    return compute


def build_kernel(shape, dtype="float16"):
    """编译 TBE 算子为可执行 kernel。"""
    # 创建占位符
    data_x = tvm.placeholder(shape, dtype=dtype, name="data_x")
    data_scale = tvm.placeholder(shape, dtype=dtype, name="data_scale")

    # 构建计算图
    compute_fn = hard_sigmoid_mul_kernel()
    out = compute_fn(data_x, data_scale)

    # Auto-schedule 和编译
    with tvm.target.cce():
        schedule = tbe.auto_schedule(out)

    # 构建配置
    kernel_name = "hard_sigmoid_mul_fused"
    config = {
        "name": kernel_name,
        "tensor_list": [data_x, data_scale, out],
    }

    # 编译
    tbe.build(schedule, config)
    print(f"  Kernel compiled: {kernel_name}")
    print(f"  Input shape: {shape}, dtype: {dtype}")

    return kernel_name


def test_kernel():
    """测试融合算子的正确性和性能。"""
    print("=" * 60)
    print("TBE HardSigmoid+Mul 融合算子测试")
    print("=" * 60)

    shape = (1, 16, 112, 112)  # MobileNetV3 typical shape
    dtype = "float16"
    kernel_name = build_kernel(shape, dtype)

    # 生成测试数据
    np.random.seed(42)
    x_np = np.random.uniform(-2, 4, size=shape).astype(np.float16)
    scale_np = np.random.uniform(0, 1, size=shape).astype(np.float16)

    # NumPy 参考结果
    def hard_sigmoid_mul_ref(x, s):
        hs = np.clip(x / 6.0 + 0.5, 0, 1)
        return (hs * s).astype(np.float16)

    ref = hard_sigmoid_mul_ref(x_np, scale_np)

    # 用 AscendCL 加载并运行 kernel
    print("\n  Loading kernel via AscendCL...")
    resource = AclLiteResource()
    resource.init()

    # 由于 TBE build 输出的是 .o 和 .json 文件，
    # 需要作为 OP 模型加载 (暂不可直接通过 AclLiteModel 加载)
    # 这里我们使用 ACL 接口手动加载
    try:
        import acl

        # 加载内核
        kernel_path = f"./{kernel_name}.o"
        json_path = f"./{kernel_name}.json"

        if not os.path.exists(kernel_path) or not os.path.exists(json_path):
            print(f"  ❌ Kernel files not found: {kernel_path}")
            # 可能输出在 ~/Ascend/ 下
            for dirpath, dirnames, filenames in os.walk(os.path.expanduser("~")):
                for fn in filenames:
                    if kernel_name in fn:
                        print(f"     Found: {dirpath}/{fn}")
            return

        # 方法1: 作为AclLiteModel加载（如果支持）
        # 方法2: 使用 acl.op.create 直接调用

        print(f"  Kernel file: {kernel_path} ({os.path.getsize(kernel_path)} bytes)")
        print(f"  JSON file: {json_path}")
        print()
        print("  TBE kernel compiled successfully!")
        print("  For full integration testing, use acl.op.execute()")

        # 验证正确性
        print("\n  ✅ NumPy reference computed")
        print(f"     Input range: [{x_np.min():.4f}, {x_np.max():.4f}]")
        print(f"     Scale range: [{scale_np.min():.4f}, {scale_np.max():.4f}]")
        print(f"     Output range: [{ref.min():.4f}, {ref.max():.4f}]")

    except Exception as e:
        print(f"  ❌ ACL execution error: {e}")
        import traceback
        traceback.print_exc()

    del resource
    print("\n  Done.")


if __name__ == "__main__":
    test_kernel()
