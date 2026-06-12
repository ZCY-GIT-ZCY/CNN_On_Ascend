"""
测试 TBE 融合算子：通过 AscendCL 直接执行编译好的 kernel。
"""
import acl
import numpy as np
import os
import sys
import time
import json
from pathlib import Path

KERNEL_DIR = Path("/home/HwHiAiUser/Desktop/CNN/Opt_Analyze/data/kernel_meta")
KERNEL_NAME = "hard_sigmoid_mul_fused"
KERNEL_FILE = str(KERNEL_DIR / f"{KERNEL_NAME}.o")
JSON_FILE = str(KERNEL_DIR / f"{KERNEL_NAME}.json")

SHAPE = (1, 16, 112, 112)
DTYPE = np.float16
N_ITERATIONS = 100


def hard_sigmoid_mul_np(x, scale):
    """NumPy 参考实现。"""
    return np.clip(x / 6.0 + 0.5, 0, 1) * scale


def test_via_acl():
    print("=" * 60)
    print("TBE 融合算子执行测试")
    print("=" * 60)

    # 初始化 ACL
    ret = acl.init()
    assert ret == 0, f"acl.init failed: {ret}"
    ret = acl.rt.set_device(0)
    assert ret == 0, f"acl.rt.set_device failed: {ret}"
    ctx, ret = acl.rt.create_context(0)
    assert ret == 0, f"acl.rt.create_context failed: {ret}"
    print("[OK] ACL initialized")

    # 检查 kernel 文件
    assert os.path.exists(KERNEL_FILE), f"Kernel file not found: {KERNEL_FILE}"
    assert os.path.exists(JSON_FILE), f"JSON file not found: {JSON_FILE}"
    print(f"[OK] Kernel: {KERNEL_FILE} ({os.path.getsize(KERNEL_FILE)} bytes)")

    # 加载 kernel JSON 元信息
    with open(JSON_FILE) as f:
        meta = json.load(f)
    block_dim = meta.get("blockDim", 1)
    print(f"[INFO] Kernel: blockDim={block_dim}, coreType={meta.get('coreType', '?')}")

    # 准备数据
    np.random.seed(42)
    x_np = np.random.uniform(-2, 4, size=SHAPE).astype(DTYPE)
    s_np = np.random.uniform(0, 1, size=SHAPE).astype(DTYPE)

    # 分配 Device 内存
    n_bytes = x_np.nbytes
    x_dev, ret = acl.rt.malloc(n_bytes, acl.const.ACL_MEM_MALLOC_HUGE_FIRST)
    s_dev, ret = acl.rt.malloc(n_bytes, acl.const.ACL_MEM_MALLOC_HUGE_FIRST)
    y_dev, ret = acl.rt.malloc(n_bytes, acl.const.ACL_MEM_MALLOC_HUGE_FIRST)
    print(f"[OK] Device memory allocated: {n_bytes} bytes × 3")

    # Host→Device 数据拷贝
    x_ptr = acl.util.numpy_to_ptr(x_np)
    s_ptr = acl.util.numpy_to_ptr(s_np)
    acl.rt.memcpy(x_dev, n_bytes, x_ptr, n_bytes, acl.const.ACL_MEMCPY_HOST_TO_DEVICE)
    acl.rt.memcpy(s_dev, n_bytes, s_ptr, n_bytes, acl.const.ACL_MEMCPY_HOST_TO_DEVICE)

    # 使用 acl.op.call 加载并执行 kernel
    # 方法：通过 acl.op.create_operator 手动注册再调用
    try:
        print("\n[Trying acl.op.call...]")
        # acl.op.call 需要 tiling 参数
        # 先执行一次（warmup）
        ret = acl.op.call(
            KERNEL_NAME,
            [x_dev, s_dev],  # inputs (raw device ptrs)
            [y_dev],          # outputs (raw device ptrs)
            block_dim,
        )
        print(f"  First call ret: {ret}")
    except Exception as e:
        print(f"  acl.op.call failed: {e}")

    # 如果 acl.op.call 不工作，尝试另一种方法
    try:
        print("\n[Trying acl.op.execute...]")
        desc = acl.op.create_attr()
        # 需要 op_type 注册
        stream, ret = acl.rt.create_stream()
        print(f"  Stream created: {ret}")
    except Exception as e:
        print(f"  acl.op.execute failed: {e}")

    # 读取设备内存做参考
    y_result = np.zeros(SHAPE, dtype=DTYPE)
    y_ptr = acl.util.numpy_to_ptr(y_result)
    acl.rt.memcpy(y_ptr, n_bytes, y_dev, n_bytes, acl.const.ACL_MEMCPY_DEVICE_TO_HOST)

    # NumPy 参考
    ref = hard_sigmoid_mul_np(x_np, s_np)

    print(f"\n  Input range:   [{x_np.min():.4f}, {x_np.max():.4f}]")
    print(f"  Scale range:   [{s_np.min():.4f}, {s_np.max():.4f}]")
    print(f"  Ref range:     [{ref.min():.4f}, {ref.max():.4f}]")

    # 释放
    acl.rt.free(x_dev)
    acl.rt.free(s_dev)
    acl.rt.free(y_dev)

    # 清理
    acl.rt.destroy_context(ctx)
    acl.rt.reset_device(0)
    acl.finalize()
    print("\n[OK] ACL finalized")


if __name__ == "__main__":
    test_via_acl()
