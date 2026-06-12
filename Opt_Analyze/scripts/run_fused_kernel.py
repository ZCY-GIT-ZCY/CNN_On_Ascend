"""
通过 AscendCL 的 op API 执行编译好的 TBE 融合算子 kernel。
"""
import acl
import numpy as np
import os, sys, json, time
from pathlib import Path

KERNEL_DIR = Path("/home/HwHiAiUser/Desktop/CNN/Opt_Analyze/data/kernel_meta")
KERNEL_NAME = "hard_sigmoid_mul_fused"
SHAPE = (1, 16, 112, 112)  # MobileNetV3 typical SE feature map
DTYPE = np.float16
N_WARMUP = 10
N_ITER = 100

def hard_sigmoid_mul_ref(x, scale):
    return np.clip(x / 6.0 + 0.5, 0, 1) * scale

def main():
    acl.init()
    acl.rt.set_device(0)

    # 准备数据
    np.random.seed(42)
    x_host = np.random.uniform(-2, 4, size=SHAPE).astype(DTYPE)
    s_host = np.random.uniform(0, 1, size=SHAPE).astype(DTYPE)
    y_host = np.zeros(SHAPE, dtype=DTYPE)
    n_bytes = x_host.nbytes

    # Device 内存
    x_dev = acl.rt.malloc(n_bytes, 0)
    s_dev = acl.rt.malloc(n_bytes, 0)
    y_dev = acl.rt.malloc(n_bytes, 0)

    # Host→Device
    acl.rt.memcpy(x_dev, n_bytes, acl.util.numpy_to_ptr(x_host), n_bytes, 1)
    acl.rt.memcpy(s_dev, n_bytes, acl.util.numpy_to_ptr(s_host), n_bytes, 1)

    # 注册 kernel
    json_path = str(KERNEL_DIR / f"{KERNEL_NAME}.json")
    with open(json_path) as f:
        meta = json.load(f)
    block_dim = meta.get("blockDim", 1)
    print(f"Kernel: {KERNEL_NAME}, blockDim={block_dim}")

    # 方式1: acl.op.call - 直接调用
    print("\nTrying acl.op.call...")
    try:
        # 准备 args: [x, scale, y] device pointers
        args_list = [x_dev, s_dev, y_dev]
        for i in range(N_WARMUP):
            ret = acl.op.call(KERNEL_NAME, args_list, args_list, block_dim)
        times = []
        for i in range(N_ITER):
            t0 = time.perf_counter()
            ret = acl.op.call(KERNEL_NAME, args_list, args_list, block_dim)
            times.append((time.perf_counter() - t0) * 1000)
        print(f"  ret={ret}")
        print(f"  Latency: mean={np.mean(times):.4f}ms, std={np.std(times):.4f}ms")

        # 读回结果
        acl.rt.memcpy(acl.util.numpy_to_ptr(y_host), n_bytes, y_dev, n_bytes, 0)
        ref = hard_sigmoid_mul_ref(x_host, s_host)
        max_err = np.max(np.abs(y_host.astype(np.float32) - ref.astype(np.float32)))
        print(f"  Max error: {max_err:.6f}")
    except Exception as e:
        print(f"  Failed: {e}")

    # 方式2: acl.op.execute
    print("\nTrying acl.op.execute...")
    try:
        # 注册op
        op_handle = acl.op.create_handle(KERNEL_NAME, "")
        # 设置参数
        acl.op.set_kernel_args(op_handle, len(args_list), args_list)
        # 执行
        for i in range(N_WARMUP):
            acl.op.execute_with_handle(op_handle)
        times = []
        for i in range(N_ITER):
            t0 = time.perf_counter()
            acl.op.execute_with_handle(op_handle)
            times.append((time.perf_counter() - t0) * 1000)
        print(f"  Latency: mean={np.mean(times):.4f}ms, std={np.std(times):.4f}ms")
        acl.op.destroy_handle(op_handle)
    except Exception as e:
        print(f"  Failed: {e}")

    # 清理
    acl.rt.free(x_dev)
    acl.rt.free(s_dev)
    acl.rt.free(y_dev)
    acl.rt.reset_device(0)
    acl.finalize()
    print("\nDone.")

if __name__ == "__main__":
    main()
