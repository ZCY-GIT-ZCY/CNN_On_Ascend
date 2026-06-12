"""
使用 Ascend 单算子测试框架执行编译好的 TBE 算子。
"""
import sys, os, json, time
import numpy as np

# 添加测试框架路径
TOOLS_DIR = "/usr/local/Ascend/ascend-toolkit/8.0.0/tools/msaicerr"
sys.path.insert(0, TOOLS_DIR)

from ms_interface.single_op_test_frame.common.ascend_tbe_op import (
    AscendOpKernel, AscendOpKernelRunner, AscendOpKernelParam
)

# 内核文件
KERNEL_DIR = "/home/HwHiAiUser/Desktop/CNN/Opt_Analyze/data/kernel_meta"
KERNEL_NAME = "hard_sigmoid_mul_fused"
BIN_PATH = os.path.join(KERNEL_DIR, f"{KERNEL_NAME}.o")
JSON_PATH = os.path.join(KERNEL_DIR, f"{KERNEL_NAME}.json")
SHAPE = (1, 16, 112, 112)
DTYPE = np.float16

# 参考实现
def hard_sigmoid_mul_ref(x, scale):
    return np.clip(x / 6.0 + 0.5, 0, 1) * scale

def main():
    print("=" * 60)
    print("单算子执行测试: HardSigmoid+Mul 融合")
    print("=" * 60)

    # 加载内核
    kernel = AscendOpKernel(BIN_PATH, JSON_PATH)
    print(f"[OK] 内核加载: {KERNEL_NAME}, blockDim={kernel.block_dim}")

    # 准备测试数据
    np.random.seed(42)
    x_np = np.random.uniform(-2, 4, size=SHAPE).astype(DTYPE)
    s_np = np.random.uniform(0, 1, size=SHAPE).astype(DTYPE)

    # 参考
    ref = hard_sigmoid_mul_ref(x_np, s_np)

    # 创建 Runner
    runner = AscendOpKernelRunner(device_id=0)

    try:
        # 拷贝输入到设备
        x_dev = runner.build_kernel_param(x_np)
        s_dev = runner.build_kernel_param(s_np)

        # Warmup
        print("\n[Warmup] ...")
        for i in range(10):
            runner.kernel_input_ref = []
            output = runner.build_kernel_param(np.zeros(SHAPE, dtype=DTYPE))
            out = runner.run(kernel, inputs=[x_dev, s_dev])
            runner._kernel_params = [x_dev, s_dev, output]

        # Benchmark
        print("[Benchmark] 100 次推理...")
        times = []
        for i in range(100):
            t0 = time.perf_counter_ns()
            runner.kernel_input_ref = []
            output = runner.build_kernel_param(np.zeros(SHAPE, dtype=DTYPE))
            runner.run(kernel, inputs=[x_dev, s_dev])
            runner._kernel_params = [x_dev, s_dev, output]
            times.append((time.perf_counter_ns() - t0) / 1e6)

        avg_ms = np.mean(times)
        std_ms = np.std(times)
        print(f"[OK] 平均延迟: {avg_ms:.4f}ms ± {std_ms:.4f}ms")

        # 获取输出并验证
        output_data = output.get()
        max_err = np.max(np.abs(output_data.astype(np.float32) - ref.astype(np.float32)))
        cos_sim = np.dot(output_data.flatten(), ref.flatten()) / (
            np.linalg.norm(output_data) * np.linalg.norm(ref)
        )
        print(f"[数值验证] max_err={max_err:.6f}, cosine={cos_sim:.6f}")

        if max_err < 0.01 and cos_sim > 0.999:
            print("[✅] 数值正确!")
        else:
            print("[⚠️] 数值偏差较大，请检查")

        print(f"\n  输入范围: [{x_np.min():.4f}, {x_np.max():.4f}]")
        print(f"  缩放范围: [{s_np.min():.4f}, {s_np.max():.4f}]")
        print(f"  输出范围: [{output_data.min():.4f}, {output_data.max():.4f}]")
        print(f"  参考范围: [{ref.min():.4f}, {ref.max():.4f}]")

    finally:
        runner.__exit__(None, None, None)
        print("\n[资源已释放]")

if __name__ == "__main__":
    main()
