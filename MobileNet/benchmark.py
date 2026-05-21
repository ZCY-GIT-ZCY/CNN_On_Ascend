"""
昇腾 NPU 推理性能基准测试
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common', 'acllite_utils'))

from acllite_resource import AclLiteResource
from acllite_model import AclLiteModel


def benchmark(model_path: str, warmup: int = 10, iterations: int = 100):
    """
    推理性能基准测试

    Args:
        model_path: OM 模型路径
        warmup: 预热次数
        iterations: 正式测试次数
    """
    print("=" * 50)
    print("MobileNet 昇腾 NPU 性能基准测试")
    print("=" * 50)

    # 初始化资源
    print(f"[1/5] 初始化 NPU 资源...")
    resource = AclLiteResource()
    resource.init()

    print(f"[2/5] 加载模型: {model_path}")
    model = AclLiteModel(model_path)

    # 准备输入数据（全零矩阵）
    print(f"[3/5] 准备测试数据...")
    input_data = np.zeros((224, 224, 3), dtype=np.uint8)

    # 预热阶段
    print(f"[4/5] 预热中 ({warmup} 次)...")
    for _ in range(warmup):
        model.execute([input_data])

    # 正式测试
    print(f"[5/5] 性能测试中 ({iterations} 次)...")
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        model.execute([input_data])
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        if (i + 1) % 20 == 0:
            print(f"      完成 {i + 1}/{iterations} 次...")

    # 统计结果
    times = np.array(times)

    print("\n" + "=" * 50)
    print("  性能基准测试结果")
    print("=" * 50)
    print(f"  测试次数:       {iterations}")
    print(f"  平均延迟:       {times.mean():.2f} ms")
    print(f"  最小延迟:       {times.min():.2f} ms")
    print(f"  最大延迟:       {times.max():.2f} ms")
    print(f"  吞吐量:         {1000 / times.mean():.2f} FPS")
    print(f"  延迟标准差:     {times.std():.2f} ms")
    print(f"  P50 延迟:       {np.percentile(times, 50):.2f} ms")
    print(f"  P90 延迟:       {np.percentile(times, 90):.2f} ms")
    print(f"  P99 延迟:       {np.percentile(times, 99):.2f} ms")
    print("=" * 50)

    # 释放资源
    del model
    del resource

    return {
        'mean_ms': times.mean(),
        'min_ms': times.min(),
        'max_ms': times.max(),
        'fps': 1000 / times.mean(),
        'std_ms': times.std()
    }


if __name__ == "__main__":
    benchmark("./mobilenetv3_aipp.om")
