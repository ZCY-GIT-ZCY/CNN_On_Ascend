"""
常用工具函数

提供算子开发中常用的辅助功能。
"""
import time
import numpy as np
from typing import Callable, Any, Tuple
import functools


class Timer:
    """简单的计时器上下文管理器"""
    
    def __init__(self, name: str = "操作"):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        print(f"{self.name} 耗时: {self.elapsed*1000:.2f} ms")


def benchmark(
    func: Callable,
    *args,
    warmup: int = 3,
    iterations: int = 100,
    **kwargs
) -> Tuple[float, float]:
    """
    基准测试函数
    
    Args:
        func: 要测试的函数
        *args: 函数参数
        warmup: 预热次数
        iterations: 测试迭代次数
        **kwargs: 函数关键字参数
    
    Returns:
        (平均耗时, 标准差)
    """
    # 预热
    for _ in range(warmup):
        func(*args, **kwargs)
    
    # 计时测试
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        times.append(time.perf_counter() - start)
    
    return np.mean(times) * 1000, np.std(times) * 1000


def compare_operations(
    op_name: str,
    custom_func: Callable,
    reference_func: Callable,
    input_shapes: list,
    dtype=np.float32,
    **kwargs
) -> dict:
    """
    比较自定义算子和参考实现的性能和正确性
    
    Args:
        op_name: 算子名称
        custom_func: 自定义实现
        reference_func: 参考实现 (如 PyTorch 官方实现)
        input_shapes: 输入形状列表
        dtype: 数据类型
        **kwargs: 其他参数
    
    Returns:
        包含测试结果的字典
    """
    import torch
    
    results = {
        "op_name": op_name,
        "input_shapes": input_shapes,
        "dtype": str(dtype),
        "correct": False,
        "custom_time_ms": None,
        "reference_time_ms": None,
        "max_diff": None,
        "speedup": None
    }
    
    # 创建输入张量
    inputs = [
        torch.randn(shape, dtype=dtype) for shape in input_shapes
    ]
    
    # 测试参考实现
    ref_time, _ = benchmark(reference_func, *inputs, iterations=50)
    reference_output = reference_func(*inputs)
    results["reference_time_ms"] = ref_time
    
    # 测试自定义实现
    custom_time, _ = benchmark(custom_func, *inputs, **kwargs, iterations=50)
    custom_output = custom_func(*inputs, **kwargs)
    results["custom_time_ms"] = custom_time
    
    # 计算差异
    diff = torch.abs(custom_output - reference_output)
    results["max_diff"] = float(diff.max())
    results["correct"] = results["max_diff"] < 1e-4
    
    # 计算加速比
    if ref_time > 0:
        results["speedup"] = ref_time / custom_time if custom_time > 0 else 0
    
    return results


def format_tensor_info(tensor) -> str:
    """格式化张量信息"""
    if hasattr(tensor, 'shape'):
        shape = tuple(tensor.shape)
    else:
        shape = "N/A"
    
    if hasattr(tensor, 'dtype'):
        dtype = str(tensor.dtype)
    else:
        dtype = "N/A"
    
    return f"shape={shape}, dtype={dtype}"


def validate_shape(shape1, shape2) -> bool:
    """验证两个形状是否兼容"""
    if len(shape1) != len(shape2):
        return False
    
    for s1, s2 in zip(shape1, shape2):
        if s1 != s2 and s1 != 1 and s2 != 1:
            return False
    
    return True


def broadcast_shapes(shape1, shape2) -> Tuple[int, ...]:
    """计算广播后的形状"""
    len1, len2 = len(shape1), len(shape2)
    max_len = max(len1, len2)
    
    # 左对齐
    s1 = [1] * (max_len - len1) + list(shape1)
    s2 = [1] * (max_len - len2) + list(shape2)
    
    result = []
    for a, b in zip(s1, s2):
        if a == b:
            result.append(a)
        elif a == 1:
            result.append(b)
        elif b == 1:
            result.append(a)
        else:
            raise ValueError(f"无法广播形状: {shape1} 和 {shape2}")
    
    return tuple(result)


def profile_memory(func):
    """内存 profiling 装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import torch
        
        if torch.npu.is_available():
            torch.npu.synchronize()
        
        # 这里需要使用 memory profiler
        result = func(*args, **kwargs)
        
        if torch.npu.is_available():
            torch.npu.synchronize()
        
        return result
    
    return wrapper


# ==================== 常用数学函数 ====================

def gelu_approx(x: np.ndarray) -> np.ndarray:
    """GeLU 的快速近似实现"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))


def swish(x: np.ndarray) -> np.ndarray:
    """Swish 激活函数"""
    return x / (1 + np.exp(-x))


def mish(x: np.ndarray) -> np.ndarray:
    """Mish 激活函数"""
    return x * np.tanh(np.log(1 + np.exp(x)))


def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """LeakyReLU 激活函数"""
    return np.where(x > 0, x, alpha * x)


def apply_activation(activation: str, x: np.ndarray, **kwargs) -> np.ndarray:
    """应用激活函数"""
    activations = {
        "relu": lambda: np.maximum(0, x),
        "gelu": lambda: gelu_approx(x),
        "swish": lambda: swish(x),
        "mish": lambda: mish(x),
        "leaky_relu": lambda: leaky_relu(x, kwargs.get("alpha", 0.01)),
        "sigmoid": lambda: 1 / (1 + np.exp(-x)),
        "tanh": lambda: np.tanh(x),
    }
    
    if activation.lower() not in activations:
        raise ValueError(f"未知激活函数: {activation}")
    
    return activations[activation.lower()]()


if __name__ == "__main__":
    # 简单测试
    print("工具函数测试")
    
    x = np.random.randn(100, 100)
    
    # 测试激活函数
    print("\n激活函数测试:")
    for act in ["relu", "gelu", "swish", "mish"]:
        y = apply_activation(act, x)
        print(f"  {act}: shape={y.shape}, range=[{y.min():.3f}, {y.max():.3f}]")
    
    # 测试广播形状
    print("\n广播形状测试:")
    print(f"  broadcast({(2, 3)}, {(3,)}) = {broadcast_shapes((2, 3), (3,))}")
    print(f"  broadcast({(1, 4)}, {(3, 4)}) = {broadcast_shapes((1, 4), (3, 4))}")
