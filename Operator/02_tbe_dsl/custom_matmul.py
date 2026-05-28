"""
TBE DSL 自定义算子示例 - 矩阵乘法

展示如何使用 TBE DSL 实现更复杂的矩阵乘法算子。
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi.cce import util
import numpy as np


def compute_matmul(input_x, input_y, transpose_x=False, transpose_y=False):
    """
    矩阵乘法计算
    
    计算: output = X @ Y (可选转置)
    
    Args:
        input_x: 输入张量 X
        input_y: 输入张量 Y
        transpose_x: 是否转置 X
        transpose_y: 是否转置 Y
    
    Returns:
        结果张量
    """
    dtype = input_x.dtype
    
    # 确定最后两个维度用于矩阵乘法
    shape_x = input_x.shape
    shape_y = input_y.shape
    
    # 如果是 4D 张量，先 reshape 为 2D 进行矩阵乘法
    if len(shape_x) == 4:
        batch_x, channel_x, height_x, width_x = shape_x
        # Reshape 为 (batch, channel*height, width)
        input_x = te.lang.cce.reshape(input_x, (batch_x * channel_x, height_x * width_x))
        shape_x = input_x.shape
    
    if len(shape_y) == 4:
        batch_y, channel_y, height_y, width_y = shape_y
        input_y = te.lang.cce.reshape(input_y, (batch_y * channel_y, height_y * width_y))
        shape_y = input_y.shape
    
    # 根据转置选项调整形状
    if transpose_x:
        # 对最后两个维度转置
        m = shape_x[-1]
        k = shape_x[-2]
        input_x = te.lang.cce.transpose(input_x, (0, 2, 1))
    else:
        m = shape_x[-2] if len(shape_x) > 1 else 1
        k = shape_x[-1]
    
    if transpose_y:
        k2 = shape_y[-2]
        n = shape_y[-1]
        input_y = te.lang.cce.transpose(input_y, (0, 2, 1))
    else:
        k2 = shape_y[-2] if len(shape_y) > 1 else 1
        n = shape_y[-1]
    
    # 检查维度匹配
    if k != k2:
        raise ValueError(f"矩阵维度不匹配: k={k}, k2={k2}")
    
    # 执行矩阵乘法
    result = te.lang.cce.matmul(input_x, input_y, False, transpose_y)
    
    return result


@fusion_manager.register("custom_matmul")
def custom_matmul_compute(inputs, weights, output, transpose_x=False, 
                          transpose_y=False, kernel_name="custom_matmul"):
    """
    矩阵乘法融合算子计算函数
    
    Args:
        inputs: 输入张量字典
        weights: 权重张量字典
        output: 输出张量字典
        transpose_x: 是否转置输入
        transpose_y: 是否转置权重
        kernel_name: 内核名称
    """
    shape_x = inputs.get("shape")
    shape_w = weights.get("shape")
    dtype = inputs.get("dtype")
    
    # 创建 TVM 占位符
    tensor_x = tvm.placeholder(shape_x, dtype=dtype, name="input_x")
    tensor_w = tvm.placeholder(shape_w, dtype=dtype, name="input_w")
    
    # 计算
    result = compute_matmul(tensor_x, tensor_w, transpose_x, transpose_y)
    
    # 构建调度
    with tvm.target.cce():
        schedule = te.lang.cce.auto_schedule(result)
    
    # 编译
    config = {"print_ir": False, "need_build": True}
    te.lang.cce.build(schedule,
                      [tensor_x, tensor_w, result],
                      "cce",
                      kernel_name=kernel_name,
                      attrs=config)


def custom_matmul(inputs, weights, output, transpose_x=False,
                  transpose_y=False, kernel_name="custom_matmul"):
    """
    矩阵乘法算子入口函数
    
    Args:
        inputs: 输入张量描述字典
        weights: 权重张量描述字典
        output: 输出张量描述字典
        transpose_x: 是否转置输入
        transpose_y: 是否转置权重
        kernel_name: 内核名称
    """
    # 参数检查
    util.check_shape_rule(inputs.get("shape"))
    util.check_shape_rule(weights.get("shape"))
    util.check_dtype_rule(inputs.get("dtype"), ["float16", "float32"])
    util.check_dtype_rule(weights.get("dtype"), ["float16", "float32"])
    
    # 维度检查
    shape_x = inputs.get("shape")
    shape_w = weights.get("shape")
    
    # 最后两个维度需要满足矩阵乘法条件
    k_x = shape_x[-1]
    k_w = shape_w[-2] if transpose_w else shape_w[-2]
    
    if transpose_w:
        expected_k_w = shape_w[-1]
    else:
        expected_k_w = shape_w[-2]
    
    if k_x != expected_k_w:
        raise ValueError(f"输入和权重维度不匹配: k_x={k_x}, k_w={expected_k_w}")
    
    custom_matmul_compute(inputs, weights, output, transpose_x, transpose_y, kernel_name)


# ==================== 激活函数融合示例 ====================

def compute_add_relu(input_x, input_y):
    """
    Add + ReLU 融合算子
    
    这种融合可以减少内存访问，提高性能
    """
    # 先加法
    add_result = te.lang.cce.vadd(input_x, input_y)
    
    # 再 ReLU: max(0, x)
    relu_result = te.lang.cce.vmax(add_result, tvm.const(0, dtype=add_result.dtype))
    
    return relu_result


@fusion_manager.register("custom_add_relu")
def custom_add_relu_compute(input_x, input_y, output, kernel_name="custom_add_relu"):
    """
    Add + ReLU 融合算子计算函数
    """
    shape_x = input_x.get("shape")
    shape_y = input_y.get("shape")
    dtype = input_x.get("dtype")
    
    tensor_x = tvm.placeholder(shape_x, dtype=dtype, name="input_x")
    tensor_y = tvm.placeholder(shape_y, dtype=dtype, name="input_y")
    
    result = compute_add_relu(tensor_x, tensor_y)
    
    with tvm.target.cce():
        schedule = te.lang.cce.auto_schedule(result)
    
    config = {"print_ir": False, "need_build": True}
    te.lang.cce.build(schedule,
                      [tensor_x, tensor_y, result],
                      "cce",
                      kernel_name=kernel_name,
                      attrs=config)


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 50)
    print("TBE DSL 矩阵乘法算子示例")
    print("=" * 50)
    
    # 测试矩阵乘法
    M, K, N = 64, 128, 256
    
    inputs = {
        "shape": (M, K),
        "dtype": "float16",
        "format": "ND"
    }
    
    weights = {
        "shape": (K, N),
        "dtype": "float16",
        "format": "ND"
    }
    
    output = {
        "shape": (M, N),
        "dtype": "float16",
        "format": "ND"
    }
    
    print(f"输入 X: ({M}, {K})")
    print(f"权重 W: ({K}, {N})")
    print(f"输出 Y: ({M}, {N})")
    
    # 测试 Add + ReLU 融合
    shape = (1, 64, 64)
    input_x = {"shape": shape, "dtype": "float16", "format": "NCHW"}
    input_y = {"shape": shape, "dtype": "float16", "format": "NCHW"}
    output_z = {"shape": shape, "dtype": "float16", "format": "NCHW"}
    
    print(f"\nAdd + ReLU 融合测试:")
    print(f"输入形状: {shape}")
    
    print("\n✅ 算子定义完成！")
