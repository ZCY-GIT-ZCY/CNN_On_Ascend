"""
TBE DSL 自定义算子示例 - 加法算子

TBE (Tensor Boost Engine) 是昇腾的算子开发框架。
本文件展示如何使用 TBE DSL 编写一个简单的加法算子。
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi.cce import util


# ==================== 算子定义 ====================

def compute_add(input_x, input_y, output_z, alpha=1.0):
    """
    加法算子的计算实现
    
    计算: z = x + alpha * y
    
    Args:
        input_x: 第一个输入张量 (TVM 格式)
        input_y: 第二个输入张量 (TVM 格式)
        output_z: 输出张量描述
        alpha: 缩放因子
    
    Returns:
        计算结果张量
    """
    # 将输入转换为 CCE 格式（支持向量化的张量格式）
    dtype = input_x.dtype
    shape_x = input_x.shape
    shape_y = input_y.shape
    
    # 使用 te.lang.cce.vadd 进行向量化的加法
    # 标量乘法
    input_y_scaled = te.lang.cce.vmuls(input_y, tvm.const(alpha, dtype=dtype))
    
    # 向量加法
    res = te.lang.cce.vadd(input_x, input_y_scaled)
    
    return res


@fusion_manager.register("custom_add")
def custom_add_compute(input_x, input_y, output_z, alpha=1.0, kernel_name="custom_add"):
    """
    融合算子的计算函数
    
    Args:
        input_x: 输入张量字典，包含 shape, dtype, format 等
        input_y: 输入张量字典
        output_z: 输出张量字典
        alpha: 缩放因子
        kernel_name: 内核名称
    
    Returns:
        无，直接在 output_z 中填充结果
    """
    # 获取输入张量的 shape 和 dtype
    shape_x = input_x.get("shape")
    shape_y = input_y.get("shape")
    dtype_x = input_x.get("dtype")
    dtype_y = input_y.get("dtype")
    
    # 将输入转换为 TVM 张量
    tensor_x = tvm.placeholder(shape_x, dtype=dtype_x, name="input_x")
    tensor_y = tvm.placeholder(shape_y, dtype=dtype_y, name="input_y")
    
    # 调用计算函数
    res_tensor = compute_add(tensor_x, tensor_y, output_z, alpha)
    
    # 构建调度
    with tvm.target.cce():
        schedule = te.lang.cce.auto_schedule(res_tensor)
    
    # 编译
    config = {"print_ir": False, "need_build": True}
    te.lang.cce.build(schedule, 
                      [tensor_x, tensor_y, res_tensor], 
                      "cce", 
                      kernel_name=kernel_name,
                      attrs=config)


def custom_add(input_x, input_y, output_z, alpha=1.0, kernel_name="custom_add"):
    """
    加法算子的入口函数
    
    Args:
        input_x: dict, 包含 shape, dtype, format
        input_y: dict, 包含 shape, dtype, format
        output_z: dict, 包含 shape, dtype, format
        alpha: float, 缩放因子，默认 1.0
        kernel_name: str, 内核名称
    
    Returns:
        无
    """
    util.check_shape_rule(input_x.get("shape"))
    util.check_shape_rule(input_y.get("shape"))
    util.check_dtype_rule(input_x.get("dtype"), ["float16", "float32"])
    util.check_dtype_rule(input_y.get("dtype"), ["float16", "float32"])
    
    custom_add_compute(input_x, input_y, output_z, alpha, kernel_name)


# ==================== 算子适配器 (用于 PyTorch 等框架调用) ====================

def npu_custom_add(x, y, alpha=1.0):
    """
    在 PyTorch NPU 上调用 TBE 算子的适配器
    
    Args:
        x: torch.Tensor (NPU 张量)
        y: torch.Tensor (NPU 张量)
        alpha: float
    
    Returns:
        torch.Tensor
    """
    import torch
    from torch.npu import npu_stream
    
    # 准备输入格式
    input_x = {
        "shape": list(x.shape),
        "dtype": "float16" if x.dtype == torch.float16 else "float32",
        "format": "NC1HWC0" if len(x.shape) == 5 else "NCHW"
    }
    input_y = {
        "shape": list(y.shape),
        "dtype": "float16" if y.dtype == torch.float16 else "float32",
        "format": "NC1HWC0" if len(y.shape) == 5 else "NCHW"
    }
    
    # 输出格式
    output_shape = [max(sx, sy) for sx, sy in zip(x.shape, y.shape)]
    output_z = {
        "shape": output_shape,
        "dtype": input_x["dtype"],
        "format": input_x["format"]
    }
    
    # 调用 TBE 算子
    kernel_name = "custom_add"
    custom_add(input_x, input_y, output_z, alpha, kernel_name)
    
    # 创建输出张量并执行
    output = torch.empty(output_shape, dtype=x.dtype, device="npu:0")
    
    # 使用 te.lang.cce.build 后的 kernel 需要通过 Runtime 执行
    # 这里需要实际的编译和运行代码
    
    return output


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 50)
    print("TBE DSL 加法算子示例")
    print("=" * 50)
    
    # 定义输入
    shape = (1, 32, 224, 224)
    dtype = "float16"
    
    input_x = {
        "shape": shape,
        "dtype": dtype,
        "format": "NCHW"
    }
    
    input_y = {
        "shape": shape,
        "dtype": dtype,
        "format": "NCHW"
    }
    
    output_z = {
        "shape": shape,
        "dtype": dtype,
        "format": "NCHW"
    }
    
    print(f"输入 X shape: {input_x['shape']}, dtype: {input_x['dtype']}")
    print(f"输入 Y shape: {input_y['shape']}, dtype: {input_y['dtype']}")
    print(f"输出 Z shape: {output_z['shape']}, dtype: {output_z['dtype']}")
    
    # 注意：在没有实际 NPU 环境下，这里只展示代码结构
    # 实际运行需要在昇腾开发板上执行
    
    print("\n算子定义完成！")
    print("使用说明:")
    print("1. 将此文件放入算子工程的 op_impl 目录")
    print("2. 配置对应的 json 文件定义算子接口")
    print("3. 使用 ATC 编译器编译和注册算子")
