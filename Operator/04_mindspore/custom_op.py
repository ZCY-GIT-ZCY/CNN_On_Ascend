"""
MindSpore 自定义算子示例
"""
import mindspore as ms
import mindspore.nn as nn
import numpy as np


class CustomSwishCell(nn.Cell):
    """使用 Cell 方式定义 Swish 激活函数"""
    def __init__(self):
        super(CustomSwishCell, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        return x * self.sigmoid(x)


class CustomGELUCell(nn.Cell):
    """GeLU 激活函数的 Cell 实现"""
    def __init__(self):
        super(CustomGELUCell, self).__init__()

    def construct(self, x):
        sqrt_2_over_pi = 0.7978845608028654
        coef = 0.044715
        x_cubed = x ** 3
        inner = sqrt_2_over_pi * (x + coef * x_cubed)
        return 0.5 * x * (ms.ops.tanh(inner) + 1)


class CustomDenseCell(nn.Cell):
    """自定义全连接层"""
    def __init__(self, in_channels, out_channels, has_bias=True):
        super(CustomDenseCell, self).__init__()
        self.weight = ms.Parameter(ms.Tensor(
            np.random.randn(out_channels, in_channels).astype(np.float32) * 0.01
        ))
        self.bias = None
        if has_bias:
            self.bias = ms.Parameter(ms.Tensor(
                np.zeros(out_channels, dtype=np.float32)
            ))
        self.matmul = ms.ops.MatMul(transpose_b=True)
        self.add = ms.ops.Add()

    def construct(self, x):
        output = self.matmul(x, self.weight)
        if self.bias is not None:
            output = self.add(output, self.bias)
        return output


if __name__ == "__main__":
    print("=" * 60)
    print("MindSpore 自定义算子测试")
    print("=" * 60)

    ms.context.set_context(device_target="Ascend", device_id=0)

    print("\n--- 测试 CustomSwishCell ---")
    swish = CustomSwishCell()
    x = ms.Tensor(np.random.randn(2, 32, 64, 64).astype(np.float32))
    output = swish(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{float(output.min()):.4f}, {float(output.max()):.4f}]")

    print("\n--- 测试 CustomGELUCell ---")
    gelu = CustomGELUCell()
    output_gelu = gelu(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output_gelu.shape}")

    print("\n--- 测试 CustomDenseCell ---")
    dense = CustomDenseCell(in_channels=128, out_channels=256)
    x_dense = ms.Tensor(np.random.randn(16, 128).astype(np.float32))
    output_dense = dense(x_dense)
    print(f"输入形状: {x_dense.shape}")
    print(f"输出形状: {output_dense.shape}")

    print("\n✅ 所有 MindSpore 算子测试通过!")
