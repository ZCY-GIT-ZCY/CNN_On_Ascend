"""
PyTorch NPU 自定义算子示例

这是最简单、最推荐的算子开发方式，适合快速验证和简单算子实现。
使用 torch.autograd.Function 实现自定义前向和反向传播。
"""
import torch


class SwishOp(torch.autograd.Function):
    """
    Swish 激活函数的自定义实现
    swish(x) = x * sigmoid(x)
    """
    @staticmethod
    def forward(ctx, x):
        sigmoid_x = torch.sigmoid(x)
        result = x * sigmoid_x
        ctx.save_for_backward(x, sigmoid_x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, sigmoid_x = ctx.saved_tensors
        sigmoid_derivative = sigmoid_x * (1 - sigmoid_x)
        grad_input = grad_output * (sigmoid_x + x * sigmoid_derivative)
        return grad_input


class LeakyReLUOp(torch.autograd.Function):
    """
    LeakyReLU 的自定义实现
    leaky_relu(x) = x if x > 0 else negative_slope * x
    """
    @staticmethod
    def forward(ctx, x, negative_slope=0.01):
        ctx.negative_slope = negative_slope
        ctx.save_for_backward(x)
        return torch.where(x > 0, x, x * negative_slope)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        slope = ctx.negative_slope
        grad_input = grad_output * torch.where(x > 0,
                                                torch.ones_like(x),
                                                torch.full_like(x, slope))
        return grad_input, None


class GeluOp(torch.autograd.Function):
    """
    GeLU 激活函数的自定义实现
    gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    @staticmethod
    def forward(ctx, x):
        sqrt_2_over_pi = 0.7978845608028654
        coef = 0.044715
        x_cubed = x ** 3
        inner = sqrt_2_over_pi * (x + coef * x_cubed)
        tanh_inner = torch.tanh(inner)
        result = 0.5 * x * (1 + tanh_inner)
        ctx.save_for_backward(x, tanh_inner, x_cubed)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, tanh_inner, x_cubed = ctx.saved_tensors
        sqrt_2_over_pi = 0.7978845608028654
        coef = 0.044715
        derivative = 0.5 * (1 + tanh_inner + x * (1 - tanh_inner ** 2) *
                           sqrt_2_over_pi * (1 + 3 * coef * x_cubed / (x + 1e-8)))
        return grad_output * derivative


class CustomMatMulOp(torch.autograd.Function):
    """矩阵乘法算子，支持批量处理"""
    @staticmethod
    def forward(ctx, x, weight):
        result = torch.matmul(x, weight)
        ctx.save_for_backward(x, weight)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad_x = torch.matmul(grad_output, weight.T)
        grad_weight = torch.matmul(x.transpose(-2, -1), grad_output).transpose(-2, -1)
        return grad_x, grad_weight


def swish(x):
    """Swish 激活函数"""
    return SwishOp.apply(x)


def leaky_relu(x, negative_slope=0.01):
    """LeakyReLU 激活函数"""
    return LeakyReLUOp.apply(x, negative_slope)


def gelu(x):
    """GeLU 激活函数"""
    return GeluOp.apply(x)


def custom_matmul(x, weight):
    """自定义矩阵乘法"""
    return CustomMatMulOp.apply(x, weight)


if __name__ == "__main__":
    import torch_npu

    torch_npu.npu.init()

    print("=" * 60)
    print("PyTorch NPU 自定义算子测试")
    print("=" * 60)

    if torch.npu.is_available():
        device = torch.device("npu:0")
        print(f"NPU 可用: {torch.npu.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("NPU 不可用，使用 CPU")

    x = torch.randn(16, 128, device=device)

    print("\n--- 测试 Swish ---")
    result_swish = swish(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {result_swish.shape}")
    print(f"输出范围: [{result_swish.min():.4f}, {result_swish.max():.4f}]")

    print("\n--- 测试 LeakyReLU ---")
    result_lrelu = leaky_relu(x, negative_slope=0.1)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {result_lrelu.shape}")

    print("\n--- 测试 GeLU ---")
    result_gelu = gelu(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {result_gelu.shape}")

    print("\n--- 测试梯度计算 ---")
    x_test = torch.randn(4, 32, requires_grad=True, device=device)
    y = swish(x_test).sum()
    y.backward()
    print(f"Swish 梯度计算成功，梯度形状: {x_test.grad.shape}")

    print("\n--- 测试自定义矩阵乘法 ---")
    x_mat = torch.randn(8, 64, 128, device=device)
    w_mat = torch.randn(128, 256, device=device)
    result_mm = custom_matmul(x_mat, w_mat)
    print(f"输入形状: {x_mat.shape}")
    print(f"权重形状: {w_mat.shape}")
    print(f"输出形状: {result_mm.shape}")

    print("\n--- 验证正确性 ---")
    expected = torch.nn.functional.silu(x)
    diff = (swish(x) - expected).abs().max().item()
    print(f"Swish 与 PyTorch 实现的最大差异: {diff:.6f}")

    print("\n✅ 所有测试通过!")
