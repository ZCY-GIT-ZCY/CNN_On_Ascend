# 昇腾算子开发完全指南

## 目录

1. [算子开发概述](#1-算子开发概述)
2. [昇腾硬件架构](#2-昇腾硬件架构)
3. [CANN 软件栈](#3-cann-软件栈)
4. [开发方式详解](#4-开发方式详解)
5. [PyTorch NPU 算子开发](#5-pytorch-npu-算子开发)
6. [MindSpore 算子开发](#6-mindspore-算子开发)
7. [TBE DSL 算子开发](#7-tbe-dsl-算子开发)
8. [ACLNN 算子开发](#8-aclnn-算子开发)
9. [性能优化策略](#9-性能优化策略)
10. [调试与验证](#10-调试与验证)

---

## 1. 算子开发概述

### 1.1 什么是算子？

算子（Operator）是深度学习中的基本计算单元，它定义了输入张量（Tensor）到输出张量的数学变换。

```
输入张量 → [算子] → 输出张量
   x      →  MatMul  →   y = Wx + b
```

### 1.2 为什么需要自定义算子？

| 场景 | 说明 |
|------|------|
| 新算法验证 | 框架不支持的新操作需要自己实现 |
| 性能优化 | 已有算子性能不满足需求 |
| 硬件特性 | 利用特定硬件的独特能力 |
| 融合操作 | 将多个操作合并减少开销 |

### 1.3 算子开发层次

```
┌─────────────────────────────────────────┐
│           Python 高层接口                │  ← 最简单
│      (PyTorch NPU / MindSpore)          │
├─────────────────────────────────────────┤
│           C++ 底层接口                   │
│        (ACLNN / TBE DSL)                │
├─────────────────────────────────────────┤
│           汇编/Intrinsic                 │  ← 最复杂
│        (手工优化汇编指令)                 │
└─────────────────────────────────────────┘
```

---

## 2. 昇腾硬件架构

### 2.1 Ascend 芯片架构

```
                    ┌─────────────────┐
                    │   Host CPU      │
                    │  (控制平面)      │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  CANN 软件栈     │
                    │                 │
                    ├─────────────────┤
                    │  AI Core (Vector)│
                    │  ┌─────────────┐│
                    │  │ Vector Unit ││  ← 向量计算
                    │  │  Scalar Unit││
                    │  │  MMA (矩阵)  ││
                    │  └─────────────┘│
                    └─────────────────┘
```

### 2.2 AI Core 计算单元

| 单元 | 功能 | 特点 |
|------|------|------|
| **Scalar Unit** | 控制流、地址计算 | 标量运算 |
| **Vector Unit** | 向量运算 | FP16/FP32 向量操作 |
| **Cube Unit (MMA)** | 矩阵乘法 | 16x16x16 矩阵运算 |
| **Scalar Memory** | 片上存储 | 高速数据缓存 |

### 2.3 内存层次

```
┌────────────────────────────────────┐
│        DDR / HBM (Device)          │  ← 带宽较低，容量大
├────────────────────────────────────┤
│        L2 Cache                     │  ← 带宽中等
├────────────────────────────────────┤
│        L1 Cache (UB)                │  ← 带宽高，容量小
├────────────────────────────────────┤
│        寄存器文件                   │  ← 带宽最高
└────────────────────────────────────┘
```

**优化原则**：尽量减少 DDR 访问，增加数据复用。

---

## 3. CANN 软件栈

### 3.1 CANN 架构

```
┌────────────────────────────────────────┐
│           深度学习框架                  │
│    (PyTorch / TensorFlow / MindSpore)  │
├────────────────────────────────────────┤
│           ACL (Ascend CL)              │  ← 应用接口层
├────────────────────────────────────────┤
│           TBE (Tensor Boost Engine)    │  ← 算子开发框架
├────────────────────────────────────────┤
│           HCCL                         │  ← 集合通信库
├────────────────────────────────────────┤
│           Runtime                      │  ← 运行时
├────────────────────────────────────────┤
│           Driver                       │  ← 驱动层
└────────────────────────────────────────┘
```

### 3.2 关键组件

| 组件 | 路径 | 说明 |
|------|------|------|
| **ATC** | `atc/bin/atc` | 模型转换器 |
| **OPP** | `opp/` | 算子库 |
| **TBE** | `opp/built-in/op_impl/ai_core/tbe/` | 算子开发引擎 |
| **Runtime** | `runtime/` | 运行时库 |
| **ACL** | `include/acl/` | 应用接口 |

### 3.3 算子编译流程

```
1. 编写算子代码 (Python/C++)
       ↓
2. 解析计算图
       ↓
3. 选择算子实现 (TBE/ACLNN)
       ↓
4. 编译为适配文件 (.o)
       ↓
5. 注册到 OPP
       ↓
6. 加载执行
```

---

## 4. 开发方式详解

### 4.1 四种开发方式对比

| 方式 | 难度 | 性能 | 适用场景 | 开发效率 |
|------|------|------|----------|----------|
| **PyTorch NPU** | ⭐ | 一般 | 快速验证 | ⭐⭐⭐⭐⭐ |
| **MindSpore** | ⭐⭐ | 良好 | 网络构建 | ⭐⭐⭐⭐ |
| **ACLNN** | ⭐⭐⭐ | 优秀 | 生产部署 | ⭐⭐⭐ |
| **TBE DSL** | ⭐⭐⭐⭐ | 最优 | 极致优化 | ⭐⭐ |

### 4.2 选择指南

```
开始一个新项目？
    │
    ├── 快速验证算法 → PyTorch NPU
    │
    ├── 使用 MindSpore 框架 → MindSpore 自定义算子
    │
    ├── 需要高性能 + CANN 8.0+ → ACLNN
    │
    └── 极致性能要求 → TBE DSL
```

---

## 5. PyTorch NPU 算子开发

### 5.1 原理讲解

PyTorch NPU 算子基于 `torch.autograd.Function` 实现，核心思想是：
- **前向传播**：使用 PyTorch 已有操作组合
- **反向传播**：利用 PyTorch 自动微分引擎

```
forward(x) ──→ output
                 ↑
backward(grad) ─┘
```

### 5.2 实现方式

#### 方式一：torch.autograd.Function

```python
import torch

class SwishOp(torch.autograd.Function):
    """
    Swish 激活函数: swish(x) = x * sigmoid(x)
    
    原理：
    1. forward: 执行前向计算，保存中间结果供反向使用
    2. backward: 根据链式法则计算梯度
    """
    
    @staticmethod
    def forward(ctx, x):
        # 计算 sigmoid
        sigmoid_x = torch.sigmoid(x)
        # 计算结果
        result = x * sigmoid_x
        # 保存中间变量供反向传播使用
        ctx.save_for_backward(x, sigmoid_x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        # 取出保存的中间变量
        x, sigmoid_x = ctx.saved_tensors
        # 计算 Swish 的梯度
        # d/dx[x*sigmoid(x)] = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))
        sigmoid_derivative = sigmoid_x * (1 - sigmoid_x)
        grad_input = grad_output * (sigmoid_x + x * sigmoid_derivative)
        return grad_input

# 便捷函数
def swish(x):
    return SwishOp.apply(x)
```

#### 方式二：nn.Module 封装

```python
import torch.nn as nn

class Swish(nn.Module):
    """封装为 nn.Module，方便在网络中使用"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return SwishOp.apply(x)
```

### 5.3 完整示例代码

```python
"""
文件位置: 01_pytorch_npu/custom_ops.py
PyTorch NPU 自定义算子完整示例
"""
import torch

class SwishOp(torch.autograd.Function):
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
    """LeakyReLU: x if x > 0 else alpha * x"""
    @staticmethod
    def forward(ctx, x, negative_slope=0.01):
        ctx.negative_slope = negative_slope
        ctx.save_for_backward(x)
        return torch.where(x > 0, x, x * negative_slope)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        slope = ctx.negative_slope
        grad_input = grad_output * torch.where(
            x > 0, 
            torch.ones_like(x), 
            torch.full_like(x, slope)
        )
        return grad_input, None


class GeluOp(torch.autograd.Function):
    """GeLU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))"""
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
    """自定义矩阵乘法"""
    @staticmethod
    def forward(ctx, x, weight):
        result = torch.matmul(x, weight)
        ctx.save_for_backward(x, weight)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        # y = x @ w, dy/dx = grad @ w.T, dy/dw = x.T @ grad
        grad_x = torch.matmul(grad_output, weight.T)
        grad_weight = torch.matmul(x.transpose(-2, -1), grad_output).transpose(-2, -1)
        return grad_x, grad_weight


# 便捷函数
def swish(x):
    return SwishOp.apply(x)

def leaky_relu(x, negative_slope=0.01):
    return LeakyReLUOp.apply(x, negative_slope)

def gelu(x):
    return GeluOp.apply(x)

def custom_matmul(x, weight):
    return CustomMatMulOp.apply(x, weight)


# 使用示例
if __name__ == "__main__":
    import torch_npu
    torch_npu.npu.init()
    
    device = torch.device("npu:0")
    x = torch.randn(16, 128, device=device)
    
    # 测试
    print(swish(x).shape)           # torch.Size([16, 128])
    print(gelu(x).shape)            # torch.Size([16, 128])
    print(custom_matmul(x, torch.randn(128, 256, device=device)).shape)  # torch.Size([16, 256])
```

### 5.4 使用流程

```bash
# 1. 初始化 NPU
import torch_npu
torch_npu.npu.init()

# 2. 创建 NPU 张量
x = torch.randn(10, 10).npu()

# 3. 调用自定义算子
y = swish(x)
```

---

## 6. MindSpore 算子开发

### 6.1 原理讲解

MindSpore 使用 Cell 作为网络构建的基本单元。Cell 是一种可组合的网络层抽象。

```
nn.Cell
    ├── __init__: 初始化子模块和参数
    └── construct: 定义计算逻辑
```

### 6.2 实现方式

#### 方式一：nn.Cell

```python
import mindspore as ms
import mindspore.nn as nn

class CustomSwish(nn.Cell):
    """使用 Cell 方式定义 Swish"""
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
    
    def construct(self, x):
        return x * self.sigmoid(x)
```

#### 方式二：组合已有算子

```python
class CustomConvBN(nn.Cell):
    """卷积 + BatchNorm + 激活 融合"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
```

### 6.3 完整示例代码

```python
"""
文件位置: 04_mindspore/custom_op.py
MindSpore 自定义算子完整示例
"""
import mindspore as ms
import mindspore.nn as nn
import numpy as np


class CustomSwishCell(nn.Cell):
    """Swish 激活函数实现"""
    def __init__(self):
        super(CustomSwishCell, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        return x * self.sigmoid(x)


class CustomGELUCell(nn.Cell):
    """GeLU 激活函数实现"""
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


# 使用示例
if __name__ == "__main__":
    ms.context.set_context(device_target="Ascend")
    
    # 测试
    swish = CustomSwishCell()
    x = ms.Tensor(np.random.randn(2, 32, 64, 64).astype(np.float32))
    output = swish(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
```

---

## 7. TBE DSL 算子开发

### 7.1 原理详解：TVM 是什么？

TBE 的核心是 **TVM (Tensor Virtual Machine)**，它是一个深度学习编译器框架。TVM 将**声明式的计算描述**转换为**高效的硬件执行代码**。

#### TVM 的编译流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TVM 编译器架构                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Tensor Expression (TE)         ← 你写的计算描述                   │
│     a = tvm.placeholder(...)                                      │
│     b = tvm.placeholder(...)                                      │
│     c = tvm.compute(..., lambda: a + b)                           │
│                                                                     │
│                         ↓                                           │
│                                                                     │
│  2. Operation (TensorIR)           ← 内部的计算图表示                 │
│     描述了"要计算什么"                                              │
│                                                                     │
│                         ↓                                           │
│                                                                     │
│  3. Schedule                         ← 你写的调度优化                 │
│     s = tvm.create_schedule(c.op)                                   │
│     yo, yi = s[c].split(axis, factor=16)                          │
│                                                                     │
│                         ↓                                           │
│                                                                     │
│  4. TensorIR (优化后)              ← 描述了"如何计算"                │
│     分块、向量化的计算图                                             │
│                                                                     │
│                         ↓                                           │
│                                                                     │
│  5. Code Generation                  ← 生成的代码                    │
│     C/汇编/CUDA/CCE 代码                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 核心概念详解

#### 概念一：Placeholder（占位符）

Placeholder 是**声明**输入张量的形状和数据类型，但不分配实际内存：

```python
from te import tvm

# 声明一个形状为 (N, K)，数据类型为 float16 的输入张量
# 这只是"声明"，不是"计算"
# 注意：Placeholder 的 shape 使用变量名（如 n, k）表示动态维度
A = tvm.placeholder(
    shape=(n, k),  # 形状：(N, K)
    dtype="float16",           # 数据类型
    name="input_A"        # 名称（用于调试）
)

# 声明另一个张量，形状为 (K, M)
B = tvm.placeholder(
    shape=(k, m),
    dtype="float16",
    name="input_B"
)

# 注意：此时还没有任何计算发生！
print(f"A shape: {A.shape}")  # (n, k)
print(f"B shape: {B.shape}")  # (k, m)
```

**类比理解**：
```
Placeholder 就像 C 语言的函数参数声明：
void compute(int x, int y);  // 只是声明，不执行
```

#### 概念二：Compute（计算定义）

`compute` 是**定义计算逻辑**的地方，描述"输出张量的每个元素如何计算"：

```python
from te import tvm

# 定义输入
A = tvm.placeholder((n, k), dtype="float16", name="A")
B = tvm.placeholder((k, m), dtype="float16", name="B")

# 定义矩阵乘法的 reduce axis
k_axis = tvm.reduce_axis((0, k), name="k")

# 定义计算
# C[i, j] = Σ(k) A[i, k] * B[k, j]
C = tvm.compute(
    shape=(n, m),  # 输出形状
    
    # lambda 函数定义每个输出元素的计算方式
    # f(i, j) 会被调用 n*m 次，每次返回该位置的计算值
    fcompute=lambda i, j: tvm.sum(
        A[i, k_axis] * B[k_axis, j],
        axis=k_axis  # 对 k 轴求和
    ),
    
    name="C",
    tag="matmul"  # 标签，用于融合
)

# 此时生成的计算图：
# C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j] + ... + A[i,k-1]*B[k-1,j]
```

**工作原理**：

```
┌─────────────────────────────────────────────────────────────┐
│  tvm.compute 创建了什么？                                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入:                                                      │
│    A: shape (N, K)                                         │
│    B: shape (K, M)                                         │
│                                                             │
│  输出 C: shape (N, M)                                      │
│                                                             │
│  对每个 (i, j) 位置，lambda 函数返回:                       │
│                                                             │
│    C[i,j] = Σ(k=0 to K-1) A[i,k] * B[k,j]                │
│                                                             │
│  这个 lambda 会被调用 N*M 次                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 概念三：Schedule（调度）

**调度是 TBE/TVM 最核心也最难理解的概念！**

##### 什么是调度？

调度是对**计算执行方式的声明**，告诉编译器"如何"计算：

```
┌─────────────────────────────────────────────────────────────┐
│  调度 = 告诉编译器"如何"计算                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  同一份计算，不同调度 → 不同性能                              │
│                                                             │
│  例：矩阵乘法 C = A @ B                                     │
│                                                             │
│  调度1（朴素）：按行逐个计算                                 │
│    for i in range(N):                                      │
│      for j in range(M):                                    │
│        C[i,j] = sum(A[i,:] * B[:,j])                      │
│                                                             │
│  调度2（分块）：每次计算 16x16 小块                          │
│    for ii in range(0, N, 16):                             │
│      for jj in range(0, M, 16):                           │
│        for i in range(ii, min(ii+16,N)):                   │
│          for j in range(jj, min(jj+16,M)):                 │
│            C[i,j] = sum(...)                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

##### 常用调度操作

```python
from te import tvm

# 创建调度
s = tvm.create_schedule(C.op)

# ====================
# 操作1: split（分块）
# ====================
# 将一个轴分成多个小轴
# 例如：把 axis=0 (0~1023) 分成 (0~15), (16~31), ...
axis = s[C].op.axis[0]  # 取第一个轴 (N 维度)
co, ci = s[C].split(axis, factor=16)  # 分成外轴 co 和内轴 ci

# 结果：
# 原：for i in range(0, 1024)
# 后：for co in range(0, 64):     # 外循环
#        for ci in range(0, 16):   # 内循环

# ====================
# 操作2: reorder（重排）
# ====================
# 调整循环顺序
s[C].reorder(co, ci, j)  # 变成 co, ci, j 的顺序

# ====================
# 操作3: bind（绑定）
# ====================
# 将循环绑定到线程（用于并行）
thread_x = tvm.thread_axis("threadIdx.x")
s[C].bind(co, thread_x)  # co 循环绑定到线程

# ====================
# 操作4: unroll（展开）
# ====================
# 将小循环展开，减少分支开销
s[C].unroll(ci)  # 展开 ci 循环

# ====================
# 操作5: vectorize（向量化）
# ====================
# 使用 SIMD 指令，一次处理多个数据
s[C].vectorize(inner_axis)  # 向量化内轴
```

##### 调度可视化示例

```
原始计算：
  for i in 0..1023:
    for j in 0..1023:
      C[i,j] = ...

应用分块调度后 (factor=16)：
  for ii in 0..63:         # 外层大循环
    for jj in 0..63:       # 外层大循环
      for i in 0..15:      # 内层小块
        for j in 0..15:    # 内层小块
          C[ii*16+i, jj*16+j] = ...

内存访问模式对比：

朴素版本：
  A[0,0], A[0,1], A[0,2]...    ← 可能跳转到不连续地址
  
分块版本：
  A[0:16, 0:16]  ← 连续访问 16x16 块
  放入 L1 Cache   ← 后续访问命中缓存！
```

### 7.3 te.lang.cce API 详解

`te.lang.cce` 是昇腾提供的计算 API，是对 TVM compute 的封装：

```python
import te.lang.cce as te
from te import tvm

# ====================
# te.lang.cce.vadd
# ====================
# 向量加法：c = a + b
c = te.lang.cce.vadd(a, b)

# 等价于：
# c = tvm.compute(
#     shape=a.shape,
#     fcompute=lambda i: a[i] + b[i],
#     name="c"
# )

# ====================
# te.lang.cce.vmuls
# ====================
# 标量乘法：b = a * scalar
b = te.lang.cce.vmuls(a, tvm.const(2.0, dtype="float16"))

# ====================
# te.lang.cce.matmul
# ====================
# 矩阵乘法：c = a @ b
c = te.lang.cce.matmul(a, b)

# ====================
# te.lang.cce.relu
# ====================
# ReLU：b = max(0, a)
b = te.lang.cce.relu(a)

# ====================
# te.lang.cce.conv2d
# ====================
# 2D 卷积
c = te.lang.cce.conv2d(
    input_tensor=a,
    weight_tensor=b,
    stride=(1, 1),
    padding=(1, 1),
    dilation=(1, 1)
)
```

### 7.4 融合算子机制

融合是 TBE 的核心优势，可以将多个操作合并成一个 kernel，减少内存访问。

#### 融合前 vs 融合后

```
┌─────────────────────────────────────────────────────────────────────┐
│  融合前：3 个独立 kernel，3 次 DDR 访问                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Conv:  DDR → 计算 → DDR    (写中间结果)                            │
│  BN:    DDR → 计算 → DDR    (写中间结果)                            │
│  ReLU:  DDR → 计算 → DDR    (写最终结果)                            │
│                                                                     │
│  总计：6 次 DDR 读写                                                 │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  融合后：1 个 kernel，1 次 DDR 访问                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Fused: DDR → 片上计算 → DDR   (直接写最终结果)                     │
│                                                                     │
│  总计：2 次 DDR 读写（输入+输出）                                    │
│                                                                     │
│  优势：                                                             │
│  1. 消除中间结果的 DDR 读写                                          │
│  2. 减少 kernel 启动开销                                            │
│  3. 中间数据驻留在高速片上存储                                       │
└─────────────────────────────────────────────────────────────────────┘
```

#### 方式一：使用融合 API（简单）

```python
@fusion_manager.register("custom_fused_conv_bn_relu")
def fused_conv_bn_relu_compute(inputs, output, kernel_name="custom_fused"):
    """
    使用 te.lang.cce 提供的融合 API
    te.lang.cce 已经实现了常用的融合模式
    """
    input_tensor = inputs[0]
    shape = input_tensor.get("shape")
    dtype = input_tensor.get("dtype")
    
    # 创建输入张量
    x = tvm.placeholder(shape, dtype=dtype, name="x")
    weight = tvm.placeholder((64, 3, 3, 3), dtype=dtype, name="weight")
    bias = tvm.placeholder((64,), dtype=dtype, name="bias")
    
    with tvm.target.cce():
        # 使用融合 API：Conv + BiasAdd + ReLU
        # 底层会生成一个融合 kernel
        output_tensor = te.lang.cce.conv2d_bias_add_relu(
            x, weight, bias,
            stride=(1, 1),
            padding=(1, 1)
        )
        schedule = te.lang.cce.auto_schedule(output_tensor)
    
    te.lang.cce.build(schedule, [x, weight, bias, output_tensor],
                      "cce", kernel_name=kernel_name)
```

#### 方式二：手动融合（完整示例）

```python
@fusion_manager.register("custom_add_relu")
def custom_add_relu_compute(input_x, input_y, output, kernel_name="custom_add_relu"):
    """
    手动融合示例：Add + ReLU
    
    融合前的计算图：
        x ──┬──> Add ──> ReLU ──> y
        y ──┘
    
    融合后的计算：
        y = max(0, x + y)
    
    只需要一次 DDR 读写！
    """
    shape = input_x.get("shape")
    dtype = input_x.get("dtype")
    
    # 1. 创建输入张量
    tensor_x = tvm.placeholder(shape, dtype=dtype, name="x")
    tensor_y = tvm.placeholder(shape, dtype=dtype, name="y")
    
    # 2. 定义融合计算（手动融合的关键！）
    # 注意：我们把 Add 和 ReLU 的计算合并到一个 compute 中
    with tvm.target.cce():
        # Add: z = x + y
        temp = te.lang.cce.vadd(tensor_x, tensor_y)
        
        # ReLU: output = max(0, z)
        # 注意：这里直接把 temp 作为输入，形成融合
        output_tensor = te.lang.cce.relu(temp)
        
        # 3. 自动调度
        # TBE 会识别出这是 Add→ReLU 模式，自动生成融合 kernel
        schedule = te.lang.cce.auto_schedule(output_tensor)
    
    # 4. 编译
    te.lang.cce.build(
        schedule,
        [tensor_x, tensor_y, output_tensor],
        "cce",
        kernel_name=kernel_name
    )
```

#### 方式三：多操作融合（Conv + BN + Scale + ReLU）

```python
@fusion_manager.register("custom_fused_convolution")
def custom_fused_convolution_compute(inputs, output, kernel_name="custom_fused_conv"):
    """
    复杂融合示例：Conv + BN + Scale + ReLU
    
    融合前：
        input → Conv → BatchNorm → Scale → ReLU → output
        (4 次 kernel 调用，3 次 DDR 读写中间结果)
    
    融合后：
        input → [Conv + BN + Scale + ReLU 融合] → output
        (1 次 kernel 调用，1 次 DDR 读写)
    """
    shape = input_x.get("shape")  # NCHW 格式
    dtype = input_x.get("dtype")
    
    # 输入张量
    input_x = tvm.placeholder(shape, dtype=dtype, name="input_x")
    
    # 卷积权重 (Cout, Cin, Kh, Kw)
    weight_shape = (out_channels, shape[1], 3, 3)
    weight = tvm.placeholder(weight_shape, dtype=dtype, name="weight")
    
    # BN 参数 (均值, 方差, gamma, beta)
    bn_mean = tvm.placeholder((out_channels,), dtype=dtype, name="bn_mean")
    bn_var = tvm.placeholder((out_channels,), dtype=dtype, name="bn_var")
    bn_gamma = tvm.placeholder((out_channels,), dtype=dtype, name="bn_gamma")
    bn_beta = tvm.placeholder((out_channels,), dtype=dtype, name="bn_beta")
    
    # ============ 融合计算定义 ============
    with tvm.target.cce():
        # Step 1: 卷积
        conv_out = te.lang.cce.conv2d(
            input_x, weight,
            stride=(1, 1),
            padding=(1, 1)
        )
        
        # Step 2: BatchNorm (在融合中直接展开)
        # bn_out = gamma * (conv_out - mean) / sqrt(var + eps) + beta
        bn_sub = te.lang.cce.vsub(conv_out, bn_mean)
        bn_div = te.lang.cce.vdiv(bn_sub, 
                                   te.lang.cce.vsqrt(bn_var + tvm.const(1e-5, dtype)))
        bn_out = te.lang.cce.vadd(
            te.lang.cce.vmuls(bn_div, bn_gamma),
            bn_beta
        )
        
        # Step 3: ReLU
        output_tensor = te.lang.cce.relu(bn_out)
        
        # 自动调度（TBE 会自动识别融合模式）
        schedule = te.lang.cce.auto_schedule(output_tensor)
    
    # 编译
    te.lang.cce.build(
        schedule,
        [input_x, weight, bn_mean, bn_var, bn_gamma, bn_beta, output_tensor],
        "cce",
        kernel_name=kernel_name
    )
```

#### 融合的原理

```
┌─────────────────────────────────────────────────────────────────────┐
│                        融合如何工作？                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. 计算图构建期                                                    │
│     ─────────────────                                              │
│     你定义了：                                                       │
│       temp = vadd(x, y)       # 中间结果                            │
│       output = relu(temp)     # 使用中间结果                         │
│                                                                     │
│  2. 调度分析期                                                      │
│     ────────────────                                                │
│     TBE 分析计算图，发现：                                           │
│       - temp 只被 relu 使用                                         │
│       - temp 不需要写回 DDR                                         │
│       - 可以在片上直接传递给 relu                                    │
│                                                                     │
│  3. 代码生成期                                                      │
│     ──────────────                                                  │
│     TBE 生成融合 kernel：                                            │
│       for i:                                                       │
│         temp[i] = x[i] + y[i]   # Add 计算                        │
│         output[i] = max(0, temp[i])  # ReLU 计算                   │
│       # 两次计算在一个循环中完成，不写 DDR                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.5 完整示例：加法算子

```python
"""
TBE DSL 加法算子完整解析
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi.cce import util


def compute_add(input_x, input_y, alpha=1.0):
    """
    步骤1: 定义计算逻辑
    
    te.lang.cce.vadd 是向量化加法：
    - 输入: 两个形状相同的张量
    - 输出: 对应位置相加的结果
    - 自动向量化，一次处理多个元素
    """
    dtype = input_x.dtype
    
    # 第一步：标量乘法 (y = alpha * input_y)
    # te.lang.cce.vmuls 会编译为向量化指令
    input_y_scaled = te.lang.cce.vmuls(
        input_y, 
        tvm.const(alpha, dtype=dtype)  # 必须是 tvm.const
    )
    
    # 第二步：向量加法 (z = x + scaled_y)
    # te.lang.cce.vadd 自动利用 SIMD 指令
    result = te.lang.cce.vadd(input_x, input_y_scaled)
    
    return result


@fusion_manager.register("custom_add")
def custom_add_compute(input_x, input_y, output_z, alpha=1.0, kernel_name="custom_add"):
    """
    步骤2: 创建调度并编译
    
    这个函数会被框架调用，流程：
    1. 解析输入输出描述
    2. 创建 TVM 张量（声明）
    3. 调用计算函数（定义）
    4. 创建调度
    5. 编译生成 .o 文件
    """
    # 从输入描述获取元信息
    shape_x = input_x.get("shape")
    dtype_x = input_x.get("dtype")
    
    # 创建 TVM 占位符（声明输入张量）
    tensor_x = tvm.placeholder(
        shape_x, 
        dtype=dtype_x, 
        name="input_x"
    )
    tensor_y = tvm.placeholder(
        input_y.get("shape"),
        dtype=dtype_x,
        name="input_y"
    )
    
    # 调用计算函数，得到结果张量
    result_tensor = compute_add(tensor_x, tensor_y, alpha)
    
    # 创建调度
    with tvm.target.cce():
        # auto_schedule 会自动选择合适的调度策略
        # 包括：分块、向量化、并行化
        schedule = te.lang.cce.auto_schedule(result_tensor)
    
    # 编译
    config = {"print_ir": False, "need_build": True}
    te.lang.cce.build(
        schedule,
        # 绑定输入输出张量
        # tensor_x, tensor_y 是输入，result_tensor 是输出
        [tensor_x, tensor_y, result_tensor],
        "cce",  # 目标后端 (CCE = CANN Compute Engine)
        kernel_name=kernel_name,
        attrs=config
    )


def custom_add(input_x, input_y, output_z, alpha=1.0, kernel_name="custom_add"):
    """
    步骤3: 算子入口（参数检查）
    
    实际使用时会调用这个入口函数
    """
    # 参数检查
    util.check_shape_rule(input_x.get("shape"))
    util.check_shape_rule(input_y.get("shape"))
    util.check_dtype_rule(input_x.get("dtype"), ["float16", "float32"])
    
    # 调用核心计算
    custom_add_compute(input_x, input_y, output_z, alpha, kernel_name)
```

---

## 8. ACLNN 算子开发

### 8.1 原理详解：ACLNN 是什么？

ACNN (Ascend Cloud Neural Network) 是 CANN 8.0 推出的新一代算子开发接口，相比 TBE DSL 更加简洁。

#### ACLNN vs TBE DSL 对比

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ACLNN vs TBE DSL                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  TBE DSL:                                                          │
│    1. 定义计算 (compute)                                            │
│    2. 定义调度 (schedule) ← 需要手动优化                             │
│    3. 编译                                                           │
│    → 优点：极致性能                                                  │
│    → 缺点：需要理解硬件架构                                          │
│                                                                     │
│  ACLNN:                                                            │
│    1. 定义算子类 (继承 OpExecutor)                                   │
│    2. 实现 Init 和 Execute                                          │
│    3. 编译注册                                                      │
│    → 优点：接口简单，代码量少                                        │
│    → 缺点：优化空间有限                                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### ACLNN 工作流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                      ACLNN 开发流程                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. 编写算子类                                                      │
│     class CustomOp : public OpExecutor {                            │
│         aclnnStatus Init(...) override;    ← 解析输入输出             │
│         aclnnStatus Execute(...) override;  ← 实现计算逻辑            │
│     };                                                             │
│                                                                     │
│                          ↓                                          │
│                                                                     │
│  2. 编译成动态库                                                    │
│     g++ custom_op.cpp -o libcustom_op.so                          │
│                                                                     │
│                          ↓                                          │
│                                                                     │
│  3. 注册算子                                                        │
│     OP_REGISTER("CustomOp", CustomOp);                             │
│                                                                     │
│                          ↓                                          │
│                                                                     │
│  4. 框架调用                                                       │
│     框架通过 ACLNN API 调用注册的算子                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 OpExecutor 生命周期详解

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OpExecutor 生命周期                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  阶段1: 创建对象                                                    │
│  ─────────────                                                      │
│  new CustomOp() → 创建算子实例                                       │
│                                                                     │
│  阶段2: Init (初始化)                                               │
│  ─────────────                                                      │
│  Init(inputs, outputs, attrs)                                       │
│       │                                                             │
│       ├── 解析 inputs 获取:                                         │
│       │   ├── 输入数据指针 (void*)                                  │
│       │   ├── 输入形状 (vector<int64_t>)                           │
│       │   └── 数据类型 (DataType)                                  │
│       │                                                             │
│       ├── 解析 outputs 获取:                                        │
│       │   └── 输出数据指针和形状                                    │
│       │                                                             │
│       └── 解析 attrs 获取:                                          │
│           └── 算子属性 (如 alpha, 阈值等)                           │
│                                                                     │
│  阶段3: Execute (执行)                                              │
│  ────────────────────                                               │
│  Execute(stream)                                                    │
│       │                                                             │
│       ├── 从输入指针读取数据                                        │
│       ├── 执行计算                                                  │
│       └── 写入输出指针                                              │
│                                                                     │
│  阶段4: 销毁                                                        │
│  ──────────                                                         │
│  delete 或智能指针自动释放                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.3 核心接口详解

```cpp
// OpExecutor 是所有 ACLNN 算子的基类
class OpExecutor {
public:
    virtual ~OpExecutor() = default;
    
    // ====================
    // 必须实现的方法
    // ====================
    
    // 1. 获取算子名称
    //    框架通过这个名称识别算子
    virtual const char* GetOpName() const = 0;
    
    // 2. 初始化
    //    在这里解析输入输出描述，保存需要的参数
    virtual aclnnStatus Init(
        const OpArgs& inputs,   // 输入参数
        const OpArgs& outputs,  // 输出参数
        const OpArgs& attrs     // 属性参数
    ) = 0;
    
    // 3. 执行
    //    在这里实现具体的计算逻辑
    virtual aclnnStatus Execute(aclrtStream stream) = 0;
    
    // ====================
    // 可选重写的方法
    // ====================
    
    // 返回计算应该在哪个硬件上执行
    virtual op::CoreType GetCoreType() const {
        return op::CoreType::AI_CORE;  // 默认 NPU
    }
    
    // 返回实现模式
    virtual op::OpImplMode GetOpImplMode() const {
        return op::OpImplMode::IMPL_MODE_DEFAULT;
    }
};
```

### 8.4 OpArgs 接口详解

```cpp
// OpArgs 是参数的容器
class OpArgs {
public:
    // 获取参数数量
    size_t Size() const;
    
    // 获取第 i 个参数
    OpArg GetArg(size_t i) const;
};

// OpArg 是单个参数
class OpArg {
public:
    // 获取数据指针
    void* GetData() const;
    
    // 获取形状
    std::vector<int64_t> GetShape() const;
    
    // 获取数据类型
    opdev::DataType GetDataType() const;
    
    // 获取属性值（标量）
    template<typename T>
    T GetScalar() const;
};

// 使用示例
void MyOp::Init(const OpArgs& inputs, 
                const OpArgs& outputs, 
                const OpArgs& attrs) {
    
    // 获取第一个输入
    auto& input_desc = inputs.GetArg(0);
    void* input_data = input_desc.GetData();           // 数据指针
    auto shape = input_desc.GetShape();                 // 形状
    auto dtype = input_desc.GetDataType();             // 类型
    
    // 获取属性
    if (attrs.Size() > 0) {
        float alpha = attrs.GetArg(0).GetScalar<float>();
    }
}
```

### 8.5 Execute 中的数据访问

```cpp
// Execute 是核心计算逻辑
aclnnStatus MyOp::Execute(aclrtStream stream) {
    
    // ====================
    // Step 1: 指针类型转换
    // ====================
    // 输入数据是 void*，需要转换为具体类型
    float* input_x = static_cast<float*>(input_x_);
    float* input_y = static_cast<float*>(input_y_);
    float* output = static_cast<float*>(output_);
    
    // ====================
    // Step 2: 执行计算
    // ====================
    for (int64_t i = 0; i < num_elements_; ++i) {
        output[i] = input_x[i] + alpha_ * input_y[i];
    }
    
    // ====================
    // 注意：这是朴素实现
    // 实际生产代码需要：
    // 1. 使用向量化指令
    // 2. 分块处理提高缓存命中率
    // 3. 使用异步操作
    // ====================
    
    return aclnnStatus::ACNN_SUCCESS;
}
```

### 8.6 完整示例：加法算子

```cpp
/**
 * 文件位置: 03_aclnn/custom_add.h
 * ACLNN 加法算子头文件
 * 
 * 算子功能: output = input_x + alpha * input_y
 */
#ifndef CUSTOM_ADD_H
#define CUSTOM_ADD_H

#include "aclnn/opdev/op_executor.h"
#include "aclnn/opdev/op_def.h"
#include "aclnn/opdev/common_types.h"
#include <vector>

namespace aclnn_op {

class CustomAddOp : public opdev::OpExecutor {
public:
    CustomAddOp() = default;
    ~CustomAddOp() override = default;
    
    // 算子名称，框架通过这个名字找到算子
    const char* GetOpName() const override {
        return "CustomAdd";
    }
    
    // 初始化：解析输入输出和属性
    aclnnStatus Init(
        const opdev::OpArgs& inputs,   // [input_x, input_y]
        const opdev::OpArgs& outputs,  // [output]
        const opdev::OpArgs& attrs    // [alpha]
    ) override;
    
    // 执行：实现计算逻辑
    aclnnStatus Execute(aclrtStream stream) override;
    
    // 指定在 NPU 上执行
    op::CoreType GetCoreType() const override {
        return op::CoreType::AI_CORE;
    }

private:
    // 数据指针（在 Init 中获取）
    void* input_x_ = nullptr;    // 输入 X
    void* input_y_ = nullptr;    // 输入 Y
    void* output_ = nullptr;     // 输出
    
    // 张量信息（在 Init 中获取）
    std::vector<int64_t> shape_;     // 形状
    opdev::DataType data_type_;       // 数据类型
    float alpha_ = 1.0f;             // 属性
    int64_t num_elements_ = 1;       // 元素数量
};

} // namespace aclnn_op

#endif // CUSTOM_ADD_H
```

```cpp
/**
 * 文件位置: 03_aclnn/custom_add.cpp
 * ACLNN 加法算子实现
 */
#include "custom_add.h"
#include "aclnn/opdev/op_log.h"
#include "aclnn/opdev/data_type_utils.h"
#include <cstring>
#include <cmath>

namespace aclnn_op {

// ==================== Init 实现 ====================

aclnnStatus CustomAddOp::Init(
    const opdev::OpArgs& inputs,
    const opdev::OpArgs& outputs,
    const opdev::OpArgs& attrs) {
    
    // ====================
    // 从 inputs 获取输入信息
    // ====================
    auto& input_x_desc = inputs.GetArg(0);  // 第一个输入
    auto& input_y_desc = inputs.GetArg(1);  // 第二个输入
    
    // 获取数据指针
    input_x_ = input_x_desc.GetData();
    input_y_ = input_y_desc.GetData();
    
    // 获取形状
    shape_ = input_x_desc.GetShape();
    
    // 获取数据类型
    data_type_ = input_x_desc.GetDataType();
    
    // ====================
    // 从 outputs 获取输出信息
    // ====================
    auto& output_desc = outputs.GetArg(0);
    output_ = output_desc.GetData();
    
    // ====================
    // 从 attrs 获取属性
    // ====================
    if (attrs.Size() > 0) {
        // 获取第一个属性 (alpha)
        alpha_ = attrs.GetArg(0).GetScalar<float>();
    }
    
    // ====================
    // 计算元素总数
    // ====================
    num_elements_ = 1;
    for (auto dim : shape_) {
        num_elements_ *= dim;
    }
    
    // 打印日志（调试用）
    ACLNN_LOGI("CustomAddOp Init:"
             << " shape=" << ShapeToString(shape_)
             << " dtype=" << opdev::DataTypeToString(data_type_)
             << " alpha=" << alpha_
             << " elements=" << num_elements_);
    
    return aclnnStatus::ACNN_SUCCESS;
}

// ==================== Execute 实现 ====================

aclnnStatus CustomAddOp::Execute(aclrtStream stream) {
    ACLNN_LOGI("CustomAddOp Execute started");
    
    // 根据数据类型选择计算方式
    switch (data_type_) {
        case opdev::DataType::FLOAT32: {
            // ====================
            // float32 计算
            // ====================
            float* x = static_cast<float*>(input_x_);
            float* y = static_cast<float*>(input_y_);
            float* out = static_cast<float*>(output_);
            
            // 朴素循环实现
            // 注意：实际应该用向量化指令优化
            for (int64_t i = 0; i < num_elements_; ++i) {
                out[i] = x[i] + alpha_ * y[i];
            }
            break;
        }
        
        case opdev::DataType::FLOAT16: {
            // ====================
            // float16 计算
            // ====================
            // float16 需要额外转换
            auto* x_raw = static_cast<uint16_t*>(input_x_);
            auto* y_raw = static_cast<uint16_t*>(input_y_);
            auto* out_raw = static_cast<uint16_t*>(output_);
            
            for (int64_t i = 0; i < num_elements_; ++i) {
                // 从 float16 转换为 float32
                float fx = HalfToFloat(x_raw[i]);
                float fy = HalfToFloat(y_raw[i]);
                
                // 计算
                float result = fx + alpha_ * fy;
                
                // 转换回 float16
                out_raw[i] = FloatToHalf(result);
            }
            break;
        }
        
        case opdev::DataType::INT32: {
            int32_t* x = static_cast<int32_t*>(input_x_);
            int32_t* y = static_cast<int32_t*>(input_y_);
            int32_t* out = static_cast<int32_t*>(output_);
            
            for (int64_t i = 0; i < num_elements_; ++i) {
                out[i] = x[i] + static_cast<int32_t>(alpha_) * y[i];
            }
            break;
        }
        
        default:
            ACLNN_LOGE("Unsupported data type: " 
                     << opdev::DataTypeToString(data_type_));
            return aclnnStatus::ACNN_STATUS_INVALID_PARAM;
    }
    
    ACLNN_LOGI("CustomAddOp Execute completed");
    return aclnnStatus::ACNN_SUCCESS;
}

// ==================== 注册算子 ====================
// 这个宏将算子注册到 ACLNN 框架
// 注册后，框架可以通过名字 "CustomAdd" 找到这个算子
OP_REGISTER("CustomAdd", CustomAddOp);

} // namespace aclnn_op
```

### 8.7 向量化优化示例

上面的朴素循环性能不好，实际应该使用向量化指令：

```cpp
aclnnStatus CustomAddOp::ExecuteOptimized(aclrtStream stream) {
    
    // ====================
    // 优化版本：使用向量化
    // ====================
    
    // 假设数据长度是 16 的倍数
    int64_t vector_count = num_elements_ / 16;
    
    switch (data_type_) {
        case opdev::DataType::FLOAT32: {
            // 使用 4 元素向量 (128-bit)
            // 一次处理 4 个 float32
            for (int64_t i = 0; i < vector_count; ++i) {
                // 注意：这是伪代码，实际需要使用 NPU intrinsic
                // _mm256_load_ps 等价于 NPU 的 vector_load
                // _mm256_mul_ps 等价于 NPU 的 vector_mul
                // _mm256_add_ps 等价于 NPU 的 vector_add
                // _mm256_store_ps 等价于 NPU 的 vector_store
                
                // float* x_vec = load 4 floats from x
                // float* y_vec = load 4 floats from y
                // float* out_vec = x_vec + alpha * y_vec
                // store out_vec to output
            }
            break;
        }
    }
    
    // 处理剩余元素（不是 16 倍数的部分）
    // 处理尾部...
    
    return aclnnStatus::ACNN_SUCCESS;
}
```

### 8.8 ACLNN vs TBE DSL 总结

| 方面 | ACLNN | TBE DSL |
|------|-------|---------|
| 代码量 | 少 | 多 |
| 学习曲线 | 平缓 | 陡峭 |
| 性能 | 良好 | 极致 |
| 灵活性 | 一般 | 高 |
| 调度控制 | 无 | 完全控制 |
| 适用场景 | 快速开发 | 性能优化 |

---

## 9. 性能优化策略

### 9.1 通用优化原则

| 策略 | 说明 | 效果 |
|------|------|------|
| **算子融合** | 将多个操作合并 | 减少内存访问 |
| **Tiling** | 分块处理大数据 | 提高缓存命中率 |
| **向量化** | 使用 SIMD 指令 | 提高计算吞吐 |
| **异步执行** | 异步数据传输 | 隐藏延迟 |
| **内存对齐** | 对齐访问地址 | 提高带宽利用率 |

### 9.2 Tiling 优化示例

```python
# 未优化的矩阵乘法
for i in range(M):
    for j in range(N):
        for k in range(K):
            C[i,j] += A[i,k] * B[k,j]

# Tiling 优化 (以 16x16 为块)
for ii in range(0, M, 16):
    for jj in range(0, N, 16):
        for kk in range(0, K, 16):
            # 处理 16x16 块
            for i in range(ii, min(ii+16, M)):
                for j in range(jj, min(jj+16, N)):
                    # 累加
```

### 9.3 算子融合优化

```python
# 融合前: 3 个 kernel
y1 = conv(x)      # kernel 1
y2 = bn(y1)       # kernel 2
y3 = relu(y2)     # kernel 3

# 融合后: 1 个 kernel
y3 = fused_conv_bn_relu(x)  # kernel 1 (融合)
```

**融合优势**：
1. 减少内存读写
2. 减少 kernel 启动开销
3. 提高数据局部性

---

## 10. 调试与验证

### 10.1 环境检查工具

```python
# 文件位置: common/check_env.py
# 用于检查开发环境是否就绪

import os
from pathlib import Path

def check_env():
    checks = [
        ("Python", lambda: True),
        ("CANN", lambda: Path(os.getenv("ASCEND_TOOLKIT_HOME", "")).exists()),
        ("NPU Device", lambda: Path("/dev/davinci0").exists()),
    ]
    
    for name, check in checks:
        status = "✅" if check() else "❌"
        print(f"{status} {name}")

if __name__ == "__main__":
    check_env()
```

### 10.2 正确性验证

```python
# 对比自定义算子和参考实现
def verify_op(custom_op, reference_op, inputs):
    custom_result = custom_op(*inputs)
    reference_result = reference_op(*inputs)
    
    diff = abs(custom_result - reference_result).max()
    assert diff < 1e-4, f"验证失败，最大差异: {diff}"
    print(f"✅ 验证通过，最大差异: {diff:.6f}")
```

### 10.3 性能测试

```python
import time

def benchmark(func, *args, iterations=100):
    # 预热
    for _ in range(10):
        func(*args)
    
    # 计时
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - start)
    
    avg_time = sum(times) / len(times)
    print(f"平均耗时: {avg_time*1000:.2f} ms")
```

---

## 附录：快速参考

### 环境变量

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
ASCEND_OPP_PATH=${ASCEND_TOOLKIT_HOME}/opp
LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:...
PYTHONPATH=${ASCEND_TOOLKIT_HOME}/python/site-packages:...
```

### 关键路径

| 路径 | 说明 |
|------|------|
| `/usr/local/Ascend/ascend-toolkit/8.0.0/` | CANN 安装目录 |
| `/usr/local/Ascend/ascend-toolkit/8.0.0/include/` | 头文件 |
| `/usr/local/Ascend/ascend-toolkit/8.0.0/lib64/` | 库文件 |
| `/usr/local/Ascend/ascend-toolkit/8.0.0/opp/` | 算子库 |
| `/dev/davinci0` | NPU 设备节点 |

### 常用命令

```bash
# 检查 NPU 状态
npu-smi

# 环境检查
python3 common/check_env.py

# 编译 TBE 算子
bash build.sh

# 运行测试
python3 01_pytorch_npu/custom_ops.py
```

---

**文档版本**: 1.0  
**更新时间**: 2026-05-28  
**适用环境**: 香橙派昇腾开发板 (Ascend310B1) + CANN 8.0.0
