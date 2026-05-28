# 昇腾算子开发指南

本目录提供在香橙派昇腾开发板（OrangePi Ai Pro）上进行算子开发的完整指南和示例代码。

## 环境信息

| 项目 | 版本/信息 |
|------|-----------|
| 设备 | OrangePi Ai Pro |
| 架构 | ARM64 (aarch64) |
| CANN | 8.0.0 |
| PyTorch | 2.1.0 + NPU |
| MindSpore | 2.4.10 |

## 目录结构

```
Operator/
├── README.md                    # 本文件
├── run_tests.py                 # 快速测试脚本
│
├── 01_pytorch_npu/              # PyTorch NPU 算子（推荐入门）
│   ├── custom_ops.py             # 自定义算子实现
│   └── README.md                 # 详细说明
│
├── 02_tbe_dsl/                 # TBE DSL 算子（传统方式）
│   ├── custom_add.py             # 加法算子
│   ├── custom_matmul.py          # 矩阵乘法算子
│   └── README.md                 # 详细说明
│
├── 03_aclnn/                   # ACLNN 算子（CANN 8.0 新接口）
│   ├── custom_add.h              # 头文件
│   ├── custom_add.cpp            # 实现
│   ├── CMakeLists.txt            # 构建配置
│   └── README.md                 # 详细说明
│
├── 04_mindspore/               # MindSpore 算子
│   ├── custom_op.py              # 自定义算子
│   └── README.md                 # 详细说明
│
├── common/                      # 公共工具
│   ├── check_env.py              # 环境检查脚本
│   └── utils.py                  # 常用工具函数
│
├── 05_development_guide.md      # 算子开发指南
└── 06_performance_analysis.md   # 性能分析指南
```

## 性能分析与日志

### 1. 算子性能分析脚本

| 脚本 | 功能 | 使用方法 |
|------|------|----------|
| `MobileNet/operator_analysis.py` | MobileNetV3 算子分析 | `python operator_analysis.py` |
| `ResNet/operator_analysis.py` | ResNet50 算子分析 | `python operator_analysis.py` |

### 2. 输出日志

| 日志文件 | 说明 |
|----------|------|
| `MobileNet/operator_log.md` | MobileNetV3 性能分析报告 |
| `ResNet/operator_log.md` | ResNet50 性能分析报告 |
| `MobileNet/operator_log.json` | 原始数据 (JSON) |
| `ResNet/operator_log.json` | 原始数据 (JSON) |

### 3. 分析内容

- **端到端性能**：延迟、吞吐量、FPS
- **算子级分析**：Conv2d、DepthwiseConv、BN、ReLU 等
- **资源利用**：AI Core、Cube、Vector 利用率
- **对比分析**：MobileNet vs ResNet

### 4. 运行分析

```bash
# 分析 MobileNetV3
cd MobileNet
python operator_analysis.py

# 分析 ResNet50
cd ResNet
python operator_analysis.py
```

## 快速开始

### 1. 检查环境

```bash
cd /home/HwHiAiUser/Desktop/CNN/Operator
python3 common/check_env.py
```

### 2. 运行测试

```bash
# 运行所有测试
python3 run_tests.py all

# 运行特定测试
python3 run_tests.py pytorch
python3 run_tests.py mindspore
python3 run_tests.py env
```

## 算子开发方式对比

| 方式 | 难度 | 适用场景 | 推荐度 | 性能 |
|------|------|----------|--------|------|
| **PyTorch NPU** | ⭐ | 快速验证、简单算子 | ⭐⭐⭐⭐⭐ | 良好 |
| **MindSpore** | ⭐⭐ | 使用 MindSpore 框架 | ⭐⭐⭐ | 良好 |
| **ACLNN** | ⭐⭐⭐ | 新项目、需要高性能 | ⭐⭐⭐⭐ | 优秀 |
| **TBE DSL** | ⭐⭐⭐⭐ | 极致性能、复杂算子 | ⭐⭐⭐ | 最优 |

## 各方式详解

### 方式一：PyTorch NPU 自定义算子（推荐入门）

**优点**：
- 最简单，上手快
- 无需编译
- 适合快速验证算法思路
- 可复用 PyTorch 生态

**缺点**：
- 性能不如手写 kernel
- 复杂算子效率较低

**适用场景**：
- 算法研究和验证
- 简单的逐元素操作
- 原型开发

**示例代码**：

```python
import torch

class SwishOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        result = x * torch.sigmoid(x)
        ctx.save_for_backward(x)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        sigmoid_x = torch.sigmoid(x)
        grad_x = grad_output * sigmoid_x * (1 + x * (1 - sigmoid_x))
        return grad_x

# 使用
x = torch.randn(10, 10).npu()
y = SwishOp.apply(x)
```

### 方式二：MindSpore 自定义算子

**优点**：
- 与 MindSpore 框架无缝集成
- 支持 CPU/NPU 统一编程
- 丰富的算子组合能力

**缺点**：
- 需要熟悉 MindSpore API
- 灵活性不如底层开发

**适用场景**：
- 使用 MindSpore 的项目
- 需要同时支持多种后端
- 网络层开发

**示例代码**：

```python
import mindspore as ms
import mindspore.nn as nn

class CustomSwish(nn.Cell):
    def __init__(self):
        super().__init__()
    
    def construct(self, x):
        return x * ms.ops.sigmoid(x)
```

### 方式三：ACLNN 算子开发

**优点**：
- CANN 8.0 官方推荐
- 接口现代化
- 支持 CPU/NPU 双后缀
- 性能优秀

**缺点**：
- 需要 C++ 编译
- 学习曲线适中

**适用场景**：
- 新项目初始化
- 需要高性能的生产代码
- 跨平台部署

**示例代码**：

```cpp
#include "aclnn/opdev/op_executor.h"

class CustomAddOp : public opdev::OpExecutor {
public:
    aclnnStatus Init(...) override { /* 初始化 */ }
    aclnnStatus Execute(aclrtStream stream) override { /* 执行计算 */ }
};

// 注册算子
OP_REGISTER("CustomAdd", CustomAddOp);
```

### 方式四：TBE DSL 开发

**优点**：
- 极致性能
- 精确控制 tiling
- 昇腾官方深度优化

**缺点**：
- 开发复杂
- 需要深入了解硬件架构
- 调试困难

**适用场景**：
- 极致性能要求
- 复杂张量操作
- 量产部署

**示例代码**：

```python
import te.lang.cce
from te import tvm

def compute_add(input_x, input_y):
    return te.lang.cce.vadd(input_x, input_y)

@fusion_manager.register("custom_add")
def custom_add_compute(inputs, outputs, kernel_name):
    tensor_x = tvm.placeholder(shape, dtype="float16")
    tensor_y = tvm.placeholder(shape, dtype="float16")
    result = compute_add(tensor_x, tensor_y)
    
    with tvm.target.cce():
        schedule = te.lang.cce.auto_schedule(result)
    
    te.lang.cce.build(schedule, ...)
```

## 性能优化建议

### 1. 选择合适的开发方式

```
简单验证 → PyTorch NPU
   ↓ 性能不够？
MindSpore / ACLNN
   ↓ 仍不够？
TBE DSL
```

### 2. 常见优化策略

1. **算子融合**：减少内存访问
2. **Tiling 优化**：充分利用缓存
3. **向量化**：使用 SIMD 指令
4. **异步执行**：隐藏数据传输延迟
5. **内存对齐**：提高访问效率

### 3. Profiling 工具

```bash
# 使用 PyTorch Profiler
python -m torch.profiler ...

# 使用 CANN Profiler
msprof --output=./prof
```

## 常见问题

### Q1: ATC 编译器无法运行

检查环境变量：
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### Q2: NPU 不可用

检查设备节点：
```bash
ls -la /dev/davinci*
```

### Q3: 编译失败

检查依赖库路径：
```bash
echo $LD_LIBRARY_PATH
```

## 参考资源

- [昇腾开发者社区](https://www.hiascend.com/)
- [CANN 文档](https://www.hiascend.com/document/)
- [PyTorch NPU 文档](https://github.com/Ascend/pytorch)
- [MindSpore 文档](https://www.mindspore.cn/)

## 许可

本项目代码遵循 Apache 2.0 许可。
