# MindSpore 自定义算子开发指南

MindSpore 是华为自研的 AI 框架，提供了多种自定义算子的方式。

## 环境要求

```bash
# 安装 MindSpore (NPU 版本)
pip install mindspore-2.2.11 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 目录结构

```
04_mindspore/
├── custom_op.py       # 自定义算子示例
└── README.md
```

## 开发方式

### 方式一：使用 nn.Cell（推荐）

适合组合多个已有算子，构建复杂网络结构：

```python
class MyOpCell(nn.Cell):
    def __init__(self):
        super().__init__()
        self.op1 = ops.SomeOp()
        self.op2 = ops.AnotherOp()
    
    def construct(self, x):
        return self.op2(self.op1(x))
```

### 方式二：使用 ops.Custom

适合需要自定义前向或反向逻辑的场景：

```python
custom_op = ops.Custom(
    forward_func=forward_fn,
    bprop_func=backward_fn,
    out_shape=lambda x: x,
    out_dtype=lambda x: x,
    func_type="hybrid"  # 或 "akg" 用于高性能
)
```

### 方式三：使用 PyBoost

MindSpore 2.0+ 的新特性，使用 Python 原生语法：

```python
@ms.pyboost
def my_op(x, y):
    return x + y
```

## 常用 ops 一览

| 算子 | 说明 |
|------|------|
| `ops.MatMul` | 矩阵乘法 |
| `ops.Conv2d` | 二维卷积 |
| `ops.BatchNorm` | 批归一化 |
| `ops.ReLU/Sigmoid/Tanh` | 激活函数 |
| `ops.Softmax` | Softmax |
| `ops.LayerNorm` | 层归一化 |

## 运行测试

```bash
cd /home/HwHiAiUser/Desktop/CNN/Operator/04_mindspore
python custom_op.py
```
