# PyTorch NPU 原生内核扩展示例

本目录展示如何使用 C++ 扩展实现 NPU 原生内核，获得更好的性能。

## 文件说明

- `custom_add.cpp` - NPU 原生内核实现
- `custom_add.py` - Python 绑定
- `setup.py` - 编译脚本

## 内核实现示例 (custom_add.cpp)

```cpp
#include <torch/torch.h>
#include <torch/extension.h>

// 使用 ATen 格式实现自定义算子
// 这种方式可以更好地利用 NPU 的向量化指令

torch::Tensor npu_custom_add_forward(torch::Tensor x, torch::Tensor y, float alpha) {
    // 检查设备类型
    TORCH_CHECK(x.is_npu(), "输入 x 必须在 NPU 上");
    TORCH_CHECK(y.is_npu(), "输入 y 必须在 NPU 上");
    TORCH_CHECK(x.sizes() == y.sizes(), "输入维度必须匹配");
    
    // 直接使用 at::add 实现
    return at::add(x, y, alpha);
}

torch::Tensor npu_custom_add_backward(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor y,
    float alpha) {
    
    // 计算 x 的梯度
    torch::Tensor grad_x = grad_output;
    // 计算 y 的梯度
    torch::Tensor grad_y = grad_output * alpha;
    
    return std::make_tuple(grad_x, grad_y);
}

// Python 绑定
torch::Tensor custom_add(torch::Tensor x, torch::Tensor y, float alpha = 1.0) {
    return npu_custom_add_forward(x, y, alpha);
}

// 梯度版本（需要保存中间结果）
class CustomAddFunction : public torch::autograd::Function<CustomAddFunction> {
public:
    static torch::autograd::Variable forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor x,
        torch::Tensor y,
        float alpha) {
        
        torch::Tensor output = at::add(x, y, alpha);
        ctx->save_for_backward({x, y});
        ctx->saved_data["alpha"] = alpha;
        
        return output;
    }
    
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_output) {
        
        auto saved = ctx->get_saved_variables();
        auto x = saved[0];
        auto y = saved[1];
        float alpha = ctx->saved_data["alpha"].toFloat();
        
        torch::Tensor grad_x = grad_output[0];
        torch::Tensor grad_y = grad_output[0] * alpha;
        
        return {grad_x, grad_y, torch::Tensor()};
    }
};

// 模块接口
class CustomAddModule : public torch::nn::Module {
public:
    float alpha = 1.0;
    
    torch::Tensor forward(torch::Tensor x, torch::Tensor y) {
        return CustomAddFunction::apply(x, y, alpha);
    }
};

// Python 绑定
TORCH_LIBRARY(npu_custom_ops, m) {
    m.def("add(Tensor x, Tensor y, float alpha=1.0) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu_custom_ops, NPU, m) {
    m.impl("add", &custom_add);
}
```

## Python 绑定示例 (custom_add.py)

```python
import torch
import torch_npu
from torch.utils.cpp_extension import load

# 加载 C++ 扩展（如果已编译）
try:
    custom_ops = load(
        name="npu_custom_ops",
        sources=["custom_add.cpp"],
        extra_include_paths=[],
        verbose=False
    )
    print("✅ C++ 扩展加载成功")
except:
    print("⚠️  C++ 扩展未编译，将使用纯 Python 实现")
    custom_ops = None


def native_add(x, y, alpha=1.0):
    """
    原生 NPU 加法算子
    
    Args:
        x: 第一个输入张量
        y: 第二个输入张量
        alpha: y 的缩放因子
    
    Returns:
        x + alpha * y
    """
    if custom_ops is not None:
        return custom_ops.add(x, y, alpha)
    else:
        # 回退到 PyTorch 实现
        return torch.add(x, y, alpha)
```

## 编译脚本 (setup.py)

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='npu_custom_ops',
    version='1.0',
    ext_modules=[
        CUDAExtension(
            name='npu_custom_ops',
            sources=['custom_add.cpp'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': ['-O3', '--use_fast_math']
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires='>=3.9',
)
```

## 使用方法

```bash
# 1. 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. 编译扩展
python setup.py build

# 3. 安装扩展
python setup.py install

# 4. 测试
python custom_add.py
```

## 注意事项

1. 编译需要 NPU 的头文件，通常位于：
   `/usr/local/Ascend/ascend-toolkit/8.0.0/aarch64-linux/include/`

2. 确保 `LD_LIBRARY_PATH` 包含 NPU 驱动库

3. 如果编译失败，检查 `PYTHONPATH` 是否正确配置
