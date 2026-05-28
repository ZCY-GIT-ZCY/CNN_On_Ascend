# TBE DSL 算子开发指南

TBE (Tensor Boost Engine) 是昇腾 AI 处理器提供的算子开发框架，使用 Python DSL 进行算子开发。

## 环境准备

```bash
# 1. 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. 验证 TBE 环境
python3 -c "import te; print(te.__version__)"
```

## 目录结构

```
02_tbe_dsl/
├── custom_add.py       # 加法算子
├── custom_matmul.py    # 矩阵乘法算子
└── README.md
```

## 核心概念

### 1. TVM 张量

TBE 使用 TVM 作为底层计算抽象：

```python
import tvm

# 创建占位符张量
tensor = tvm.placeholder(shape, dtype="float16", name="input")
```

### 2. te.lang.cce API

常用的计算 API：

| API | 说明 |
|-----|------|
| `vadd` | 向量加法 |
| `vsub` | 向量减法 |
| `vmul` | 向量乘法 |
| `vdiv` | 向量除法 |
| `vmuls` | 标量乘法 |
| `matmul` | 矩阵乘法 |
| `relu` | ReLU 激活 |
| `vmax/vmin` | 逐元素最大/最小值 |
| `reshape` | 张量变形 |
| `transpose` | 张量转置 |

### 3. 融合算子

使用 `@fusion_manager.register` 装饰器注册融合算子：

```python
from te.platform.fusion_manager import fusion_manager

@fusion_manager.register("my_op")
def my_op_compute(inputs, outputs, kernel_name="my_op"):
    # 实现算子计算
    pass
```

## 开发流程

### 1. 定义计算函数

```python
def compute_xxx(input_x, input_y):
    dtype = input_x.dtype
    
    # 使用 te.lang.cce API 实现计算
    result = te.lang.cce.vadd(input_x, input_y)
    
    return result
```

### 2. 定义融合算子

```python
@fusion_manager.register("custom_op")
def custom_op_compute(input_x, input_y, output, kernel_name):
    shape = input_x.get("shape")
    dtype = input_x.get("dtype")
    
    # 创建 TVM 张量
    tensor_x = tvm.placeholder(shape, dtype=dtype, name="input_x")
    tensor_y = tvm.placeholder(shape, dtype=dtype, name="input_y")
    
    # 计算
    result = compute_xxx(tensor_x, tensor_y)
    
    # 调度
    with tvm.target.cce():
        schedule = te.lang.cce.auto_schedule(result)
    
    # 编译
    te.lang.cce.build(schedule, [tensor_x, tensor_y, result], 
                      "cce", kernel_name=kernel_name)
```

### 3. 完整项目结构

```
custom_op/
├── op_impl/
│   └── ai_core/
│       └── tbe/
│           ├── config/
│           │   └── kir_k610v1_xx.json     # 芯片配置
│           └── impl/
│               ├── custom_add.py           # 算子实现
│               └── custom_add.json         # 算子定义
├── op_proto/
│   └── custom_add.proto                    # 算子原型
└── scripts/
    └── build.sh                            # 编译脚本
```

## 算子 JSON 配置

```json
{
    "op": "CustomAdd",
    "input_desc": [
        {
            "name": "x",
            "shape": [1, 3, 224, 224],
            "dtype": "float16",
            "format": "NCHW"
        },
        {
            "name": "y",
            "shape": [1, 3, 224, 224],
            "dtype": "float16",
            "format": "NCHW"
        }
    ],
    "output_desc": [
        {
            "name": "z",
            "shape": [1, 3, 224, 224],
            "dtype": "float16",
            "format": "NCHW"
        }
    ],
    "attr": [
        {
            "name": "alpha",
            "type": "float",
            "value": 1.0
        }
    ]
}
```

## 编译和部署

```bash
# 1. 编译算子
bash scripts/build.sh

# 2. 生成算子包
zip -r custom_op.om *.o

# 3. 注册算子到 OPP
cp custom_op.om $ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe/op_api/
```

## 性能优化建议

1. **使用融合算子**：将多个操作融合，减少内存访问
2. **选择合适的 Tiling 策略**：根据芯片缓存大小优化
3. **使用向量化指令**：充分利用 SIMD 能力
4. **减少数据搬移**：尽量在片上内存完成计算

## 参考资料

- [昇腾算子开发文档](https://www.hiascend.com/document/)
- [TBE DSL API 参考](https://www.hiascend.com/document/...)
