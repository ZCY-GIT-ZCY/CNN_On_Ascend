# 香橙派 AIpro ResNet50 部署实验日志

**实验日期**: 2026-05-21
**实验设备**: 香橙派 AIpro (Ascend 310B1)
**实验目的**: 在昇腾 NPU 上完成 CNN 模型离线部署与推理验证

---

## 一、实验环境

### 1.1 硬件规格

| 项目 | 规格 |
|------|------|
| NPU 芯片 | Ascend 310B1 |
| AI 算力 | 8 TOPS (INT8) / 4 TFLOPS (FP16) |
| CPU | 4 核 64 位 Arm 处理器 |
| 内存 | 8GB / 16GB LPDDR4X |
| 操作系统 | Ubuntu 22.04 |

### 1.2 软件环境

| 项目 | 版本/配置 |
|------|----------|
| CANN | 23.0.0 |
| Python | 3.9.2 |
| Conda 环境 | ascend_cnn |
| ATC | 已配置 |
| OpenCV | 4.13.0.92 |
| NumPy | 1.24.4 |
| ONNX | 1.17.0 |
| PyTorch | 2.4.1 (CPU) |
| torchvision | 0.19.1 |

---

## 二、实验步骤

### 2.1 环境准备

1. **创建 Conda 环境**
   ```bash
   conda create -n ascend_cnn python=3.8 -y
   ```

2. **安装 Python 依赖** (使用清华镜像源)
   ```bash
   pip install numpy opencv-python onnx torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

3. **下载 ACLLite 工具库**
   - 从华为官方仓库下载核心推理库文件到 `acllite_utils/` 目录

4. **补充 ATC 依赖**
   ```bash
   pip install decorator attrs psutil jinja2 tornado cloudpickle scipy ml-dtypes -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

### 2.2 ONNX 模型导出

**脚本**: `export_onnx.py`

```python
# 导出 ResNet50 为 ONNX
torch.onnx.export(
    model,
    dummy_input,
    "resnet50.onnx",
    input_names=["input_image"],
    output_names=["output_tensor"],
    opset_version=11
)
```

**结果**:
- 模型文件: `resnet50.onnx` (98MB)
- 输入 Shape: (1, 3, 224, 224)
- 验证: ONNX 模型结构检查通过

### 2.3 AIPP 配置

**文件**: `aipp.cfg`

AIPP (Artificial Intelligence Pre-Processing) 用于硬件加速图像预处理。

**归一化参数计算** (ImageNet 标准):
- RGB 均值: [0.485, 0.456, 0.406]
- RGB 标准差: [0.229, 0.224, 0.225]

**AIPP 参数计算公式**:
```
mean_chn_i = 255 × 原始均值
var_reci_chn_i = 1 / (255 × 原始标准差)
```

**计算结果** (BGR 通道顺序):
```
mean_chn_0 (B): 104
mean_chn_1 (G): 116
mean_chn_2 (R): 124

var_reci_chn_0: 0.01742919
var_reci_chn_1: 0.01750700
var_reci_chn_2: 0.01712475
```

> **注意**: 提供了 `gen_aipp_cfg.py` 工具用于自动计算自定义参数的 AIPP 配置。

### 2.4 ATC 模型编译

**编译命令**:
```bash
atc --model=resnet50.onnx \
    --framework=5 \
    --output=resnet50_aipp \
    --soc_version=Ascend310B1 \
    --input_shape="input_image:1,3,224,224" \
    --insert_op_conf=aipp.cfg \
    --enable_small_channel=1
```

**结果**:
- 输出文件: `resnet50_aipp.om` (50MB)
- 编译时间: ~4 分钟
- 状态: ✅ ATC run success

### 2.5 ACLLite 工具库

**目录**: `acllite_utils/`

| 文件 | 说明 |
|------|------|
| `acllite_resource.py` | ACL 资源管理 |
| `acllite_model.py` | 模型加载与推理 |
| `acllite_image.py` | 图像处理 |
| `acllite_logger.py` | 日志工具 |
| `acllite_utils.py` | 通用工具函数 |
| `constants.py` | 常量定义 |
| `__init__.py` | 模块初始化 |

---

## 三、推理验证

### 3.1 测试图片

| 图片 | 内容 | 尺寸 | 预期类别 |
|------|------|------|----------|
| `test.jpg` | 狗 | 1546x1213 | 狗 (Dog) |
| `car.png` | 车 | 1125x670 | 车 (Car) |

### 3.2 分类结果

#### test.jpg (狗)

| 排名 | 类别 ID | 类别名称 | 置信度 |
|------|---------|----------|--------|
| 1 | 258 | Maltese dog | 12.9766 |
| 2 | 259 | Samoyed | 12.0469 |
| 3 | 261 | Pug | 11.2422 |
| 4 | 260 | Pomeranian | 9.1250 |
| 5 | 279 | - | 8.1953 |

**结果**: ✅ 正确识别为犬类

#### car.png (车)

| 排名 | 类别 ID | 类别名称 | 置信度 |
|------|---------|----------|--------|
| 1 | 817 | streetcar | 15.9531 |
| 2 | 479 | jeep | 13.9531 |
| 3 | 511 | trolleybus | 13.4375 |
| 4 | 581 | tow truck | 12.7188 |
| 5 | 468 | cab | 12.6484 |

**结果**: ✅ 正确识别为交通工具

---

## 四、性能测试

**脚本**: `benchmark.py`
**测试条件**: 预热 10 次，正式测试 100 次

### 4.1 测试结果

| 指标 | 数值 |
|------|------|
| 平均延迟 | 3.37 ms |
| 最小延迟 | 3.33 ms |
| 最大延迟 | 3.42 ms |
| **吞吐量** | **297.13 FPS** |
| 延迟标准差 | 0.02 ms |
| P50 延迟 | 3.36 ms |
| P90 延迟 | 3.39 ms |
| P99 延迟 | 3.41 ms |

### 4.2 性能分析

- 延迟非常稳定 (标准差仅 0.02ms)
- 吞吐量接近 300 FPS，满足实时推理需求
- P99 延迟仅 3.41ms，99% 请求在 3.5ms 内完成

---

## 五、代码文件清单

```
CNN/
├── common/                    # 公用资源目录
│   ├── Pipeline.md            # 部署流程文档
│   ├── gen_aipp_cfg.py       # AIPP 参数计算工具
│   ├── test.jpg              # 测试图片 - 狗
│   ├── car.png               # 测试图片 - 车
│   └── acllite_utils/         # ACLLite 推理工具库
│       ├── __init__.py
│       ├── acllite_resource.py
│       ├── acllite_model.py
│       ├── acllite_image.py
│       ├── acllite_logger.py
│       ├── acllite_utils.py
│       └── constants.py
│
├── ResNet/                   # ResNet50 模型目录
│   ├── log.md               # 本实验日志
│   ├── export_onnx.py       # ONNX 导出脚本
│   ├── inference.py          # 推理测试脚本
│   ├── benchmark.py         # 性能基准测试脚本
│   ├── resnet50.onnx       # ONNX 模型 (98MB)
│   ├── resnet50_aipp.om    # 离线模型 (50MB)
│   └── aipp.cfg             # AIPP 配置文件
│
└── MobileNet/               # MobileNet 模型目录 (已完成)
```

---

## 六、遇到的问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| ATC 报错 `No module named 'decorator'` | Python 依赖缺失 | `pip install decorator attrs psutil jinja2 tornado cloudpickle scipy ml-dtypes` |
| ATC 报错 `Unknown enumeration value 'BGR888_U8'` | AIPP 配置格式错误 | 使用数字枚举值 `input_format: 3` |
| 推理报错 `resource.destroy() not found` | API 使用错误 | 改用 `del resource` 释放资源 |
| 分类置信度偏低 | AIPP mean_chn_1 计算错误 (117→116) | 修正参数并重新编译模型 |

---

## 七、结论

### 7.1 实验成功

- ✅ 成功在昇腾 310B1 NPU 上部署 ResNet50 模型
- ✅ AIPP 硬件加速预处理正常工作
- ✅ 推理结果正确，分类准确
- ✅ 性能优秀，达到 297 FPS 吞吐量

### 7.2 性能评估

| 指标 | 数值 | 评价 |
|------|------|------|
| 推理延迟 | 3.37 ms | 优秀 |
| 吞吐量 | 297 FPS | 优秀 |
| 稳定性 | 标准差 0.02ms | 非常稳定 |

### 7.3 可复现性

本项目所有代码和配置已保存在重组后的目录结构中，可通过以下步骤复现：

1. 进入 `CNN/ResNet/` 目录
2. 配置 Conda 环境并安装依赖
3. 运行 `python inference.py` 进行推理测试
4. 运行 `python benchmark.py` 进行性能测试

**目录结构说明**:
- `common/`: 公用资源 (ACLLite、测试图片、Pipeline文档)
- `ResNet/`: ResNet50 模型相关文件
- `MobileNet/`: MobileNet 模型相关文件 (已完成部署)

---

**实验完成时间**: 2026-05-21 08:38 (UTC)
**实验状态**: ✅ 成功
