# 香橙派 AIpro MobileNetV3-Small 部署实验日志

**实验日期**: 2026-05-21
**实验设备**: 香橙派 AIpro (Ascend 310B1)
**实验目的**: 在昇腾 NPU 上完成 MobileNetV3 模型离线部署与推理验证

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
| OpenCV | 4.13.0.92 |
| NumPy | 1.24.4 |
| ONNX | 1.17.0 |
| PyTorch | 2.4.1 (CPU) |
| torchvision | 0.19.1 |

---

## 二、实验步骤

### 2.1 ONNX 模型导出

**脚本**: `export_onnx.py`

```python
# 导出 MobileNetV3-Small 为 ONNX
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
torch.onnx.export(
    model,
    dummy_input,
    "mobilenetv3.onnx",
    input_names=["input_image"],
    output_names=["output_tensor"],
    opset_version=11
)
```

**结果**:
- 模型文件: `mobilenetv3.onnx` (9.8MB)
- 输入 Shape: (1, 3, 224, 224)
- 验证: ONNX 模型结构检查通过

### 2.2 AIPP 配置

**文件**: `aipp.cfg`

AIPP 使用与 ResNet50 相同的 ImageNet 归一化参数：

```
mean_chn_0 (B): 104
mean_chn_1 (G): 116
mean_chn_2 (R): 124

var_reci_chn_0: 0.01742919
var_reci_chn_1: 0.01750700
var_reci_chn_2: 0.01712475
```

### 2.3 ATC 模型编译

**编译命令**:
```bash
atc --model=mobilenetv3.onnx \
    --framework=5 \
    --output=mobilenetv3_aipp \
    --soc_version=Ascend310B1 \
    --input_shape="input_image:1,3,224,224" \
    --insert_op_conf=aipp.cfg \
    --enable_small_channel=1
```

**结果**:
- 输出文件: `mobilenetv3_aipp.om` (7.4MB)
- 编译时间: ~5 分钟
- 状态: ✅ ATC run success

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
| 1 | 258 | Maltese dog | 14.4766 |
| 2 | 259 | Samoyed | 12.1953 |
| 3 | 279 | - | 9.4609 |
| 4 | 270 | - | 7.9883 |
| 5 | 248 | - | 7.9492 |

**结果**: ✅ 正确识别为犬类 (Top-1 为马尔济斯犬)

#### car.png (车)

| 排名 | 类别 ID | 类别名称 | 置信度 |
|------|---------|----------|--------|
| 1 | 468 | cab | 12.1875 |
| 2 | 817 | streetcar | 11.3438 |
| 3 | 479 | jeep | 10.3281 |
| 4 | 581 | tow truck | 10.1250 |
| 5 | 436 | - | 9.6484 |

**结果**: ✅ 正确识别为交通工具 (Top-1 为出租车)

---

## 四、性能测试

**脚本**: `benchmark.py`
**测试条件**: 预热 10 次，正式测试 100 次

### 4.1 测试结果

| 指标 | 数值 |
|------|------|
| 平均延迟 | 0.96 ms |
| 最小延迟 | 0.93 ms |
| 最大延迟 | 1.01 ms |
| **吞吐量** | **1047.12 FPS** |
| 延迟标准差 | 0.01 ms |
| P50 延迟 | 0.95 ms |
| P90 延迟 | 0.97 ms |
| P99 延迟 | 0.99 ms |

### 4.2 性能分析

- 延迟极低 (平均 < 1ms)
- 吞吐量超过 1000 FPS，性能优异
- 延迟极其稳定 (标准差仅 0.01ms)
- P99 延迟 < 1ms，满足实时推理需求

---

## 五、与 ResNet50 性能对比

| 指标 | ResNet50 | MobileNetV3-Small | 提升 |
|------|----------|-------------------|------|
| 模型大小 | 50 MB | 7.4 MB | 6.8x 更小 |
| 平均延迟 | 3.37 ms | 0.96 ms | 3.5x 更快 |
| 吞吐量 | 297 FPS | 1047 FPS | 3.5x 更高 |
| P99 延迟 | 3.41 ms | 0.99 ms | 3.4x 更低 |

**结论**: MobileNetV3-Small 在性能上全面优于 ResNet50，尤其适合移动端/边缘设备部署

---

## 六、代码文件清单

```
MobileNet/
├── export_onnx.py       # ONNX 导出脚本
├── inference.py          # 推理测试脚本
├── benchmark.py         # 性能基准测试脚本
├── build.sh            # ATC 编译脚本
├── aipp.cfg            # AIPP 配置文件
├── log.md              # 本实验日志
├── mobilenetv3.onnx   # ONNX 模型 (9.8MB)
└── mobilenetv3_aipp.om # 离线模型 (7.4MB)
```

---

## 七、结论

### 7.1 实验成功

- ✅ 成功在昇腾 310B1 NPU 上部署 MobileNetV3-Small 模型
- ✅ AIPP 硬件加速预处理正常工作
- ✅ 推理结果正确，分类准确
- ✅ 性能卓越，达到 1047 FPS 吞吐量

### 7.2 性能评估

| 指标 | 数值 | 评价 |
|------|------|------|
| 推理延迟 | 0.96 ms | 极佳 |
| 吞吐量 | 1047 FPS | 极佳 |
| 稳定性 | 标准差 0.01ms | 非常稳定 |
| 模型大小 | 7.4 MB | 轻量 |

### 7.3 可复现性

1. 进入 `CNN/MobileNet/` 目录
2. 配置 Conda 环境并安装依赖
3. 运行 `python export_onnx.py` 导出模型 (如需)
4. 运行 `python inference.py` 进行推理测试
5. 运行 `python benchmark.py` 进行性能测试

---

**实验完成时间**: 2026-05-21 09:13 (UTC)
**实验状态**: ✅ 成功
