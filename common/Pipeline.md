# 香橙派 AIpro (Ascend 310B1) CNN 高性能离线部署全流程指南

> ⚠️ **硬件型号说明**：香橙派 AIpro 可能搭载 **Ascend 310B1** 或 **Ascend 310B4** 芯片，请通过 `npu-smi info` 命令确认实际型号，并相应修改 `--soc_version` 参数。

本指南详细介绍了如何将一个标准的 CNN 模型（以 PyTorch 为例）转换为昇腾离线模型（`.om`），并通过 AIPP 硬件加速在香橙派 AIpro 上实现极速推理。

---

## 阶段一：开发端 - 导出静态 ONNX 模型

昇腾 ATC 编译器对静态 Shape 的图结构优化最为彻底。请在你的开发机（或配置好 PyTorch 的环境内）运行以下脚本，导出符合昇腾输入规范的 ONNX 模型。

```python
import torch
import torchvision.models as models

# 1. 初始化模型（以 ResNet50 为例）
model = models.resnet50(pretrained=True)
model.eval()

# 2. 构造严格的静态输入张量 (Batch_Size=1, Channels=3, Height=224, Width=224)
dummy_input = torch.randn(1, 3, 224, 224)

# 3. 导出为 ONNX 格式
torch.onnx.export(
    model,
    dummy_input,
    "resnet50.onnx",
    verbose=False,
    input_names=["input_image"],    # 指定清晰的输入节点名称，后续ATC转换需对应
    output_names=["output_tensor"], # 指定输出节点名称
    opset_version=11                # 推荐使用 opset 11 或 13
)
print("ONNX 模型导出成功：resnet50.onnx")
```

**验证步骤**：

```bash
# 安装 onnx 检查工具
pip install onnx

# 验证模型结构
python -c "import onnx; model = onnx.load('resnet50.onnx'); onnx.checker.check_model(model); print('ONNX 模型验证通过')"
```

---

## 阶段二：香橙派端 - 配置 AIPP 硬件预处理

AIPP (Artificial Intelligence Pre-Processing) 用于在 NPU 内部的硬件流水线上直接完成图像预处理（色域转换、减均值、归一化）。这能让推理彻底摆脱 CPU 预处理的性能瓶颈。

在香橙派上创建一个名为 `aipp.cfg` 的文件，填入以下标准 ImageNet 归一化参数配置：

```ini
aipp_op {
    aipp_mode: static
    input_format: 3                  # BGR888_U8 = 3
    src_image_size_w: 224           # 模型的输入宽度
    src_image_size_h: 224           # 模型的输入高度

    # 开启常见色域转换（若不需要转换可不配置，通常保持 BGR888_U8 输入即可）
    # 以下为针对 ImageNet 的 [0.485, 0.456, 0.406] 均值和 [0.229, 0.224, 0.225] 方差的硬件转换系数
    mean_chn_0: 104                 # B通道 均值
    mean_chn_1: 117                 # G通道 均值
    mean_chn_2: 124                 # R通道 均值
    min_chn_0: 0.0
    min_chn_1: 0.0
    min_chn_2: 0.0
    var_reci_chn_0: 0.01742919      # 对应 1 / (255 * 0.225)
    var_reci_chn_1: 0.01750700      # 对应 1 / (255 * 0.224)
    var_reci_chn_2: 0.01712475      # 对应 1 / (255 * 0.229)
}
```

**AIPP 归一化公式说明**：

```
pixel_out = (pixel_in - mean_chn_i - min_chn_i) * var_reci_chn_i
```

**参数计算方法**（以 ImageNet 为例）：

| 原始均值 (RGB) | 原始方差 (RGB) | AIPP 均值计算 | AIPP var_reci 计算 |
|---------------|---------------|---------------|-------------------|
| 0.485, 0.456, 0.406 | 0.225, 0.224, 0.229 | 255 × 均值 | 1 / (255 × 方差) |

> ⚠️ **注意**：如果你的模型训练时使用了不同的归一化参数，请根据上述公式重新计算 `mean_chn_i` 和 `var_reci_chn_i`。

---

## 阶段三：香橙派端 - 使用 ATC 编译离线模型 (.om)

### 1. 注入昇腾环境变量

在香橙派终端执行以下命令（建议写入 `~/.bashrc`）：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

> 注：若提示找不到路径，请在系统中检索该文件：`find / -name set_env.sh 2>/dev/null`

### 2. 确认芯片型号

```bash
npu-smi info
```

在输出的 "Name" 列确认芯片型号（如 **310B1** 或 **310B4**）。

### 3. 执行 ATC 转换命令

将导出的 `resnet50.onnx` 和上面的 `aipp.cfg` 放在同一目录下，执行编译：

```bash
atc --model=resnet50.onnx \
    --framework=5 \
    --output=resnet50_aipp \
    --soc_version=Ascend310B1 \
    --input_shape="input_image:1,3,224,224" \
    --insert_op_conf=aipp.cfg \
    --enable_small_channel=1
```

**参数说明**：

| 参数 | 说明 |
|------|------|
| `--model` | ONNX 模型文件路径 |
| `--framework=5` | 表示输入模型为 ONNX 格式 |
| `--output` | 输出的 OM 模型名称（不含扩展名） |
| `--soc_version=Ascend310B1` | 香橙派 AIpro 的核心芯片型号，**不可错写为 Ascend310** |
| `--input_shape` | 模型输入的静态 Shape |
| `--insert_op_conf` | AIPP 配置文件路径 |
| `--enable_small_channel=1` | 加速 CNN 首层 3 通道卷积的硬件算子优化 |

看到 `ATC run success!` 后，当前目录下会生成 `resnet50_aipp.om`。

**常见错误排查**：

| 错误信息 | 可能原因 | 解决方案 |
|----------|----------|----------|
| `Can not find op model file` | ONNX 文件路径错误 | 检查 `--model` 参数路径 |
| `Can not parse the model` | ONNX 模型格式问题 | 用 `onnx.checker` 验证模型 |
| `invalid soc_version` | 芯片型号错误 | 确认 `--soc_version=Ascend310B1` 或 `Ascend310B4` |
| `input shape mismatch` | Shape 定义不匹配 | 检查 `--input_shape` 与模型实际输入 |

---

## 阶段四：香橙派端 - 搭建轻量推理环境

> ⚠️ **论坛核心避坑指南**：
> 很多开发者在全新的 conda 环境中直接 `import acllite_model` 会遭遇报错。这是因为 AclLite 是华为在样例中提供的封装组件，并未注册进标准的 pip 源。
> 最干净、不破坏系统的解决方法：直接把官方的 acllite 核心源码文件下载到你的当前工作目录下作为本地依赖模块导入。

在你的推理项目目录下，执行以下命令下载配套的 AclLite 核心 Python 脚本：

```bash
# 创建一个本地方案包目录并进入
mkdir -p acllite_utils && cd acllite_utils

# 从华为官方镜像仓直接下载核心推理库文件
wget https://gitee.com/ascend/samples/raw/master/inference/acllite/Python/acllite_resource.py
wget https://gitee.com/ascend/samples/raw/master/inference/acllite/Python/acllite_model.py
wget https://gitee.com/ascend/samples/raw/master/inference/acllite/Python/acllite_logger.py
wget https://gitee.com/ascend/samples/raw/master/inference/acllite/Python/constants.py
wget https://gitee.com/ascend/samples/raw/master/inference/acllite/Python/utils.py

# 确认文件下载完整
ls -la

# 回到项目根目录
cd ..
```

> 💡 **提示**：如果 wget 下载失败，可以尝试使用浏览器直接访问 Gitee 页面，手动下载后通过 scp 上传到香橙派。

随后，在你的 conda 环境中安装基础依赖（OpenCV 和 NumPy）：

```bash
conda activate ascend_cnn
pip install numpy opencv-python
```

**验证 CANN 环境**：

```bash
# 检查 CANN 版本
python -c "import acl; print('ACL version:', acl.__version__)"

# 检查 Ascend 驱动
npu-smi info
```

---

## 阶段五：香橙派端 - 编写 Python 推理代码

在项目根目录下创建一个 `inference.py`，代码内部通过相对路径正确索引下载好的 acllite_utils。

```python
import os
import sys
import cv2
import numpy as np

# 将我们刚刚下载的本地 acllite_utils 路径添加进 Python 寻址空间
sys.path.append(os.path.join(os.path.dirname(__file__), 'acllite_utils'))

from acllite_resource import AclLiteResource
from acllite_model import AclLiteModel


def preprocess_image(image_path: str, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    图像预处理函数

    Args:
        image_path: 输入图片路径
        target_size: 目标尺寸 (width, height)

    Returns:
        预处理后的图像数据（UINT8格式）
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    # 缩放到模型所需的静态分辨率
    input_img = cv2.resize(img, target_size)

    # 【核心注意点】因为开启了 AIPP：
    # 此时模型输入节点接收的类型已经变为 UINT8（即 0-255 的整数型字节数据）。
    # 严禁在此处做 `/255.0` 或者是减均值操作，直接传入原始数据，
    # 否则 AIPP 会进行二次重复计算导致结果错误！
    return input_img.astype(np.uint8)


def main():
    """主推理函数"""
    model_path = "./resnet50_aipp.om"
    image_path = "test.jpg"

    # 1. 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"[错误] 未找到模型文件: {model_path}")
        return 1

    # 2. 初始化昇腾 NPU 计算资源
    try:
        resource = AclLiteResource()
        resource.init()
    except Exception as e:
        print(f"[错误] NPU 资源初始化失败: {e}")
        return 1

    # 3. 加载编译好的离线 .om 模型
    try:
        model = AclLiteModel(model_path)
    except Exception as e:
        print(f"[错误] 模型加载失败: {e}")
        resource.destroy()
        return 1

    # 4. 图像读取与预处理
    try:
        if os.path.exists(image_path):
            input_data = preprocess_image(image_path, (224, 224))
            print(f"[信息] 使用图片: {image_path}")
        else:
            print(f"[警告] 未找到测试图片 {image_path}，使用全零矩阵进行链路测试...")
            input_data = np.zeros((224, 224, 3), dtype=np.uint8)
    except Exception as e:
        print(f"[错误] 图像预处理失败: {e}")
        del model
        resource.destroy()
        return 1

    # 5. 送入 NPU 推理引擎
    # 输入必须是 list 形式，对应图的多输入节点
    try:
        outputs = model.execute([input_data])
    except Exception as e:
        print(f"[错误] 模型推理执行失败: {e}")
        del model
        resource.destroy()
        return 1

    # 6. 后处理与结果解析
    # outputs[0] 为标准分类网络输出的各类别 Logits 概率分布，形状为 (1, 1000)
    logits = outputs[0]
    predicted_id = np.argmax(logits, axis=1)[0]
    predicted_score = logits[0, predicted_id]

    print("\n" + "=" * 50)
    print("  昇腾 NPU 离线推理链路测试成功！")
    print("=" * 50)
    print(f"  输出特征图形状: {logits.shape}")
    print(f"  预测类别 ID:     {predicted_id}")
    print(f"  置信度:         {predicted_score:.4f}")
    print("=" * 50 + "\n")

    # 7. 释放模型与硬件资源
    del model
    resource.destroy()

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
```

运行代码：

```bash
python inference.py
```

**预期输出示例**：

```
[信息] 使用图片: test.jpg

==================================================
  昇腾 NPU 离线推理链路测试成功！
==================================================
  输出特征图形状: (1, 1000)
  预测类别 ID:     281
  置信度:         0.8472
==================================================
```

---

## 阶段六：性能与状态监控

### NPU 状态监控

在香橙派上运行推理脚本的同时，你可以打开另一个 ssh 终端窗口执行以下命令，来监控 NPU 核心的真实运行状态：

```bash
watch -n 0.5 npu-smi info
```

通过此命令，你可以实时观测到香橙派 AIpro 的：
- **AI Core %**：NPU 利用率
- **Memory**：显存占用情况
- **Power**：工作功耗
- **Temperature**：实时温度

### 推理性能测试

创建一个 `benchmark.py` 用于性能基准测试：

```python
import os
import sys
import time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'acllite_utils'))

from acllite_resource import AclLiteResource
from acllite_model import AclLiteModel


def benchmark(model_path: str, warmup: int = 10, iterations: int = 100):
    """推理性能基准测试"""
    resource = AclLiteResource()
    resource.init()
    model = AclLiteModel(model_path)

    # 准备输入数据（全零矩阵）
    input_data = np.zeros((224, 224, 3), dtype=np.uint8)

    # 预热阶段
    print(f"预热中 ({warmup} 次)...")
    for _ in range(warmup):
        model.execute([input_data])

    # 正式测试
    print(f"性能测试中 ({iterations} 次)...")
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        model.execute([input_data])
        elapsed = (time.perf_counter() - start) * 1000  # 转换为毫秒
        times.append(elapsed)

    # 统计结果
    times = np.array(times)
    print("\n" + "=" * 50)
    print("  性能基准测试结果")
    print("=" * 50)
    print(f"  平均延迟:     {times.mean():.2f} ms")
    print(f"  最小延迟:     {times.min():.2f} ms")
    print(f"  最大延迟:     {times.max():.2f} ms")
    print(f"  吞吐量:       {1000 / times.mean():.2f} FPS")
    print(f"  延迟标准差:   {times.std():.2f} ms")
    print("=" * 50)

    del model
    resource.destroy()


if __name__ == "__main__":
    benchmark("./resnet50_aipp.om")
```

运行基准测试：

```bash
python benchmark.py
```

---

## 阶段七：端到端验证与对比

为确保 AIPP 预处理与 PyTorch 原生推理结果一致，建议进行数值对比验证。

### PyTorch 原生推理脚本

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ImageNet 标准化参数
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def pytorch_inference(image_path: str) -> int:
    """PyTorch 原生推理"""
    model = models.resnet50(pretrained=True)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    img = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_id = output.argmax(dim=1).item()

    return predicted_id


if __name__ == "__main__":
    result = pytorch_inference("test.jpg")
    print(f"PyTorch 预测结果: {result}")
```

### 对比验证流程

1. 使用相同的 `test.jpg` 图片
2. 分别运行 PyTorch 和昇腾推理
3. 验证预测的类别 ID 是否一致（允许小量浮点误差）

> ⚠️ **注意**：由于昇腾和 PyTorch 的浮点运算精度可能存在微小差异，logits 数值可能略有不同，但预测的类别 ID 应该保持一致。

---

## 常见问题排查 (FAQ)

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| `ModuleNotFoundError: No module named 'acllite'` | 未正确配置 `sys.path` | 确认 `acllite_utils` 目录存在且包含所有必需文件 |
| `ACL error: 507051` | NPU 资源被占用 | 检查是否有其他进程占用 NPU，使用 `npu-smi info` 确认 |
| `ATC run failed` | ONNX 模型问题 | 使用 `onnx.checker.check_model()` 验证模型 |
| 推理结果全零 | AIPP 配置错误 | 检查 `aipp.cfg` 的 `input_format` 是否与实际输入匹配 |
| 类别预测错误 | 预处理不一致 | 确认 AIPP 参数与训练时的归一化参数完全一致 |
| 推理速度慢 | 未启用 AIPP | 确认 ATC 转换时已添加 `--insert_op_conf=aipp.cfg` |

---

## 项目目录结构

```
project/
├── resnet50.onnx              # 阶段一导出的 ONNX 模型
├── aipp.cfg                   # 阶段二创建的 AIPP 配置文件
├── resnet50_aipp.om           # 阶段三编译生成的离线模型
├── acllite_utils/              # 阶段四下载的推理工具库
│   ├── acllite_resource.py
│   ├── acllite_model.py
│   ├── acllite_logger.py
│   ├── constants.py
│   └── utils.py
├── inference.py               # 阶段五编写的推理脚本
├── benchmark.py               # 阶段六编写的性能测试脚本
├── pytorch_verify.py          # 阶段七 PyTorch 对比脚本（可选）
└── test.jpg                   # 测试图片
```

---

## 附录：香橙派 AIpro 硬件规格

| 项目 | 规格 |
|------|------|
| **NPU 芯片** | 昇腾 310B1 / 310B4 |
| **AI 算力** | 8 TOPS (INT8) / 4 TFLOPS (FP16) |
| **CPU** | 4 核 64 位 Arm 处理器 |
| **内存** | 8GB / 16GB LPDDR4X |
| **操作系统** | Ubuntu 22.04 / openEuler 22.03 |
| **CANN 版本** | 建议使用最新稳定版 |

---

## 参考资源

- [昇腾社区文档中心](https://www.hiascend.com/document/)
- [华为 CANN 开发文档](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/inferapplicationdev/atctool/atlasatc_16_0001.html)
- [香橙派 AIpro 官方资料](http://www.orangepi.cn/html/news/news-details/news28.html)
