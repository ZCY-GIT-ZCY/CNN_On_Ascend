# 算子性能分析指南

**文档目标**：全面介绍算子性能指标、监控工具和分析方法，帮助开发者定位和解决性能瓶颈。

**适用场景**：昇腾 NPU (Ascend 310/310B1/910) + CANN 环境

---

## 目录

1. [性能指标体系](#1-性能指标体系)
2. [CANN 内置分析工具](#2-cann-内置分析工具)
3. [msprof 性能分析器](#3-msprof-性能分析器)
4. [msFmk-sys 模块详解](#4-msfmk-sys-模块详解)
5. [算子分析工具](#5-算子分析工具)
6. [实践案例](#6-实践案例)
7. [常见性能问题与优化](#7-常见性能问题与优化)

---

## 1. 性能指标体系

### 1.1 核心性能指标

```
┌─────────────────────────────────────────────────────────────────────┐
│                      性能指标层次                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  应用层指标                                                         │
│  ├── 吞吐量 (Throughput)      单位: FPS / samples/sec             │
│  ├── 延迟 (Latency)           单位: ms                           │
│  └── 首帧耗时 (First Frame)   单位: ms                           │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  算子层指标                                                         │
│  ├── 算子执行时间           单位: μs / ms                        │
│  ├── 算子调用次数           单位: 次                               │
│  ├── 算子吞吐量             单位: GFLOPS / GOPS                  │
│  └── 算子融合率             单位: %                               │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  资源层指标                                                         │
│  ├── AI Core 利用率         单位: %                               │
│  ├── Cube 利用率            单位: %                               │
│  ├── Vector 利用率          单位: %                               │
│  ├── 带宽利用率            单位: %                               │
│  └── 内存占用              单位: MB / GB                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 指标详细说明

#### 吞吐量 (Throughput)

```python
"""
吞吐量定义：单位时间内处理的数据量
"""

# 计算公式
throughput = batch_size / inference_time  # samples/sec

# 或
throughput = 1000 / avg_latency  # FPS (当 batch_size=1 时)

# 实际测量示例
import time

batch_size = 1
iterations = 100
total_time = 0

for _ in range(iterations):
    start = time.perf_counter()
    model.execute([input_data])
    end = time.perf_counter()
    total_time += (end - start)

avg_latency = total_time / iterations * 1000  # ms
throughput = 1000 / avg_latency  # FPS

print(f"平均延迟: {avg_latency:.2f} ms")
print(f"吞吐量: {throughput:.2f} FPS")
```

#### 延迟 (Latency)

```python
"""
延迟定义：从输入到输出的端到端时间
"""

# 延迟分解
latency = preprocess_time + inference_time + postprocess_time

# 各阶段延迟占比分析
total_latency = 10.5  # ms
print(f"""
延迟分解:
├── 预处理: {preprocess_time:.2f} ms ({preprocess_time/total_latency*100:.1f}%)
├── 推理:   {inference_time:.2f} ms ({inference_time/total_latency*100:.1f}%)  ← 通常最关注
└── 后处理: {postprocess_time:.2f} ms ({postprocess_time/total_latency*100:.1f}%)
""")
```

#### 算子级指标

| 指标 | 公式 | 说明 |
|------|------|------|
| **算子执行时间** | end - start | 单次执行耗时 |
| **算子调用次数** | 计数器 | 总共调用了多少次 |
| **算子占比** | op_time / total_time × 100% | 占总时间比例 |
| **融合率** | fused_ops / total_ops × 100% | 融合算子占比 |
| **GFLOPS** | flops / (time × 10^9) | 浮点运算性能 |
| **算子效率** | actual_flops / peak_flops × 100% | 接近理论峰值程度 |

### 1.3 资源利用指标

```
┌─────────────────────────────────────────────────────────────────────┐
│                    昇腾硬件资源架构                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│                      ┌─────────────────────┐                        │
│                      │      AI Core        │                        │
│                      │                     │                        │
│                      │  ┌───────────────┐ │                        │
│                      │  │   Cube Unit    │ │  ← 矩阵乘法 (16³)   │
│                      │  │   (矩阵计算)   │ │    利用率指标       │
│                      │  └───────────────┘ │                        │
│                      │                     │                        │
│                      │  ┌───────────────┐ │                        │
│                      │  │  Vector Unit  │ │  ← 逐元素运算       │
│                      │  │  (向量计算)   │ │    利用率指标       │
│                      │  └───────────────┘ │                        │
│                      │                     │                        │
│                      │  ┌───────────────┐ │                        │
│                      │  │ Scalar Unit   │ │  ← 控制逻辑         │
│                      │  └───────────────┘ │                        │
│                      │                     │                        │
│                      └─────────────────────┘                        │
│                                    │                                │
│                      ┌─────────────┴─────────────┐                 │
│                      │                           │                    │
│                      ▼                           ▼                   │
│              ┌──────────────┐           ┌──────────────┐           │
│              │  L1 Cache   │           │  L2 Cache    │           │
│              │   (400KB)   │           │   (2MB)     │           │
│              └──────────────┘           └──────────────┘           │
│                                    │                                │
│                                    ▼                                │
│                            ┌──────────────┐                        │
│                            │     DDR      │                        │
│                            │   (8GB)     │                        │
│                            └──────────────┘                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### 利用率计算

```python
"""
资源利用率计算
"""

# AI Core 利用率
ai_core_util = (ai_core_active_time / total_time) * 100

# Cube 利用率 (矩阵乘法专用单元)
cube_util = (cube_active_time / total_time) * 100

# Vector 利用率 (向量运算单元)
vector_util = (vector_active_time / total_time) * 100

# 带宽利用率
bandwidth_util = (memory_access_bytes / (memory_bandwidth * time)) * 100

# 示例输出
print(f"""
资源利用分析:
├── AI Core 利用率: {ai_core_util:.1f}%
├── Cube 利用率:    {cube_util:.1f}%  ← 矩阵运算效率
├── Vector 利用率:  {vector_util:.1f}%  ← 逐元素运算效率
└── 带宽利用率:    {bandwidth_util:.1f}%
""")
```

---

## 2. CANN 内置分析工具

### 2.1 工具全景图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CANN 性能分析工具生态                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  命令行工具                                                         │
│  ├── msFmk-sys    ← 框架级性能数据采集                            │
│  ├── msprof        ← 综合性能分析器                                │
│  ├── atlasdk       ← 芯片级诊断工具                               │
│  └── adb-ssh       ← 远程调试工具                                 │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  编程接口                                                           │
│  ├── msFmk-sys API  ← Python/C++ 集成                            │
│  ├── ACL API        ← 算子级数据采集                              │
│  └──GE API          ← 图优化数据                                   │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  可视化工具                                                         │
│  ├── MindX Studio  ← 图形化分析界面                               │
│  └── HCCS View     ← 芯片互联分析                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 工具对比

| 工具 | 用途 | 采集粒度 | 输出格式 | 学习曲线 |
|------|------|---------|---------|---------|
| **msFmk-sys** | 框架级性能分析 | 函数/算子 | JSON/TXT | 低 |
| **msprof** | 综合性能分析 | 算子+资源 | HTML/JSON | 中 |
| **atlasdk** | 芯片诊断 | 系统级 | 日志 | 低 |
| **ACL API** | 自定义采集 | 可控 | 自定义 | 高 |

---

## 3. msprof 性能分析器

### 3.1 msprof 概述

msprof 是 CANN 提供的综合性能分析工具，可采集 GPU/NPU 上的性能数据。

```
┌─────────────────────────────────────────────────────────────────────┐
│                          msprof 工作流程                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. 编译时插桩 (可选)                                              │
│     atc 编译时添加 --enable-prof 开关                              │
│                                                                     │
│  2. 运行时代码采集                                                  │
│     设置环境变量或 API 调用                                         │
│                                                                     │
│  3. 生成性能数据                                                    │
│     msprof 生成 .pb 或 .json 文件                                  │
│                                                                     │
│  4. 分析报告生成                                                    │
│     msprof --analyze 生成 HTML 报告                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 常用命令

```bash
# ====================
# 基本使用命令
# ====================

# 采集性能数据
msprof --output=./prof_output \
       --application=python inference.py \
       --sys-level=2

# 分析已采集的数据
msprof --analyze ./prof_output/*.pb

# 生成报告
msprof --export=html ./prof_output/profiler.json

# ====================
# 常用参数
# ====================

# --output: 输出目录
# --application: 待分析应用
# --sys-level: 系统级别 (1=粗, 2=细, 3=最细)
# --profiling: 采集类型 (acl/framework/all)
# --parallel: 并行采集
# --analyze: 分析模式
# --export: 导出格式 (html/json/csv)
```

### 3.3 环境变量配置

```bash
# ====================
# 性能采集环境变量
# ====================

# 启用性能采集
export ASCEND_PROFILER_ENABLE=1

# 输出目录
export ASCEND_PROFILER_LOG_DIR=./profiler_output

# 采集模式
export ASCEND_PROFILER_MODE=acl  # acl | framework | all

# 采集范围
export ASCEND_PROFILER_OPTIONS="{"enable_options":"acl"}"

# 算子级采集
export ASCEND_PROFILER_COLLECT_OPERATOR_DATA=1

# Trace 级采集
export ASCEND_PROFILER_COLLECT_TRACE_DATA=1

# 内存采集
export ASCEND_PROFILER_COLLECT_MEMORY_DATA=1
```

### 3.4 Python 集成示例

```python
"""
使用 msprof 进行性能采集
"""

# 方法1: 环境变量方式
# 在运行前设置环境变量，然后执行程序
"""
export ASCEND_PROFILER_ENABLE=1
export ASCEND_PROFILER_LOG_DIR=./profiler_output
python inference.py
"""

# 方法2: API 方式 (推荐)
import os
from msprof import Profiler

# 配置采集器
profiler = Profiler(
    enable=True,
    output_dir="./profiler_output",
    profile_mode="acl"  # acl | framework
)

# 开始采集
profiler.start()

# 执行推理
for i in range(100):
    model.execute([input_data])

# 结束采集
profiler.stop()

# 导出分析报告
profiler.analyze()
```

### 3.5 输出报告解读

```python
"""
msprof 输出报告结构
"""

profiler_output/
├── operator_statistics.json    # 算子统计
├── framework_statistics.json   # 框架统计
├── timeline_trace.json         # 时间线数据
├── memory_statistics.json      # 内存统计
└── summary.html               # 可视化报告
```

#### 算子统计文件示例

```json
{
  "operators": [
    {
      "name": "Conv2d",
      "type": "Convolution",
      "call_count": 54,
      "total_time_us": 125000,
      "avg_time_us": 2314,
      "min_time_us": 2100,
      "max_time_us": 2800,
      "占比": "45.2%"
    },
    {
      "name": "Relu",
      "type": "Activation",
      "call_count": 27,
      "total_time_us": 3200,
      "avg_time_us": 118,
      "占比": "1.2%"
    }
  ],
  "summary": {
    "total_time_us": 276500,
    "operator_count": 81,
    "fusion_count": 23
  }
}
```

---

## 4. msFmk-sys 模块详解

### 4.1 模块概述

msFmk-sys 是 MindSpore 框架提供的性能分析 Python 模块，用于采集神经网络训练/推理的性能数据。

```
┌─────────────────────────────────────────────────────────────────────┐
│                       msFmk-sys 采集架构                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│                        Python 代码                                   │
│                            │                                        │
│                            ▼                                        │
│                    ┌────────────────┐                              │
│                    │   msFmk-sys    │                              │
│                    │   Python API   │                              │
│                    └────────┬───────┘                              │
│                             │                                        │
│                             ▼                                        │
│                    ┌────────────────┐                              │
│                    │  C++ 采集器   │                              │
│                    │  (低开销)     │                              │
│                    └────────┬───────┘                              │
│                             │                                        │
│                             ▼                                        │
│                    ┌────────────────┐                              │
│                    │  性能数据文件  │                              │
│                    │  (.json)      │                              │
│                    └────────────────┘                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 安装与配置

```bash
# ====================
# 安装 msFmk-sys
# ====================

# 通过 pip 安装
pip install msfmk-sys

# 或通过 CANN 安装
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
pip install ${ASCEND_TOOLKIT_HOME}/python/site-packages/msfmk_sys*.whl

# ====================
# 验证安装
# ====================
python -c "from msFmk_sys import Profiler; print('msFmk-sys OK')"
```

### 4.3 基础使用

```python
"""
msFmk-sys 基础使用示例
"""

# 方式1: 上下文管理器 (推荐)
from msFmk_sys import Profiler

# 推理代码
import mindspore as ms

# 创建模型
network = CustomNetwork()
model = ms.Model(network)

# 性能采集
with Profiler(profiler_path="./profiler_output", 
              l2_cache_enable=True) as prof:
    
    # 执行推理
    for i in range(100):
        output = model.predict(input_data)
    
    # 训练时使用 train()
    # model.train(epoch, dataset)

# 采集完成，自动生成报告
print(f"性能数据已保存到: ./profiler_output")
```

```python
# 方式2: 手动控制
from msFmk_sys import Profiler

# 初始化
prof = Profiler(
    profiler_path="./profiler_output",
    l2_cache_enable=True,
    data_dump_enabled=True
)

# 开始采集
prof.start()

# 执行推理
for i in range(100):
    output = model.predict(input_data)

# 结束采集
prof.stop()

# 分析数据
prof.analyze()
```

### 4.4 高级配置

```python
"""
msFmk-sys 高级配置
"""

from msFmk_sys import Profiler

# 完整配置选项
profiler = Profiler(
    # 输出路径
    profiler_path="./profiler_output",
    
    # L2 Cache 统计
    l2_cache_enable=True,
    
    # 数据导出
    data_dump_enabled=True,
    
    # 采集模式
    profile_mode="Framework",  # Framework | ACL | All
    
    # 算子统计
    profile_single_op=True,
    op_time_statistic=True,
    
    # 内存统计
    profile_memory=True,
    
    # 训练轨迹
    trace_mode="train",  # train | predict
)

# 启动
profiler.start()

# 采集后分析
profiler.analyze(
    analyze_mode="timeline",  # timeline | summary
    export_mode="text"        # text | json | html
)
```

### 4.5 输出数据分析

```python
"""
读取 msFmk-sys 输出数据
"""

import json
import os

profiler_dir = "./profiler_output"

# 读取算子统计
with open(os.path.join(profiler_dir, "operator_statistics.json"), "r") as f:
    op_stats = json.load(f)

# 读取框架统计
with open(os.path.join(profiler_dir, "framework_statistics.json"), "r") as f:
    fw_stats = json.load(f)

# 分析输出
print("=" * 60)
print("                    算子性能分析报告")
print("=" * 60)

# 按耗时排序
sorted_ops = sorted(op_stats["operators"], 
                    key=lambda x: x["total_time_us"], 
                    reverse=True)

print(f"\n{'排名':<4} {'算子名称':<20} {'调用次数':<10} {'总时间(μs)':<15} {'占比':<8}")
print("-" * 60)

total_time = sum(op["total_time_us"] for op in sorted_ops)
cumulative = 0

for i, op in enumerate(sorted_ops[:10]):  # Top 10
    cumulative += op["total_time_us"]
    percentage = cumulative / total_time * 100
    print(f"{i+1:<4} {op['name']:<20} {op['call_count']:<10} "
          f"{op['total_time_us']:<15} {percentage:.1f}%")

print("\n" + "=" * 60)
print(f"总耗时: {total_time/1000:.2f} ms")
print(f"算子总数: {len(sorted_ops)}")
print(f"融合算子: {op_stats['summary']['fusion_count']}")
print("=" * 60)
```

---

## 5. 算子分析工具

### 5.1 ACL 算子分析 API

```python
"""
使用 ACL API 进行算子级分析
"""

import acl

# ====================
# 初始化 ACL
# ====================
acl.init()
device_id = 0
acl.set_device(device_id)

# ====================
# 配置算子分析
# ====================

# 创建分析配置
prof_config = acl.prof.create_config()
acl.prof.set_prof_module_level(acl.PROF_MODULE_ACL, acl.PROF_LEVEL_1)

# 启动分析
acl.prof.start()

# ====================
# 执行推理
# ====================
for i in range(100):
    model.execute([input_data])

# ====================
# 停止分析
# ====================
acl.prof.stop()

# 导出数据
acl.prof.save_data("./acl_profiler_output")

# 销毁资源
acl.prof.destroy_config()
acl.reset_device(device_id)
acl.finalize()
```

### 5.2 自定义算子计时器

```python
"""
自定义算子性能测量工具
"""

import time
from contextlib import contextmanager
from typing import Dict, List
from dataclasses import dataclass, field

@dataclass
class OpTimer:
    """算子计时器"""
    name: str
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    
    def add(self, elapsed: float):
        self.count += 1
        self.total_time += elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)
    
    @property
    def avg_time(self) -> float:
        return self.total_time / self.count if self.count > 0 else 0


class Profiler:
    """简单性能分析器"""
    
    def __init__(self):
        self.timers: Dict[str, OpTimer] = {}
        self.enabled = True
    
    @contextmanager
    def timer(self, name: str):
        """计时上下文管理器"""
        if not self.enabled:
            yield
            return
        
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000  # ms
            if name not in self.timers:
                self.timers[name] = OpTimer(name)
            self.timers[name].add(elapsed)
    
    def report(self) -> str:
        """生成性能报告"""
        if not self.timers:
            return "No profiling data"
        
        total = sum(t.total_time for t in self.timers.values())
        
        lines = [
            "=" * 70,
            "性能分析报告",
            "=" * 70,
            f"{'算子名称':<25} {'调用次数':<10} {'总时间(ms)':<15} {'平均(ms)':<12} {'占比':<8}",
            "-" * 70
        ]
        
        sorted_timers = sorted(self.timers.values(), 
                              key=lambda x: x.total_time, 
                              reverse=True)
        
        for t in sorted_timers:
            pct = t.total_time / total * 100
            lines.append(
                f"{t.name:<25} {t.count:<10} {t.total_time:<15.3f} "
                f"{t.avg_time:<12.3f} {pct:>6.1f}%"
            )
        
        lines.extend([
            "-" * 70,
            f"{'总计':<25} {'':<10} {total:<15.3f}",
            "=" * 70
        ])
        
        return "\n".join(lines)


# ====================
# 使用示例
# ====================
profiler = Profiler()

with profiler.timer("Conv2d_1"):
    # 执行卷积
    output = conv2d(input_data)

with profiler.timer("BatchNorm"):
    # 执行 BN
    output = batch_norm(output)

with profiler.timer("ReLU"):
    # 执行激活
    output = relu(output)

print(profiler.report())
```

### 5.3 NPU 工具链

```bash
# ====================
# 常用 NPU 诊断命令
# ====================

# 查看 NPU 状态
npu-smi

# 查看设备列表
npu-smi info

# 查看算力
npu-smi info -l

# 查看进程
npu-smi ps

# 监控利用率
npu-smi dmon -s utilization.gpu,utilization.memory -d 1

# 性能采样
npu-smi set -m 1 -t utilization -d 0

# ====================
# 芯片诊断
# ====================

# 运行芯片测试
atlasdk --check

# 查看错误日志
dmesg | grep -i ascend

# 性能计数器
perf stat -a -e ascend_cube// -e ascend_vector// python inference.py
```

---

## 6. 实践案例

### 6.1 案例：分析 MobileNetV3 性能瓶颈

```python
"""
MobileNetV3 性能分析脚本
"""

import os
import sys
import time
import json

# 添加路径
sys.path.insert(0, "../common/acllite_utils")

from msFmk_sys import Profiler

# 导入模型相关
import mindspore as ms
from mindspore import nn

# 创建 MobileNetV3 模型
def create_mobilenet():
    """创建 MobileNetV3 模型"""
    from mindspore.vision import mobilenet_v3_small
    network = mobilenet_v3_small(num_classes=1000)
    return network


def benchmark_model(model, input_data, iterations=100):
    """性能基准测试"""
    
    warmup = 10
    for _ in range(warmup):
        _ = model.predict(input_data)
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = model.predict(input_data)
        times.append(time.perf_counter() - start)
    
    times = [t * 1000 for t in times]  # 转换为 ms
    
    return {
        "avg": sum(times) / len(times),
        "min": min(times),
        "max": max(times),
        "p50": sorted(times)[len(times)//2],
        "p90": sorted(times)[int(len(times)*0.9)],
        "p99": sorted(times)[int(len(times)*0.99)],
        "fps": 1000 / (sum(times) / len(times))
    }


def main():
    # 1. 准备数据
    input_data = ms.Tensor.randn(1, 3, 224, 224, ms.float32)
    
    # 2. 创建模型
    print("创建 MobileNetV3 模型...")
    model = create_mobilenet()
    
    # 3. 性能采集
    print("开始性能采集...")
    profiler_output = "./mobilenet_profiler"
    os.makedirs(profiler_output, exist_ok=True)
    
    with Profiler(profiler_output, profile_mode="Framework") as prof:
        # 执行推理
        for i in range(100):
            _ = model.predict(input_data)
    
    # 4. 基准测试
    print("执行基准测试...")
    results = benchmark_model(model, input_data, iterations=100)
    
    # 5. 输出结果
    print("\n" + "=" * 60)
    print("MobileNetV3 性能报告")
    print("=" * 60)
    print(f"平均延迟:  {results['avg']:.2f} ms")
    print(f"最小延迟:  {results['min']:.2f} ms")
    print(f"最大延迟:  {results['max']:.2f} ms")
    print(f"P50 延迟:  {results['p50']:.2f} ms")
    print(f"P90 延迟:  {results['p90']:.2f} ms")
    print(f"P99 延迟:  {results['p99']:.2f} ms")
    print(f"吞吐量:    {results['fps']:.2f} FPS")
    print("=" * 60)
    
    # 6. 分析报告
    print(f"\n详细报告已保存到: {profiler_output}")


if __name__ == "__main__":
    main()
```

### 6.2 案例：分析 ResNet50 算子瓶颈

```python
"""
ResNet50 算子瓶颈分析
"""

import json
import os

def analyze_resnet_profiling():
    """分析 ResNet50 的性能瓶颈"""
    
    profiler_dir = "./resnet_profiler"
    
    # 读取算子数据
    if not os.path.exists(profiler_dir):
        print("请先运行性能采集")
        return
    
    op_stats_file = os.path.join(profiler_dir, "operator_statistics.json")
    if not os.path.exists(op_stats_file):
        print("算子统计数据不存在")
        return
    
    with open(op_stats_file, "r") as f:
        data = json.load(f)
    
    # 分析瓶颈
    operators = data.get("operators", [])
    sorted_ops = sorted(operators, key=lambda x: x.get("total_time_us", 0), reverse=True)
    
    print("\n" + "=" * 70)
    print("ResNet50 算子性能分析")
    print("=" * 70)
    
    total_time = sum(op.get("total_time_us", 0) for op in sorted_ops)
    print(f"总算子执行时间: {total_time/1000:.2f} ms")
    print(f"算子总数: {len(sorted_ops)}")
    print()
    
    # Top 10 耗时算子
    print("Top 10 耗时算子:")
    print("-" * 70)
    print(f"{'排名':<4} {'算子名称':<30} {'时间(ms)':<12} {'占比':<10} {'调用次数':<10}")
    print("-" * 70)
    
    cumulative = 0
    for i, op in enumerate(sorted_ops[:10]):
        time_ms = op.get("total_time_us", 0) / 1000
        cumulative += op.get("total_time_us", 0)
        pct = cumulative / total_time * 100
        count = op.get("call_count", 0)
        name = op.get("name", "Unknown")
        
        print(f"{i+1:<4} {name:<30} {time_ms:<12.2f} {pct:>8.1f}% {count:<10}")
        
        # 标注瓶颈
        if i < 3:
            print(f"     └─ ⚠️ 瓶颈算子 #{i+1}")
    
    print("-" * 70)
    
    # 融合分析
    fusion_count = data.get("summary", {}).get("fusion_count", 0)
    total_count = data.get("summary", {}).get("operator_count", 0)
    fusion_rate = fusion_count / total_count * 100 if total_count > 0 else 0
    
    print(f"\n融合统计:")
    print(f"  融合算子数: {fusion_count}")
    print(f"  原始算子数: {total_count}")
    print(f"  融合率: {fusion_rate:.1f}%")
    
    # 优化建议
    print("\n优化建议:")
    for i, op in enumerate(sorted_ops[:3]):
        name = op.get("name", "")
        if "Conv" in name:
            print(f"  • {name}: 考虑融合相邻的 BN/ReLU 算子")
        elif "MatMul" in name or "GEMM" in name:
            print(f"  • {name}: 考虑使用更高效的矩阵乘法参数")
        elif "Reduce" in name:
            print(f"  • {name}: 考虑融合到上游算子")
    
    print("=" * 70)


if __name__ == "__main__":
    analyze_resnet_profiling()
```

---

## 7. 常见性能问题与优化

### 7.1 问题诊断表

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| **AI Core 利用率低** | 计算密集度不足 | 增加算子融合 |
| **Cube 利用率低** | 数据布局不优 | 调整数据格式 (NC1HWC0) |
| **带宽利用率高** | 内存访问瓶颈 | 优化 tiling，减少 DDR 访问 |
| **特定算子耗时高** | 未融合 | 检查融合配置 |
| **延迟波动大** | 动态 shape | 固定 batch size |
| **首帧耗时长** | 图编译 | 预热/缓存 |

### 7.2 优化流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                       性能优化流程                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Step 1: 基准测试                                                   │
│  ──────────────                                                    │
│  建立性能基线，记录各项指标                                          │
│                                                                     │
│           │                                                        │
│           ▼                                                        │
│  Step 2: 瓶颈定位                                                   │
│  ──────────────                                                    │
│  使用 msprof/msFmk-sys 分析瓶颈算子                                │
│                                                                     │
│           │                                                        │
│           ▼                                                        │
│  Step 3: 针对性优化                                                 │
│  ───────────────────                                               │
│  ├── 算子融合                                                      │
│  ├── 调度优化                                                      │
│  ├── 内存布局调整                                                  │
│  └── 数据预取                                                      │
│                                                                     │
│           │                                                        │
│           ▼                                                        │
│  Step 4: 效果验证                                                   │
│  ──────────────                                                    │
│  重新测试，对比基线                                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.3 快速检查清单

```bash
# ====================
# 快速性能检查命令
# ====================

# 1. 检查 NPU 状态
npu-smi info

# 2. 检查进程
npu-smi ps

# 3. 实时监控
watch -n 1 npu-smi dmon

# 4. 检查 CANN 版本
python -c "import acl; print(acl.get_version())"

# 5. 验证环境
python -c "from msFmk_sys import Profiler; print('OK')"
```

---

## 附录：常用命令速查

| 任务 | 命令 |
|------|------|
| 性能采集 | `ASCEND_PROFILER_ENABLE=1 python inference.py` |
| msprof 分析 | `msprof --analyze ./profiler_output` |
| 导出报告 | `msprof --export=html ./profiler_output` |
| NPU 状态 | `npu-smi info` |
| 监控利用率 | `npu-smi dmon -s utilization.gpu -d 1` |
| msFmk-sys 采集 | `from msFmk_sys import Profiler` |

---

**文档版本**: v1.0  
**更新日期**: 2026-05-28  
**适用环境**: CANN 8.0+, Ascend 310/310B1/910
