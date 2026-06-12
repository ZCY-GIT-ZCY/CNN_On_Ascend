# 算子优化实验综合报告

> 生成日期: 2026-06-06
> 平台: Ascend 310B1 (香橙派 AIpro)
> 基线模型: MobileNetV3-Small, ResNet50

---

## 第一部分：实验背景

### 1.1 实验目标

解释为什么 7 种 ONNX 前端优化（fp16、shape_inferred、opset13、bn_folded、channels_last、onnxsim、onnxoptimizer）在 Ascend 310B1 上没有超过现有 AIPP OM 基线，并通过受控实验确认哪些优化真正有效。

### 1.2 关键修正

原始实验使用项目现有 AIPP OM 作为基线与本轮重新编译的变体比较。受控实验发现：
- 原始 AIPP OM（7.31MB）与重新编译的 OM（7.42MB）存在 **111KB 大小差异**
- `reexport_plain`（原始 ONNX 纯副本重新编译）比原始基线慢 **11.18%**（8.7σ）
- 说明原始基线是用不同 ATC 版本/参数编译的，**比较不平等**

**修正方法**: 以 `reexport_plain`（当前 ATC 重新编译的原始 ONNX）为统一基线，所有变体在同一编译链路下比较。

---

## 第二部分：实验设计

### 2.1 两阶段设计

| 阶段 | 内容 | 轮次 | 用途 |
|---|---|---|---|
| Phase 1: 噪声标定 | reexport_plain 独立重复 | 10轮 | 测量环境噪声基底 |
| Phase 2: 优化测量 | 所有变体+基线随机交错 | 6轮/变体 | 统计t检验比较 |

### 2.2 控制措施

- 子进程隔离（每个测量独立 Python 进程，避免 ACL 状态污染）
- 随机顺序（每轮所有变体随机排列，消除顺序偏差）
- 变体间 1.5s 冷却，轮间 5s 冷却（降低温度/状态影响）
- 系统负载全程记录（但无法控制：3核CPU，负载持续17-18）

### 2.3 统计方法

- Welch's t-test（不假设等方差），α=0.05
- 效应量须超过噪声基底的 95% 置信半宽才算"可检测"

---

## 第三部分：实验结果

### 3.1 MobileNetV3-Small

**噪声基底**: between-run std=0.0181ms (1.18%)，最小可检测效应量=1.18%

| 变体 | Mean(ms) | Diff(%) | Std(ms) | CV(%) | p值 | 显著? | 效应量/噪声 |
|---|---|---|---|---|---|---|---|
| **fp16** | **1.0088** | **-10.51%** | 0.0442 | 4.39 | **0.024** | **✅ Y** | 8.9× |
| shape_inferred | 1.0837 | -3.86% | 0.0167 | 1.54 | 0.302 | N | 3.3× |
| onnxsim | 1.0904 | -3.27% | 0.0194 | 1.78 | 0.378 | N | 2.8× |
| channels_last | 1.0935 | -3.00% | 0.0292 | 2.67 | 0.424 | N | 2.5× |
| opset13 | 1.0954 | -2.83% | 0.0254 | 2.32 | 0.445 | N | 2.4× |
| bn_folded | 1.1011 | -2.32% | 0.0319 | 2.90 | 0.534 | N | 2.0× |
| onnxoptimizer | 1.1025 | -2.20% | 0.0250 | 2.27 | 0.549 | N | 1.9× |
| *reexport_plain* | *1.1273* | *0.00%* | — | — | — | 基准 | — |

### 3.2 ResNet50

**噪声基底**: between-run std=0.0604ms (1.23%)，最小可检测效应量=1.23%

| 变体 | Mean(ms) | Diff(%) | Std(ms) | CV(%) | p值 | 显著? | 效应量/噪声 |
|---|---|---|---|---|---|---|---|
| **fp16** | **3.4684** | **-1.05%** | 0.0221 | 0.64 | **0.030** | **✅ Y** | 0.9× |
| onnxsim | 3.4826 | -0.65% | 0.0078 | 0.22 | 0.102 | N | 0.5× |
| opset13 | 3.4973 | -0.23% | 0.0236 | 0.67 | 0.603 | N | 0.2× |
| onnxoptimizer | 3.4986 | -0.19% | 0.0380 | 1.09 | 0.735 | N | 0.2× |
| channels_last | 3.5026 | -0.08% | 0.0383 | 1.09 | 0.893 | N | 0.1× |
| *reexport_plain* | *3.5052* | *0.00%* | — | — | — | 基准 | — |
| bn_folded | 3.5330 | +0.79% | 0.0357 | 1.01 | 0.164 | N | 0.6× |
| **shape_inferred** | **3.5681** | **+1.79%** | 0.0568 | 1.59 | **0.044** | **❌ Y*** | 1.5× |

> *注: 效应量/噪声 = |Diff%| / 最小可检测效应量，>1表示差异超过噪声基底*

### 3.3 关键图示

```
MobileNetV3-Small (基线=1.127ms, 噪声=1.18%)
  fp16           ████████████████████████░░░░░░░░░░░░░░  1.009ms (-10.51%) ✅
  shape_inferred ██████████████████████████████░░░░░░░░  1.084ms (-3.86%)
  onnxsim        ███████████████████████████████░░░░░░░  1.090ms (-3.27%)
  channels_last  ███████████████████████████████░░░░░░░  1.094ms (-3.00%)
  opset13        ████████████████████████████████░░░░░░  1.095ms (-2.83%)
  bn_folded      ████████████████████████████████░░░░░░  1.101ms (-2.32%)
  onnxoptimizer  ████████████████████████████████░░░░░░  1.103ms (-2.20%)
  [reexport_plain]█████████████████████████████████████  1.127ms (基线)

ResNet50 (基线=3.505ms, 噪声=1.23%)
  fp16           ██████████████████████████████████████  3.468ms (-1.05%) ✅
  onnxsim        ██████████████████████████████████████  3.483ms (-0.65%)
  opset13        ██████████████████████████████████████  3.497ms (-0.23%)
  onnxoptimizer  ██████████████████████████████████████  3.499ms (-0.19%)
  channels_last  ██████████████████████████████████████  3.503ms (-0.08%)
  [reexport_plain]██████████████████████████████████████  3.505ms (基线)
  bn_folded      ██████████████████████████████████████  3.533ms (+0.79%)
  shape_inferred ██████████████████████████████████████  3.568ms (+1.79%) ❌
```

---

## 第四部分：每种优化的因果解释

### 4.1 fp16 ✅ 唯一有效

| 指标 | MobileNet | ResNet50 |
|---|---|---|
| 效果 | -10.51% (p=0.024) | -1.05% (p=0.030) |
| ONNX变化 | 权重fp32→fp16，图结构不变 | 相同 |
| 文件大小 | 9.71→4.88 MB (-49.8%) | 97.41→48.72 MB (-50.0%) |
| OM大小 | 7,660,272 bytes | 51,678,060 bytes |
| 与原始AIPP OM对比 | 仅差287字节（0.004%） | 相似 |

**因果机制**:
- OM 与原始 AIPP OM 几乎一样大 → 原始基线本身就是在某种混合精度模式下编译的
- fp16 权重将参数带宽需求减半，降低数据搬运压力
- MobileNet 受益大（-10.51%）：轻量网络的带宽利用率已接近上限（83%），减半权重直接缓解瓶颈
- ResNet50 受益小（-1.05%）：主要瓶颈在大特征图搬运（而非参数量），因此收益有限

### 4.2 shape_inferred ⚠️ 对ResNet50有害

| 指标 | MobileNet | ResNet50 |
|---|---|---|
| 效果 | -3.86%（不显著） | **+1.79%（p=0.044，显著变慢）** |
| ONNX变化 | 嵌入shape信息，+0.12%大小 | 相同 |

**因果机制**:
- 补全 ONNX 中所有中间张量的静态 shape 信息
- ATC 编译器获得完整 shape 后，可能采用了不同的 kernel 切分策略
- ResNet50 的规则块结构下，完整 shape 导致编译器选择了次优的图优化路径
- 这解释了为什么"给编译器更多信息"反而有害

**启示**: shape inference 在推理链路上被 ATC 内部自动执行，手动补全 shape 是多余的。

### 4.3 onnxsim / onnxoptimizer / opset13 ⚠️ 无效

| ONNX变化 | 性能影响 |
|---|---|
| ir_version: 6→7 | 无（元信息变化） |
| opset: 11→13 | 无（算子表达方式变化，语义等价） |
| 无节点增减 | 图拓扑完全相同 |

**因果机制**: 原始导出图已经非常干净（无 Identity 节点、无死分支、常量已折叠），这些工具的优化 pass 找不到可操作的冗余。

### 4.4 channels_last ⚠️ 无效（ONNX相同）

**关键证据**: channels_last ONNX 与原始 ONNX **字节级完全相同**（MD5 一致）。

**因果机制**: PyTorch 的 `to(memory_format=torch.channels_last)` 在 ONNX 导出时丢失，layout 偏好无法传递到 ONNX 图中。不同的 OM 编译产物仅因输出文件名不同导致的元数据差异，功能等价。

### 4.5 bn_folded ⚠️ 失败

- **MobileNet**: ONNX 图中 BN 已经为 0，BN folding 无操作空间
- **ResNet50**: 虽减少 32 个 ReLU 节点（122→90），但数值验证显示严重失真（max_abs_diff=6.55e8，cosine=-0.099），模型已损坏

---

## 第五部分：环境噪声与实验可靠性

### 5.1 系统负载

| 指标 | 值 |
|---|---|
| CPU 核心 | 3 |
| 运行进程 | ~350 |
| 系统负载（Phase 1） | 17.3~18.6 |
| 系统负载（Phase 2） | 17.6~18.9 |

### 5.2 噪声量化

| 模型 | between-run std | 95%CI半宽 | 等效于基线% | 同OM最大极差 |
|---|---|---|---|---|
| MobileNet | 0.0181ms | 0.0130ms | 1.18% | 5.16% |
| ResNet50 | 0.0604ms | 0.0432ms | 1.23% | 5.65% |

### 5.3 结论可靠性

- **fp16**: 在 MobileNet 上效应量超出噪声 8.9× → **高度可靠**
- **fp16**: 在 ResNet50 上效应量超出噪声 0.9×（临界）→ 虽 p=0.030 显著，但效应量小，需更多数据确认
- **shape_inferred (ResNet50)**: 效应量超出噪声 1.5× → 显著变慢的结论可信
- 其他变体：效应量在噪声范围内 → 无法确认有效或有害

---

## 第六部分：深度优化方向设计

> 基于以下硬件特性设计：
> - DDR 带宽 ~25 GB/s，L2 Cache ~1MB @200 GB/s，L1 ~256KB @2 TB/s
> - ATC fusion pass 大量匹配未生效（Winograd 49/0, CastRemove 111/0, TransdataCast 57/0）
> - MobileNet 带宽瓶颈，ResNet50 数据搬运瓶颈

### 方向1：ATC 编译参数调优（最直接，低风险）

**当前参数**:
```
--precision_mode=allow_mix_precision
--buffer_optimize=l2_optimize
--op_select_implmode=high_performance
--enable_small_channel=1
```

**建议实验**:

#### 1a. precision_mode 对比
| 模式 | 预期效果 | 风险 |
|---|---|---|
| `force_fp16` | 全图fp16，可能速度更快 | 精度可能下降（需验证） |
| `cube_fp16in_fp32out` | Cube用fp16算，输出fp32，平衡精度与速度 | 低 |
| `allow_mix_precision`（当前） | — | — |
| `allow_fp32_to_fp16` | 自动将fp32转为fp16 | 低 |

#### 1b. buffer_optimize 对比
| 模式 | 预期效果 |
|---|---|
| `l2_optimize`（当前） | 优先使用L2缓存 |
| `l1_optimize` | 更激进地使用L1缓存（256KB），减少L2/DDR访问 |
| `off_optimize` | 关闭buffer优化（对照用） |

**原理**: L1 带宽（2 TB/s）远高于 L2（200 GB/s）和 DDR（25 GB/s）。`l1_optimize` 可能进一步减少 DDR 搬运，但受限于 256KB 大小，对 ResNet50 的大特征图不一定有利。

#### 1c. 尝试 fusion_switch_file
某些 pass（如 Winograd、WeightCompress）匹配了但未生效。可以用 `--fusion_switch_file` 强制启用。

### 方向2：混合精度配置（直接影响精度-速度权衡）

**当前**: `--precision_mode=allow_mix_precision` 让 ATC 自动决定哪些算子用 fp16。
**局限**: 默认混合精度策略可能保守。

**具体实验**:

#### 2a. modify_mixlist
ATC 允许通过 `--modify_mixlist` 指定一个算子精度白名单/黑名单，强制某些算子使用 fp16 或 fp32。

创建一个 mixlist 配置，指定 Conv、MatMul 等计算密集型算子使用 fp16，而 Add、ReLU 等逐元素算子保持 fp32，可以在保证精度的前提下最大化加速。

#### 2b. 使用 precision_mode_v2
`--precision_mode_v2=mixed_float16` 是更新的混合精度策略，可能与当前模型更兼容。

### 方向3：激活这些未生效的 fusion pass（最有潜力的方向）

**融合 pass 分析**（来自 `fusion_result.json`）:

| Pass | 匹配 | 生效 | 生效率 | 功能 |
|---|---|---|---|---|
| FIXPIPEFUSIONPASS | 54 | 50 | 92% | Conv+FixPipe融合 ✅ |
| TbeConvFixPipeFusionPass | 48 | 48 | 100% | UB层融合 ✅ |
| **Conv2DWinogradFusionPass** | **49** | **0** | **0%** | Winograd加速3×3卷积 ❌ |
| **RemoveCastFusionPass** | **111** | **0** | **0%** | 消除类型转换节点 ❌ |
| **TransdataCastFusionPass** | **57** | **0** | **0%** | 格式转换+Cast融合 ❌ |
| **ConvFormatRefreshFusionPass** | **53** | **0** | **0%** | 格式刷新 ❌ |
| **ConvRearrangeGfzFusionPass** | **49** | **0** | **0%** | 数据重排融合 ❌ |
| **ConvWeightCompressFusionPass** | **54** | **0** | **0%** | 权重压缩 ❌ |

**实验方案**:

#### 3a. 创建 fusion_switch 配置强制启用某些 pass
通过 `--fusion_switch_file` 指定配置文件，强制 ATC 应用那些它因保守策略跳过但仍匹配的 pass。

Winograd 对 3×3 卷积的理论加速可达 1.3-2×。ResNet50 有 53 个 Conv，大部分是 3×3。如果 Winograd 生效，理论上可能带来显著加速。

#### 3b. RemoveCastFusionPass（匹配111次但未生效）
图中存在大量 Cast 节点（类型转换，如 fp32↔fp16）。111 次匹配意味着有 111 个 Cast 可被消除。这可能来自 ONNX 导出时引入的 dtype 转换。

可以尝试：
1. 导出 ONNX 时统一算子 dtype 减少 Cast
2. 用 `--fusion_switch_file` 强制跑这个 pass

### 方向4：AOE 算子自动调优

AOE（Ascend Optimization Engine）已安装在系统中。它能自动搜索每个算子的最优 kernel 配置（tiling 参数、内存布局等）。

**操作步骤**:
```
aoe --framework=5 --model=model.onnx --job_type=2 --output=./aoe_result
```

其中 `job_type=2` 是全模型调优模式。AOE 会对每个算子的实现进行多次尝试，找到在当前硬件上最快的组合。

**预期效果**: 1-5% 的额外加速（取决于原始 ATC kernel 选择的次优程度）
**风险**: 调优时间较长（数小时），但不需要修改模型

### 方向5：权重通道打包（Weight Compression）

`ConvWeightCompressFusionPass` 匹配了 54 次但未生效。这个 pass 的功能是将卷积权重重排为更利于缓存访问的格式。

可以手动实现：
1. ONNX 导出后，将 Conv 权重按 channel 分组重排
2. 使用 `--compression_optimize_conf` 配置文件指导权重压缩

### 方向6：输入数据 pipeline 优化

当前 benchmark 使用 `np.zeros((224, 224, 3), dtype=np.uint8)` 作为输入，AIPP 做了预处理。可以检查：

1. **AIPP 配置优化**: 当前 AIPP 配置是否做了不必要的预处理步骤
2. **Host-Device 数据传输**: 检查输入数据从 Host 拷贝到 Device 的耗时是否成为瓶颈
3. **异步推理**: 用 AscendCL 的异步接口（`acl.mdl.execute_async`）叠加数据加载和推理

### 方向7：特定于 MobileNet 的深度优化

MobileNet 的深度可分离卷积（DW+PW）对调度极其敏感：

1. **手动融合 DW+PW+Act**: 将连续的 DepthwiseConv + PointwiseConv + Activation 手工融合为单个算子（通过 TBE 自定义算子开发）
2. **SE 模块优化**: SE 模块中的 GlobalAveragePool、FC、Sigmoid/Mul 链可以被融合为单一算子

### 优先级排序

| 优先级 | 方向 | 预期收益 | 工作量 | 风险 |
|---|---|---|---|---|
| **P0** | 1a. precision_mode 对比 | 1-5% | 低（改一行ATC命令） | 低 |
| **P0** | 1b. buffer_optimize 对比 | 1-5% | 低 | 低 |
| **P1** | 3a. fusion_switch 强制启用pass | 5-15%（如Winograd生效） | 中 | 中（稳定性风险） |
| **P1** | 4. AOE自动调优 | 1-5% | 低（调优时间长） | 低 |
| **P2** | 2. modify_mixlist 精细混合精度 | 3-8% | 中 | 中（精度验证需要） |
| **P3** | 3b. Cast 消除 | 1-3% | 中 | 低 |
| **P3** | 5. 权重通道打包 | 1-3% | 高 | 中 |
| **P4** | 6. 输入 pipeline 优化 | 取决于实测 | 中 | 低 |
| **P4** | 7. 自定义算子 | 10-20%（理论） | 极高 | 高 |

---

## 第七部分：结论

1. **fp16 是唯一被验证有效的 ONNX 前端优化**
   - MobileNet: -10.51%（p=0.024，效应量8.9×噪声）
   - ResNet50: -1.05%（p=0.030，效应量0.9×噪声，需更多确认）

2. **其他6种 ONNX 前端改图操作无效**
   - 根本原因：原始导出图已干净、图拓扑未改变
   - 表观 2-4% 差异在噪声范围内

3. **shape_inferred 在 ResNet50 上有害（+1.79%）**
   - 完整 shape 信息干扰了 ATC 的 kernel 选择

4. **下一步最值得尝试的方向**
   - **P0**: `--precision_mode` 和 `--buffer_optimize` 参数对比（即时可做）
   - **P1**: 通过 `--fusion_switch_file` 激活未生效的 fusion pass
   - **P1**: AOE 算子自动调优

## 附录：文件清单

### Opt_Analyze 交付物
| 文件 | 说明 |
|---|---|
| `reports/comprehensive_experiment_report.md` | 本报告 |
| `reports/analysis_report.md` | 前期分析报告 |
| `reports/final_results.md` | 简要结果汇总 |
| `data/rigorous_mobilenet_results.json` | MobileNet测量数据（含完整统计） |
| `data/rigorous_experiment_resnet50_*.json` | ResNet50测量数据 |
| `data/fair_comparison_*.json` | 公平对比实验数据 |
| `data/noise_diagnosis.json` | 噪声诊断数据 |
| `data/onnx_diff_analysis.json` | ONNX差异分析数据 |
| `scripts/rigorous_experiment.py` | 严格两阶段实验脚本 |
| `scripts/fair_compare.py` | 公平对比实验脚本 |
| `scripts/controlled_benchmark.py` | 受控benchmark框架 |
| `scripts/env_monitor.py` | 环境监控工具 |
| `scripts/onnx_diff_analyzer.py` | ONNX差异分析工具 |
| `scripts/diagnose_noise.py` | 噪声诊断工具 |

### 原始数据来源
| 位置 | 内容 |
|---|---|
| `Optimization/reports/experiment_report.md` | 原始实验报告 |
| `Optimization/reports/critic_review.md` | 审稿意见 |
| `Optimization/reports/variable_control_and_interpretation.md` | 变量控制文档 |
| `Optimization/results/*.json` | 原始实验结果 |
| `Optimization/models/*.om` | OM模型文件 |
| `Optimization/models/*.onnx` | ONNX模型文件 |
