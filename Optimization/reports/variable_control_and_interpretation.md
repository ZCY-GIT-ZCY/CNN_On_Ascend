# 变量控制与后续工作 Todo List

## 核心思路

后续分析的目标，不是继续堆更多优化尝试，而是回答一个统一问题：

> 每种优化尝试相对 baseline，最早是从哪一层开始产生差别的？

可追踪的差别起始点主要有三类：

- `torch` 层：如导出方式、layout 痕迹、精度设置、算子表达来源不同
- `onnx` 层：如图结构、dtype、opset、常量折叠、节点增删改不同
- `om` 层：如编译产物大小、算子融合、执行路径、任务组织不同

后续工作应尽量把每个变体相对 baseline 的差异链条写成：

> 差异起始点 -> ONNX 可观测差异 -> OM 可观测差异 -> 性能差异 -> 数值有效性

重点不是先猜后端，而是尽可能把 `onnx` 和 `om` 层的差异读出来。

---

## Todo 1：先给每种优化标记“差异起始点”

- [ ] 为每个变体相对 baseline 标记最早分叉层级：`torch` / `onnx` / `om`
- [ ] 对每个变体写一句话说明：
  - 它是从哪里开始和 baseline 不同的
  - 它之后会影响哪些下游产物
- [ ] 建立统一判断原则：
  - 如果差异先出现在导出/重导出阶段，就记为 `torch` 起始
  - 如果差异先出现在 ONNX 图改写阶段，就记为 `onnx` 起始
  - 如果前面都一致、只是在编译产物或部署产物上不同，就记为 `om` 起始

可先按下面方式粗分：

- [ ] `shape_inferred`：`onnx` 起始
- [ ] `opset13`：`onnx` 起始
- [ ] `onnxsim`：`onnx` 起始
- [ ] `onnxoptimizer`：`onnx` 起始
- [ ] `bn_folded`：通常视为 `onnx` 起始
- [ ] `fp16`：需要区分是 `torch` 导出时完成，还是 `onnx` 后处理完成
- [ ] `channels_last`：更可能是 `torch` 起始
- [ ] `baseline_copy`：`om` 起始或直接视为 baseline 参照项

---

## Todo 2：先固定分析边界

- [ ] 明确当前对比的**基线**是已部署的 AIPP OM，而不是“原始 ONNX 直接编译结果”
- [ ] 在后续报告中统一使用更严谨的表述：
  - 当前结果说明的是“现有部署基线”与“重新导出并优化后的变体”的工程对比
  - 不直接把结果写成“某种前端优化方法在 Ascend 上无效”
- [ ] 单独写清楚：当前差异里混合了两类因素：
  - 优化方法本身的影响
  - 重新导出 / 重新编译链路的整体影响

---

## Todo 3：把 AIPP 从“疑点”变成“受控变量”

- [ ] 核对所有变体编译时是否使用同一份 `aipp.cfg` 逻辑
- [ ] 在文档中明确说明：
  - AIPP 中的均值、方差倒数等常数会影响绝对输入数值
  - 但若 baseline 与各变体使用相同配置，它们就不是解释**相对性能差异**的主要变量
- [ ] 若历史 baseline 的 AIPP 配置与当前重编译链路一致，则将 AIPP 标记为**已控制变量**
- [ ] 后续不要再把 AIPP 常数本身作为 `fp16` / `onnxsim` / `onnxoptimizer` 性能差异的主要解释来源

---

## Todo 4：优先追 ONNX 差异，因为这是最容易读出来的一层

适用对象：`onnxsim`、`onnxoptimizer`、`bn_folded`、`fp16`，以及必要时的 `shape_inferred`、`opset13`、`channels_last`

- [ ] 对每个变体，先记录“baseline 对应 ONNX”与“当前变体 ONNX”的差异
- [ ] 若 baseline 没有完全同源 ONNX，则至少记录“原始导出 ONNX”与“优化后 ONNX”的差异
- [ ] 至少提取以下基础统计：
  - [ ] 总节点数
  - [ ] initializer 数量
  - [ ] 各类 op 计数
  - [ ] transpose-like / cast-like / reshape-like 节点情况
  - [ ] 文件大小
- [ ] 对关键变体补充“语义级差异摘要”：
  - [ ] 删除了哪些节点类型
  - [ ] 新增了哪些节点类型
  - [ ] 哪些模式被合并/折叠
  - [ ] 是否只是清理边角节点，还是改动了主计算链
- [ ] 形成一个原则：
  - 先把 ONNX 层的变化读清楚
  - 再讨论这些变化如何传递到 `.om` 和性能结果

---

## Todo 5：继续追 OM 差异，因为性能结果最终落在这一层

- [ ] 对每个变体补充读取 `.om` 层面的可观测差异
- [ ] 至少记录：
  - [ ] `.om` 文件大小
  - [ ] 编译日志中 warning / fallback / 特殊 pass 信息
  - [ ] 是否有 operator log
  - [ ] 是否能看到融合、切分、执行路径、task 组织差异
- [ ] 若 ONNX 差异很小但性能差异明显，优先去 `.om` 层找解释
- [ ] 若 ONNX 差异明显但 `.om` 差异不明显，记录为“前端变化未明显转化为后端收益”
- [ ] 对每个变体尽量形成一句归纳：
  - ONNX 变化是否大
  - OM 变化是否大
  - 性能变化是否与前两者一致

---

## Todo 6：单独把 `fp16` 作为“可精确分析变量”处理

- [ ] 不把 `fp16` 直接归入“只能猜测后端机制”的范畴
- [ ] 先明确 `fp16` 的直接变化：
  - [ ] 哪些 initializer / tensor dtype 从 `fp32` 变为 `fp16`
  - [ ] 输入输出 dtype 是否变化
  - [ ] 是否引入额外 `Cast`
  - [ ] 模型文件大小变化了多少
- [ ] 将 `fp16` 的分析重点放在：
  - [ ] dtype 变化
  - [ ] cast 变化
  - [ ] 模型体积变化
  - [ ] 对应 `.om` 体积变化
  - [ ] 数值验证状态
  - [ ] 性能变化是否与“模型变小”一致
- [ ] 在报告中明确区分：
  - `fp16` 的作用边界是可描述的
  - 当前尚未完全解释的，是这些已知变化经过编译器后为何只带来有限收益

---

## Todo 7：给不同优化类型分级分析

- [ ] 把变体分成三类分别讨论

### A. 显式改图类
- `onnxsim`
- `onnxoptimizer`
- `bn_folded`

后续要求：优先做 ONNX diff，再看这些 diff 有没有传到 `.om`。

### B. 精度/类型类
- `fp16`

后续要求：优先做 dtype / cast / size diff，再看 `.om` 和性能是否同步变化。

### C. 表达方式/元信息类
- `shape_inferred`
- `opset13`
- `channels_last`

后续要求：结合差异起始点、图统计、编译响应和性能一起解释。

---

## Todo 8：建立统一“差异跟踪表”

每个模型的每个变体一行，后续统一填表。

- [ ] 基础标识字段
  - `model_key`
  - `experiment_id`
  - `variant_name`
  - `source_path`
  - `artifact_path`
  - `compile_artifact`
  - `stage`
  - `status`
- [ ] 差异起始点字段
  - `divergence_origin` (`torch` / `onnx` / `om`)
  - `origin_note`
- [ ] ONNX 差异字段
  - `onnx_total_nodes`
  - `onnx_initializer_count`
  - `onnx_op_counts`
  - `onnx_transpose_like_nodes`
  - `onnx_cast_like_nodes`
  - `onnx_reshape_like_nodes`
  - `onnx_graph_file_size_mb`
  - `onnx_semantic_diff_summary`
- [ ] OM 差异字段
  - `om_size_mb`
  - `compile_warning_count`
  - `has_operator_log`
  - `fusion_summary`
  - `compile_path_note`
  - `runtime_bottleneck_hint`
- [ ] 运行时性能字段
  - `latency_ms`
  - `fps`
  - `std_ms`
  - `min_ms`
  - `max_ms`
  - `p50_ms`
  - `p90_ms`
  - `p95_ms`
  - `p99_ms`
- [ ] 数值有效性字段
  - `numerical_status`
  - `sample_count`
  - `max_abs_diff`
  - `mean_abs_diff`
  - `rmse`
  - `cosine_similarity_mean`
  - `top1_match_rate`
  - `validation_note`

---

## Todo 9：先给样本打解释标签，再写结论

建议给每个样本补标签：

- [ ] `valid_for_main_claim`
- [ ] `numerically_verified`
- [ ] `diverged_at_torch`
- [ ] `diverged_at_onnx`
- [ ] `diverged_at_om`
- [ ] `onnx_changed_significantly`
- [ ] `om_changed_significantly`
- [ ] `dtype_changed_significantly`
- [ ] `likely_compile_path_changed`
- [ ] `candidate_memory_bound`

这样写结论时能先筛选样本，也能直接看出每个结论是在追哪一层差异。

---

## Todo 10：明确哪些样本能进入主结论

- [ ] 将样本分为三类：
  - **数值正确样本**：可进入主性能解释
  - **数值失真样本**：只能作为失败案例或反例
  - **验证未闭环样本**：可记录性能，但不能直接当作最终有效候选
- [ ] 明确记录当前特殊样本：
  - `ResNet50 bn_folded`：数值失真，不能作为成功优化样本
  - `ResNet50 fp16`：验证未闭环，当前问题来自 ONNX Runtime 支持链路，而不是已证实模型错误

---

## Todo 11：把“能强说的”和“只能保守说的”分开

### 可以较强表述的内容
- [ ] 当前工程中的已部署 AIPP OM 基线，在两类模型上都快于所有已测试重导出变体
- [ ] `MobileNetV3-Small` 对前端改写更敏感
- [ ] `ResNet50` 各正确样本相对基线退化较小
- [ ] 简单前端图清理或 shape 补全没有自动转化为更快的 `.om`

### 需要保守表述的内容
- [ ] 某些变体可能诱发了不同的后端融合或任务组织路径
- [ ] `channels_last` / `onnxsim` 等退化可能与格式处理或数据搬运有关
- [ ] `MobileNetV3-Small` 的退化可能与其 DW/PW/SE 链路更碎、更依赖调度有关
- [ ] `fp16` 收益有限，可能说明瓶颈不只在参数体积

### 当前不要直接写死的内容
- [ ] “编译器一定插入了 transpose 并导致性能下降”
- [ ] “切块方式一定改变了，并且这是唯一原因”
- [ ] “某种前端优化在 Ascend 平台上无效”
- [ ] “重新导出本身就是错误的”

---

## Todo 12：后续写报告时的推荐顺序

- [ ] 先写 baseline 定义与同源性边界
- [ ] 再写每种优化的差异起始点
- [ ] 再写 AIPP 是否已控制
- [ ] 再写 ONNX 可观测差异
- [ ] 再写 OM 可观测差异
- [ ] 再写数值有效性筛选
- [ ] 再写性能排序
- [ ] 最后写编译器/运行时解释与结论边界

---

## 最终提醒

后续分析时遵守这四条原则：

- [ ] 先问：这个变体和 baseline 最早从哪一层开始不同
- [ ] 对 AIPP：优先判断它是不是已控制变量，而不是反复怀疑常数本身
- [ ] 对 `fp16`：优先做 dtype / cast / size 的精确分析，而不是先黑箱化
- [ ] 对 `onnxsim` / `onnxoptimizer`：优先读取 ONNX 前后差异，再继续追 `.om` 差异，而不是先猜后端发生了什么
