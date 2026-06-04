# Optimization 实验目录

本目录用于完成 Ascend 310B1 上 CNN 算子融合与优化实验，聚焦两类模型：

- `ResNet50`：普通卷积 / 残差块主导
- `MobileNetV3-Small`：深度可分离卷积 / SE / 短残差主导

## 当前内容

- `fusion_analysis.json`：模型结构与潜在融合点统计
- `config.py`：实验配置
- `analyze_fusion.py`：融合形态总结脚本
- `run_experiments.py`：图优化与基准测试主脚本
- `check_env.sh`：环境检查脚本
- `requirements.txt`：实验补充依赖

## 计划中的实验方向

1. ONNX 简化与 shape inference
2. BN folding
3. 不同 opset 导出
4. channels_last / layout 优化
5. FP16 导出与模型体积压缩
6. 编译后 OM 的统一基准测试
7. 汇总报告与可视化
