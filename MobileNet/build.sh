#!/bin/bash
# MobileNetV3 ATC 编译脚本
# 用法: ./build.sh [onnx_file] [output_name]

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 设置模型名称
ONNX_FILE="${1:-mobilenetv3.onnx}"
OUTPUT_NAME="${2:-mobilenetv3_aipp}"

echo "=================================================="
echo "MobileNetV3 ATC 编译"
echo "=================================================="
echo "ONNX 文件: $ONNX_FILE"
echo "输出名称: $OUTPUT_NAME"
echo ""

# 检查源文件
if [ ! -f "$SCRIPT_DIR/$ONNX_FILE" ]; then
    echo "[错误] 未找到 ONNX 文件: $SCRIPT_DIR/$ONNX_FILE"
    echo "[提示] 请先运行 python export_onnx.py 导出模型"
    exit 1
fi

# 检查 AIPP 配置
if [ ! -f "$SCRIPT_DIR/aipp.cfg" ]; then
    echo "[错误] 未找到 AIPP 配置文件: $SCRIPT_DIR/aipp.cfg"
    exit 1
fi

# 执行 ATC 编译
echo "[1/1] 执行 ATC 编译..."
atc --model="$SCRIPT_DIR/$ONNX_FILE" \
    --framework=5 \
    --output="$SCRIPT_DIR/$OUTPUT_NAME" \
    --soc_version=Ascend310B1 \
    --input_shape="input_image:1,3,224,224" \
    --insert_op_conf="$SCRIPT_DIR/aipp.cfg" \
    --enable_small_channel=1

echo ""
echo "=================================================="
echo "编译完成! 输出文件: $OUTPUT_NAME.om"
echo "=================================================="
