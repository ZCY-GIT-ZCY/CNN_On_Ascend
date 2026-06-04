#!/bin/bash
# Ascend 310B1 环境检查脚本
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate ascend_cnn

echo "========== 环境检查 =========="
echo "Python: $(python --version)"
echo "CANN: $ASCEND_OPP_PATH"
echo "ATC: $(which atc)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "ONNX: $(python -c 'import onnx; print(onnx.__version__)')"
echo "OpenCV: $(python -c 'import cv2; print(cv2.__version__)')"

echo ""
echo "========== NPU 状态 =========="
python -c "
from acllite_resource import AclLiteResource
try:
    resource = AclLiteResource()
    resource.init()
    print('NPU 初始化: OK')
    del resource
except Exception as e:
    print(f'NPU 初始化: FAILED - {e}')
" 2>/dev/null || echo "NPU 状态检查完成"

echo ""
echo "========== ONNX 模型文件 =========="
for f in mobilenetv3.onnx resnet50.onnx; do
    for d in MobileNet ResNet; do
        if [ -f "/home/HwHiAiUser/Desktop/CNN/$d/$f" ]; then
            size=$(du -h "/home/HwHiAiUser/Desktop/CNN/$d/$f" | cut -f1)
            echo "  $d/$f: $size"
        fi
    done
done

echo ""
echo "========== OM 模型文件 =========="
for f in mobilenetv3_aipp.om resnet50_aipp.om; do
    for d in MobileNet ResNet; do
        if [ -f "/home/HwHiAiUser/Desktop/CNN/$d/$f" ]; then
            size=$(du -h "/home/HwHiAiUser/Desktop/CNN/$d/$f" | cut -f1)
            echo "  $d/$f: $size"
        fi
    done
done

echo ""
echo "========== 检查完成 =========="
