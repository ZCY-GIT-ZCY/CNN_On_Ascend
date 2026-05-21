"""
昇腾 NPU 推理测试脚本
用于香橙派 AIpro (Ascend 310B1)
"""
import os
import sys
import cv2
import numpy as np

# 添加公用 ACLLite 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common', 'acllite_utils'))

from acllite_resource import AclLiteResource
from acllite_model import AclLiteModel


def preprocess_image(image_path: str, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    图像预处理函数

    Args:
        image_path: 输入图片路径
        target_size: 目标尺寸 (width, height)

    Returns:
        预处理后的图像数据（UINT8格式，用于 AIPP）
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    # 缩放到模型所需的静态分辨率
    input_img = cv2.resize(img, target_size)

    # 【核心注意点】因为开启了 AIPP：
    # 模型输入节点接收的类型是 UINT8（0-255 整数）
    # 严禁在此处做 /255.0 或减均值操作，否则 AIPP 会二次处理！
    return input_img.astype(np.uint8)


def main():
    """主推理函数"""
    model_path = "./resnet50_aipp.om"
    image_path = "../common/test.jpg"

    print("=" * 50)
    print("ResNet50 昇腾 NPU 推理测试")
    print("=" * 50)

    # 1. 检查模型文件
    if not os.path.exists(model_path):
        print(f"[错误] 未找到模型文件: {model_path}")
        return 1

    # 2. 初始化昇腾 NPU 资源
    try:
        print("[1/6] 初始化 NPU 资源...")
        resource = AclLiteResource()
        resource.init()
    except Exception as e:
        print(f"[错误] NPU 资源初始化失败: {e}")
        return 1

    # 3. 加载模型
    try:
        print("[2/6] 加载离线模型...")
        model = AclLiteModel(model_path)
    except Exception as e:
        print(f"[错误] 模型加载失败: {e}")
        del resource
        return 1

    # 4. 图像预处理
    try:
        print("[3/6] 图像预处理...")
        if os.path.exists(image_path):
            input_data = preprocess_image(image_path, (224, 224))
            print(f"      使用图片: {image_path}")
        else:
            print(f"[警告] 未找到测试图片，使用全零矩阵...")
            input_data = np.zeros((224, 224, 3), dtype=np.uint8)
    except Exception as e:
        print(f"[错误] 图像预处理失败: {e}")
        del model
        del resource
        return 1

    # 5. 执行推理
    try:
        print("[4/6] 执行 NPU 推理...")
        outputs = model.execute([input_data])
        print("      推理完成")
    except Exception as e:
        print(f"[错误] 模型推理失败: {e}")
        del model
        del resource
        return 1

    # 6. 后处理
    print("[5/6] 后处理结果...")
    logits = outputs[0]
    predicted_id = np.argmax(logits, axis=1)[0]
    predicted_score = logits[0, predicted_id]

    # 7. 释放资源
    print("[6/6] 释放资源...")
    del model
    del resource

    # 输出结果
    print("\n" + "=" * 50)
    print("  推理成功!")
    print("=" * 50)
    print(f"  输出形状: {logits.shape}")
    print(f"  预测类别: {predicted_id}")
    print(f"  置信度:   {predicted_score:.4f}")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
