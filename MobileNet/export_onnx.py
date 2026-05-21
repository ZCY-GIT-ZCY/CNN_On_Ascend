"""
导出 MobileNetV3 模型为 ONNX 格式
用于昇腾 ATC 转换
"""
import torch
import torchvision.models as models

def export_mobilenet_onnx(output_path="mobilenetv3.onnx", model_type="small"):
    """导出 MobileNetV3 为 ONNX 格式

    Args:
        output_path: 输出 ONNX 文件路径
        model_type: "small" 或 "large"
    """

    print("=" * 50)
    print(f"MobileNetV3 ({model_type}) ONNX 导出工具")
    print("=" * 50)

    # 1. 初始化模型
    print("[1/4] 加载 MobileNetV3 模型...")
    if model_type == "small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    else:
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    model.eval()
    print(f"      MobileNetV3-{model_type} 模型加载完成")

    # 2. 构造静态输入张量
    print("[2/4] 构造输入张量...")
    # Batch_Size=1, Channels=3, Height=224, Width=224
    dummy_input = torch.randn(1, 3, 224, 224)
    print(f"      输入 Shape: {dummy_input.shape}")

    # 3. 导出 ONNX
    print("[3/4] 导出 ONNX 模型...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        verbose=False,
        input_names=["input_image"],
        output_names=["output_tensor"],
        opset_version=11,
        dynamic_axes=None  # 静态 Shape，利于 ATC 优化
    )
    print(f"      已保存至: {output_path}")

    # 4. 验证
    print("[4/4] 验证 ONNX 模型...")
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("      ONNX 模型验证通过")
    except Exception as e:
        print(f"      验证警告: {e}")

    print("=" * 50)
    print("导出完成!")
    print("=" * 50)

    return output_path


if __name__ == "__main__":
    # 默认导出 MobileNetV3-Small
    # 如需导出 Large 版本: export_mobilenet_onnx("mobilenetv3_large.onnx", "large")
    export_mobilenet_onnx()
