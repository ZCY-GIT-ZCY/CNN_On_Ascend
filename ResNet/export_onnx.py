"""
导出 ResNet50 模型为 ONNX 格式
用于昇腾 ATC 转换
"""
import torch
import torchvision.models as models

def export_resnet50_onnx(output_path="resnet50.onnx"):
    """导出 ResNet50 为 ONNX 格式"""

    print("=" * 50)
    print("ResNet50 ONNX 导出工具")
    print("=" * 50)

    # 1. 初始化模型
    print("[1/4] 加载 ResNet50 模型...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    print("      模型加载完成")

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
    export_resnet50_onnx()
