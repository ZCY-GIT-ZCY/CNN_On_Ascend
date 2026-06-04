"""导出多种优化版本 ONNX，用于后续 ATC 与性能对比。"""
from pathlib import Path
from typing import Tuple

import torch
import torchvision.models as models
from torch.ao.quantization import fuse_modules

from config import MODELS_DIR


INPUT_SHAPE = (1, 3, 224, 224)


def build_dummy_input(dtype: torch.dtype = torch.float32, channels_last: bool = False) -> torch.Tensor:
    dummy = torch.randn(*INPUT_SHAPE, dtype=dtype)
    if channels_last:
        dummy = dummy.contiguous(memory_format=torch.channels_last)
    return dummy


def export_onnx(model, output_path: Path, opset: int = 11, channels_last: bool = False) -> None:
    model.eval()
    dtype = next(model.parameters()).dtype
    dummy = build_dummy_input(dtype=dtype, channels_last=channels_last)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        verbose=False,
        input_names=["input_image"],
        output_names=["output_tensor"],
        opset_version=opset,
        dynamic_axes=None,
    )


def simplify_onnx(input_path: Path, output_path: Path) -> Tuple[bool, str]:
    try:
        from onnxsim import simplify
        import onnx

        model = onnx.load(str(input_path))
        simplified, ok = simplify(model)
        if not ok:
            return False, "onnxsim check failed"
        onnx.save(simplified, str(output_path))
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def optimize_onnx(input_path: Path, output_path: Path) -> Tuple[bool, str]:
    try:
        import onnx
        import onnxoptimizer

        model = onnx.load(str(input_path))
        optimized = onnxoptimizer.optimize(
            model,
            [
                "eliminate_identity",
                "eliminate_deadend",
                "eliminate_nop_dropout",
                "eliminate_nop_pad",
                "extract_constant_to_initializer",
                "eliminate_unused_initializer",
                "eliminate_duplicate_initializer",
                "fuse_bn_into_conv",
                "fuse_consecutive_transposes",
                "fuse_transpose_into_gemm",
            ],
        )
        onnx.save(optimized, str(output_path))
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def fuse_resnet_for_inference(model):
    model = model.eval()
    fuse_modules(model, [["conv1", "bn1", "relu"]], inplace=True)
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(model, layer_name)
        for block in layer:
            fuse_modules(block, [["conv1", "bn1", "relu"]], inplace=True)
            fuse_modules(block, [["conv2", "bn2"]], inplace=True)
            fuse_modules(block, [["conv3", "bn3"]], inplace=True)
            if block.downsample is not None:
                fuse_modules(block.downsample, [["0", "1"]], inplace=True)
    return model


def fuse_mobilenet_for_inference(model):
    model = model.eval()
    for feature in [model.features[0], model.features[-1]]:
        modules = feature._modules
        activation = modules.get("2")
        if activation is not None and activation.__class__.__name__ == "ReLU":
            fuse_modules(feature, [["0", "1", "2"]], inplace=True)
        else:
            fuse_modules(feature, [["0", "1"]], inplace=True)
    for layer in model.features:
        if not hasattr(layer, "block"):
            continue
        for sub in layer.block:
            if hasattr(sub, "_modules") and "0" in sub._modules and "1" in sub._modules:
                keys = list(sub._modules.keys())
                activation = sub._modules.get("2")
                if keys == ["0", "1", "2"] and activation.__class__.__name__ == "ReLU":
                    fuse_modules(sub, [["0", "1", "2"]], inplace=True)
                else:
                    fuse_modules(sub, [["0", "1"]], inplace=True)
    return model


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)

    export_onnx(resnet, MODELS_DIR / "resnet50_opset13.onnx", opset=13)
    export_onnx(mobilenet, MODELS_DIR / "mobilenet_v3_small_opset13.onnx", opset=13)

    export_onnx(
        fuse_resnet_for_inference(models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)),
        MODELS_DIR / "resnet50_bn_folded.onnx",
        opset=11,
    )
    export_onnx(
        fuse_mobilenet_for_inference(models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)),
        MODELS_DIR / "mobilenet_v3_small_bn_folded.onnx",
        opset=11,
    )

    resnet_cl = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).eval().to(memory_format=torch.channels_last)
    mobilenet_cl = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1).eval().to(memory_format=torch.channels_last)
    export_onnx(resnet_cl, MODELS_DIR / "resnet50_channels_last.onnx", opset=11, channels_last=True)
    export_onnx(mobilenet_cl, MODELS_DIR / "mobilenet_v3_small_channels_last.onnx", opset=11, channels_last=True)

    resnet_fp16 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).eval().half()
    mobilenet_fp16 = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1).eval().half()
    export_onnx(resnet_fp16, MODELS_DIR / "resnet50_fp16.onnx", opset=11)
    export_onnx(mobilenet_fp16, MODELS_DIR / "mobilenet_v3_small_fp16.onnx", opset=11)

    for model_name in ["resnet50", "mobilenet_v3_small"]:
        source = MODELS_DIR / f"{model_name}_opset13.onnx"
        simplified = MODELS_DIR / f"{model_name}_onnxsim.onnx"
        optimized = MODELS_DIR / f"{model_name}_onnxoptimizer.onnx"
        simplify_ok, simplify_msg = simplify_onnx(source, simplified)
        optimize_ok, optimize_msg = optimize_onnx(source, optimized)
        print(f"{model_name}:onnxsim:{'ok' if simplify_ok else 'failed'}:{simplify_msg}")
        print(f"{model_name}:onnxoptimizer:{'ok' if optimize_ok else 'failed'}:{optimize_msg}")

    print("export_done")


if __name__ == "__main__":
    main()
