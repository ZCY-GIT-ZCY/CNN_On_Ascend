"""
AIPP 参数计算工具
根据 ImageNet 或自定义均值/方差生成 AIPP 配置文件

用法:
    python gen_aipp_cfg.py                    # 使用默认 ImageNet 参数
    python gen_aipp_cfg.py --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5  # 自定义参数
"""

import argparse
import numpy as np


def calculate_aipp_params(mean_rgb, std_rgb):
    """
    计算 AIPP 参数

    AIPP 公式: pixel_out = (pixel_in - mean_chn_i - min_chn_i) * var_reci_chn_i

    Args:
        mean_rgb: RGB 均值列表 [R, G, B]
        std_rgb: RGB 标准差列表 [R, G, B]

    Returns:
        dict: AIPP 参数
    """
    # BGR 顺序: 0=B, 1=G, 2=R
    # AIPP 输入是 BGR 格式，所以需要反向映射
    bgr_mean = [
        255 * mean_rgb[2],  # B 通道 = 255 * B 均值
        255 * mean_rgb[1],  # G 通道 = 255 * G 均值
        255 * mean_rgb[0]   # R 通道 = 255 * R 均值
    ]

    bgr_std = [
        std_rgb[2],  # B 通道
        std_rgb[1],  # G 通道
        std_rgb[0]   # R 通道
    ]

    var_reci = [
        1 / (255 * bgr_std[0]),
        1 / (255 * bgr_std[1]),
        1 / (255 * bgr_std[2])
    ]

    return {
        'mean': [int(round(v)) for v in bgr_mean],
        'var_reci': var_reci
    }


def generate_aipp_config(mean_rgb, std_rgb, input_size=224, output_path='aipp.cfg'):
    """
    生成 AIPP 配置文件

    Args:
        mean_rgb: RGB 均值列表
        std_rgb: RGB 标准差列表
        input_size: 输入图片尺寸
        output_path: 输出文件路径
    """
    params = calculate_aipp_params(mean_rgb, std_rgb)

    config = f'''# AIPP 配置文件 - 自动生成
# 归一化参数: 均值={mean_rgb}, 标准差={std_rgb}

aipp_op {{
    aipp_mode: static
    input_format: 3                  # BGR888_U8 = 3
    src_image_size_w: {input_size}
    src_image_size_h: {input_size}

    # BGR 通道均值
    mean_chn_0: {params['mean'][0]}                  # B 通道
    mean_chn_1: {params['mean'][1]}                  # G 通道
    mean_chn_2: {params['mean'][2]}                  # R 通道

    # 最小值
    min_chn_0: 0.0
    min_chn_1: 0.0
    min_chn_2: 0.0

    # 方差倒数
    var_reci_chn_0: {params['var_reci'][0]:.8f}
    var_reci_chn_1: {params['var_reci'][1]:.8f}
    var_reci_chn_2: {params['var_reci'][2]:.8f}
}}
'''

    with open(output_path, 'w') as f:
        f.write(config)

    print(f"AIPP 配置已生成: {output_path}")
    print()
    print("参数预览:")
    print(f"  mean_chn_0: {params['mean'][0]}")
    print(f"  mean_chn_1: {params['mean'][1]}")
    print(f"  mean_chn_2: {params['mean'][2]}")
    print(f"  var_reci_chn_0: {params['var_reci'][0]:.8f}")
    print(f"  var_reci_chn_1: {params['var_reci'][1]:.8f}")
    print(f"  var_reci_chn_2: {params['var_reci'][2]:.8f}")

    return params


def main():
    parser = argparse.ArgumentParser(description='AIPP 参数计算工具')
    parser.add_argument('--mean', type=float, nargs=3,
                        default=[0.485, 0.456, 0.406],
                        help='RGB 均值 (默认: 0.485 0.456 0.406)')
    parser.add_argument('--std', type=float, nargs=3,
                        default=[0.229, 0.224, 0.225],
                        help='RGB 标准差 (默认: 0.229 0.224 0.225)')
    parser.add_argument('--size', type=int, default=224,
                        help='输入图片尺寸 (默认: 224)')
    parser.add_argument('-o', '--output', default='aipp.cfg',
                        help='输出文件路径 (默认: aipp.cfg)')

    args = parser.parse_args()

    print("=" * 50)
    print("AIPP 参数计算工具")
    print("=" * 50)
    print(f"RGB 均值: {args.mean}")
    print(f"RGB 标准差: {args.std}")
    print(f"输入尺寸: {args.size}x{args.size}")
    print("-" * 50)

    generate_aipp_config(args.mean, args.std, args.size, args.output)


if __name__ == "__main__":
    main()
