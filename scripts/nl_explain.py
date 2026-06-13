#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LAIGCD 自然语言解释脚本

使用FakeVLM生成AIGC检测的自然语言解释
可以独立运行，不依赖检测器训练流程
"""

import os
import sys
import argparse
import json
from pathlib import Path
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.fakevlm_explainer import create_explainer
from configs import FakeVLMConfig, get_config


def load_images_for_explanation(
    original_path,
    spatial_heatmap_path=None,
    frequency_heatmap_path=None,
):
    """
    加载用于解释的图像

    Args:
        original_path: 原图路径
        spatial_heatmap_path: 空域热力图路径
        frequency_heatmap_path: 频域热力图路径

    Returns:
        图像路径字典
    """
    return {
        'original': original_path,
        'spatial_heatmap': spatial_heatmap_path,
        'frequency_heatmap': frequency_heatmap_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description='LAIGCD 自然语言解释 - 使用FakeVLM生成解释',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 单图解释（仅原图）
  python nl_explain.py image.jpg

  # 使用热力图解释
  python nl_explain.py image.jpg --spatial spatial.png --frequency freq.png

  # 批量解释
  python nl_explain.py --image_dir images/

  # 使用检测结果信息
  python nl_explain.py image.jpg --prediction Fake --confidence 0.87

  # 使用本地模型
  python nl_explain.py image.jpg --model_path /path/to/model
        """
    )

    # 输入参数
    parser.add_argument('image', nargs='?', help='单张图像路径')
    parser.add_argument('--image_dir', help='图像目录路径')
    parser.add_argument('--spatial', help='空域热力图路径')
    parser.add_argument('--frequency', help='频域热力图路径')

    # FakeVLM配置
    parser.add_argument('--model_path', default='lingcco/fakeVLM',
                        help='FakeVLM模型路径')
    parser.add_argument('--device', default='cuda', help='运行设备')
    parser.add_argument('--no_8bit', dest='load_in_8bit', action='store_false', default=True,
                        help='禁用8bit量化（使用更高精度但需要更多显存）')

    # 检测结果信息（可选）
    parser.add_argument('--prediction', choices=['Real', 'Fake'],
                        help='检测结果（用于生成更准确的解释）')
    parser.add_argument('--confidence', type=float,
                        help='检测置信度 (0-1)')
    parser.add_argument('--top_prototypes', help='Top原型信息 (JSON格式)')

    # 生成参数
    parser.add_argument('--max_new_tokens', type=int, default=256,
                        help='最大生成token数')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='采样温度')
    parser.add_argument('--preset', choices=['default', 'fast', 'detailed', 'low_memory'],
                        default='default', help='预设配置')

    # 输出参数
    parser.add_argument('--output_dir', default='nl_explanations',
                        help='输出目录')
    parser.add_argument('--save_images', action='store_true',
                        help='保存输入图像的副本')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载配置
    config = get_config(args.preset, **{
        'model_path': args.model_path,
        'device': args.device,
        'load_in_8bit': args.load_in_8bit,
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
    })

    # 创建解释器
    print(f"\n正在初始化 FakeVLM 解释器...")
    print(f"模型: {config.model_path}")
    print(f"预设: {args.preset}")
    explainer = create_explainer(
        model_path=config.model_path,
        device=config.device,
        load_in_8bit=config.load_in_8bit,
    )

    explainer.load_model()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def process_single(image_path, spatial_path=None, freq_path=None):
        """处理单张图像"""
        image_name = Path(image_path).stem

        # 解析top_prototypes
        top_prototypes = None
        if args.top_prototypes:
            try:
                top_prototypes = json.loads(args.top_prototypes)
            except:
                pass

        # 生成解释
        print(f"\n{'='*60}")
        print(f"正在处理: {Path(image_path).name}")
        print(f"{'='*60}")

        result = explainer.generate_explanation(
            original=image_path,
            spatial_heatmap=spatial_path,
            frequency_heatmap=freq_path,
            prediction=args.prediction,
            confidence=args.confidence,
            top_prototypes=top_prototypes,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
        )

        # 保存结果
        image_output_dir = output_dir / image_name
        image_output_dir.mkdir(parents=True, exist_ok=True)

        # 保存JSON
        with open(image_output_dir / 'explanation.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # 保存纯文本
        with open(image_output_dir / 'explanation.txt', 'w', encoding='utf-8') as f:
            f.write(f"图像: {image_path}\n")
            f.write(f"模型: {config.model_path}\n")
            f.write(f"\n自然语言解释:\n{'-'*50}\n")
            f.write(result['explanation'])

        # 打印结果
        print(f"\n{result['explanation']}")
        print(f"\n✓ 结果已保存: {image_output_dir}")

        return result

    # 处理逻辑
    if args.image:
        # 单张图像
        process_single(args.image, args.spatial, args.frequency)

    elif args.image_dir:
        # 批量处理
        image_dir = Path(args.image_dir)
        extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        image_paths = []
        for ext in extensions:
            image_paths.extend(image_dir.glob(f'**/*{ext}'))
            image_paths.extend(image_dir.glob(f'**/*{ext.upper()}'))
        image_paths = sorted(image_paths)

        if not image_paths:
            print(f"在 {args.image_dir} 中未找到图像")
            return

        print(f"\n找到 {len(image_paths)} 张图像")
        results = []

        for idx, image_path in enumerate(image_paths, 1):
            try:
                result = process_single(str(image_path))
                results.append({
                    'image': str(image_path),
                    'explanation': result['explanation'],
                })
                if idx % 5 == 0 or idx == len(image_paths):
                    print(f"\n已处理 {idx}/{len(image_paths)}")
            except Exception as e:
                print(f"处理 {image_path} 时出错: {e}")
                continue

        # 保存汇总
        with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 批量结果已保存: {output_dir}")

    else:
        parser.print_help()
        print("\n错误: 请指定图像路径或图像目录")
        return

    # 卸载模型
    explainer.unload_model()
    print("\n✓ 处理完成")


if __name__ == '__main__':
    main()
