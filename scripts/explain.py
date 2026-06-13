#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LAIGCD 解释脚本
生成空域热力图、频域热力图、原型注意力分析和自然语言解释
"""

import os
import sys
import argparse
import json
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import matplotlib.pyplot as plt

from scripts.inference import get_inference_transform, load_model
from utils import (
    denormalize_image,
    tensor_to_heatmap,
    create_overlay,
    save_rgb_image,
    save_heatmap_image,
    summarize_peaks
)
from models.fakevlm_explainer import create_explainer


def save_prototype_attention_chart(attn_weights, save_path):
    """保存原型注意力柱状图。"""
    attn = attn_weights.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['#FF6B6B' if w > attn.mean() else '#4ECDC4' for w in attn]
    ax.bar(range(len(attn)), attn, color=colors)
    ax.set_xlabel('Prototype Index')
    ax.set_ylabel('Attention Weight')
    ax.set_title('Prototype Attention')
    ax.axhline(attn.mean(), linestyle='--', color='gray', label=f'Mean {attn.mean():.3f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def build_explanation_result(model, image_path, transform, device='cuda'):
    """对单张图片生成解释结果。"""
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    result = model.explain(image_tensor)

    original = denormalize_image(image_tensor)
    spatial_heatmap = tensor_to_heatmap(result['spatial_heatmap'])
    spatial_overlay = create_overlay(original, spatial_heatmap, cmap='jet', alpha=0.45)

    frequency_heatmap = None
    frequency_overlay = None
    if result['frequency_heatmap'] is not None:
        frequency_heatmap = tensor_to_heatmap(result['frequency_heatmap'])
        frequency_overlay = create_overlay(original, frequency_heatmap, cmap='magma', alpha=0.45)

    proto_attn = result['prototype_attention']
    top_prototypes = []
    if proto_attn is not None:
        attn = proto_attn[0]
        topk = min(3, attn.shape[0])
        values, indices = torch.topk(attn, k=topk)
        top_prototypes = [
            {'prototype': int(idx.item()), 'weight': float(val.item())}
            for idx, val in zip(indices, values)
        ]

    fake_probability = float(result['probabilities'][0].item())
    prediction = 'Fake' if int(result['predictions'][0].item()) == 1 else 'Real'

    summary = {
        'path': str(image_path),
        'prediction': prediction,
        'fake_probability': fake_probability,
        'confidence': fake_probability if prediction == 'Fake' else 1 - fake_probability,
        'top_prototypes': top_prototypes,
        'spatial_peak_regions': summarize_peaks(spatial_heatmap, top_k=3),
        'frequency_peak_regions': summarize_peaks(frequency_heatmap, top_k=3) if frequency_heatmap is not None else [],
        'frequency_peak_score': float(frequency_heatmap.max()) if frequency_heatmap is not None else None
    }

    return {
        'summary': summary,
        'original': original,
        'spatial_heatmap': spatial_heatmap,
        'spatial_overlay': spatial_overlay,
        'frequency_heatmap': frequency_heatmap,
        'frequency_overlay': frequency_overlay,
        'prototype_attention': proto_attn[0] if proto_attn is not None else None,
        'image_tensor': image_tensor,
    }


def save_explanation_outputs(explanation, output_dir):
    """保存解释结果到目录。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_rgb_image(explanation['original'], output_dir / 'original.png')
    save_heatmap_image(explanation['spatial_heatmap'], output_dir / 'spatial_heatmap.png', cmap='jet')
    save_rgb_image(explanation['spatial_overlay'], output_dir / 'spatial_overlay.png')

    if explanation['frequency_heatmap'] is not None:
        save_heatmap_image(explanation['frequency_heatmap'], output_dir / 'frequency_heatmap.png', cmap='magma')
        save_rgb_image(explanation['frequency_overlay'], output_dir / 'frequency_overlay.png')

    if explanation['prototype_attention'] is not None:
        save_prototype_attention_chart(
            explanation['prototype_attention'],
            output_dir / 'prototype_attention.png'
        )

    with open(output_dir / 'explanation.json', 'w', encoding='utf-8') as f:
        json.dump(explanation['summary'], f, indent=2, ensure_ascii=False)


def generate_nl_explanation(
    explainer,
    explanation_result,
    output_dir,
    max_new_tokens=256,
):
    """
    使用FakeVLM生成自然语言解释

    Args:
        explainer: FakeVLM解释器实例
        explanation_result: build_explanation_result的输出
        output_dir: 输出目录
        max_new_tokens: 最大生成token数
    """
    summary = explanation_result['summary']

    nl_result = explainer.generate_explanation(
        original=explanation_result['original'],
        spatial_heatmap=explanation_result['spatial_heatmap'],
        frequency_heatmap=explanation_result.get('frequency_heatmap'),
        prediction=summary['prediction'],
        confidence=summary['confidence'],
        top_prototypes=summary['top_prototypes'],
        max_new_tokens=max_new_tokens,
    )

    # 保存自然语言解释
    output_dir = Path(output_dir)
    with open(output_dir / 'nl_explanation.json', 'w', encoding='utf-8') as f:
        json.dump(nl_result, f, indent=2, ensure_ascii=False)

    # 保存纯文本版本
    with open(output_dir / 'nl_explanation.txt', 'w', encoding='utf-8') as f:
        f.write(f"检测结果: {summary['prediction']} (置信度: {summary['confidence']:.1%})\n\n")
        f.write(f"自然语言解释:\n{'-'*50}\n")
        f.write(nl_result['explanation'])

    print(f"\n自然语言解释:\n{'-'*50}")
    print(nl_result['explanation'])

    return nl_result


def collect_image_paths(image_dir):
    """收集目录中的图片路径。"""
    image_dir = Path(image_dir)
    extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    paths = []
    for ext in extensions:
        paths.extend(image_dir.glob(f'**/*{ext}'))
        paths.extend(image_dir.glob(f'**/*{ext.upper()}'))
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser('LAIGCD解释')
    parser.add_argument('image', type=str, nargs='?', help='单张图像路径')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/full_run/best_model.pth',
                        help='模型检查点路径')
    parser.add_argument('--image_dir', type=str, help='图像目录路径')
    parser.add_argument('--output_dir', type=str, default='explanations', help='结果输出目录')
    parser.add_argument('--img_size', type=int, default=224, help='输入图像大小')
    parser.add_argument('--device', type=str, default='cuda', help='设备')

    # FakeVLM相关参数
    parser.add_argument('--use_fakevlm', action='store_true', help='启用FakeVLM自然语言解释')
    parser.add_argument('--fakevlm_model', type=str, default='lingcco/fakeVLM',
                        help='FakeVLM模型路径')
    parser.add_argument('--no_fakevlm_8bit', dest='fakevlm_8bit', action='store_false', default=True,
                        help='禁用8bit量化加载FakeVLM')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='最大生成token数')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载检测器
    model, _, _ = load_model(args.checkpoint, device)
    transform = get_inference_transform(args.img_size)
    output_dir = Path(args.output_dir)

    # 创建FakeVLM解释器（如果需要）
    explainer = None
    if args.use_fakevlm:
        print(f"\n正在初始化 FakeVLM 解释器...")
        explainer = create_explainer(
            model_path=args.fakevlm_model,
            device=args.device,
            load_in_8bit=args.fakevlm_8bit,
        )

    def process_single_image(image_path):
        """处理单张图像的完整流程"""
        explanation = build_explanation_result(model, image_path, transform, device)
        image_output_dir = output_dir / Path(image_path).stem
        save_explanation_outputs(explanation, image_output_dir)

        print(f"\n{'='*60}")
        print(f"图像: {Path(image_path).name}")
        print(f"检测结果: {explanation['summary']['prediction']} "
              f"(置信度: {explanation['summary']['confidence']:.1%})")
        print(f"{'='*60}")
        print(json.dumps(explanation['summary'], ensure_ascii=False, indent=2))

        # FakeVLM自然语言解释
        if explainer is not None:
            print("\n正在生成自然语言解释...")
            generate_nl_explanation(
                explainer, explanation, image_output_dir, args.max_new_tokens
            )

        return explanation

    # 处理逻辑
    if args.image:
        # 单张图像
        process_single_image(args.image)
        print(f"\n✓ 解释结果已保存: {output_dir / Path(args.image).stem}")

    elif args.image_dir:
        # 批量处理
        image_paths = collect_image_paths(args.image_dir)
        if not image_paths:
            print(f"在 {args.image_dir} 中未找到图像")
            return

        print(f"找到 {len(image_paths)} 张图像")
        summaries = []

        for idx, image_path in enumerate(image_paths, start=1):
            try:
                explanation = process_single_image(image_path)
                summaries.append(explanation['summary'])

                if idx % 10 == 0 or idx == len(image_paths):
                    print(f"\n已处理 {idx}/{len(image_paths)}")
            except Exception as e:
                print(f"处理 {image_path} 时出错: {e}")
                continue

        # 保存汇总
        with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)

        print(f"\n✓ 批量解释结果已保存: {output_dir}")

    else:
        parser.print_help()
        print("\n错误: 请指定图像路径或图像目录")

    # 卸载FakeVLM释放显存
    if explainer is not None:
        explainer.unload_model()


if __name__ == '__main__':
    main()
