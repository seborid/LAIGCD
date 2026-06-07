#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LAIGCD 推理脚本
支持单图和批量推理
"""

import os
import sys
import argparse
import json
from pathlib import Path
from PIL import Image

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torchvision import transforms

from models import build_model


def get_inference_transform(img_size=224):
    """获取推理数据变换"""
    return transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_model(checkpoint_path, device='cuda'):
    """加载模型"""
    print(f"加载模型: {checkpoint_path}")

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 加载配置
    config_path = Path(checkpoint_path).parent / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # 默认配置
        config = {
            'clip_model': 'ViT-B-32',
            'num_prototypes': 16,
            'use_freq': True,
            'freq_type': 'srm',
            'dropout': 0.1
        }

    # 构建模型
    model = build_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"模型加载完成 (epoch {checkpoint.get('epoch', '?')})")
    if 'metrics' in checkpoint:
        print(f"检查点指标: AP={checkpoint['metrics'].get('ap', 0):.4f}")

    return model, config


def predict_single_image(model, image_path, transform, device='cuda'):
    """预测单张图像"""
    # 加载图像
    image = Image.open(image_path).convert('RGB')

    # 转换
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 预测
    with torch.no_grad():
        probs, preds = model.predict(input_tensor)

    prob = probs[0].item()
    pred = preds[0].item()

    return {
        'path': str(image_path),
        'label': 'Fake' if pred == 1 else 'Real',
        'confidence': prob if pred == 1 else 1 - prob,
        'fake_probability': prob
    }


def predict_directory(model, image_dir, transform, device='cuda', output_path=None):
    """预测目录中的所有图像"""
    image_dir = Path(image_dir)

    # 支持的图像格式
    extensions = {'.jpg', '.jpeg', '.png', '.webp'}

    # 收集所有图像
    image_paths = []
    for ext in extensions:
        image_paths.extend(image_dir.glob(f'**/*{ext}'))
        image_paths.extend(image_dir.glob(f'**/*{ext.upper()}'))

    if len(image_paths) == 0:
        print(f"在 {image_dir} 中未找到图像")
        return []

    print(f"找到 {len(image_paths)} 张图像")

    # 批量预测
    results = []
    for i, img_path in enumerate(image_paths):
        result = predict_single_image(model, img_path, transform, device)
        results.append(result)

        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1}/{len(image_paths)}")

    # 统计
    fake_count = sum(1 for r in results if r['label'] == 'Fake')
    real_count = len(results) - fake_count

    print(f"\n批量推理结果:")
    print(f"  总计: {len(results)}")
    print(f"  真实: {real_count} ({real_count/len(results)*100:.1f}%)")
    print(f"  伪造: {fake_count} ({fake_count/len(results)*100:.1f}%)")

    # 保存结果
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"结果已保存: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser('LAIGCD推理')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--image', type=str, help='单张图像路径')
    parser.add_argument('--image_dir', type=str, help='图像目录路径')
    parser.add_argument('--output', type=str, help='输出结果文件路径')
    parser.add_argument('--img_size', type=int, default=224, help='输入图像大小')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--show_attention', action='store_true', help='显示原型注意力')

    args = parser.parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    model, config = load_model(args.checkpoint, device)
    transform = get_inference_transform(args.img_size)

    # 单图推理
    if args.image:
        result = predict_single_image(model, args.image, transform, device)

        print("\n" + "="*50)
        print(f"图像: {result['path']}")
        print(f"预测: {result['label']}")
        print(f"置信度: {result['confidence']:.2%}")
        print(f"伪造概率: {result['fake_probability']:.4f}")
        print("="*50)

        # 显示原型注意力
        if args.show_attention:
            from utils.viz import visualize_prototype_attention
            image = Image.open(args.image).convert('RGB')
            image_tensor = transform(image)
            output_path = Path(args.image).stem + '_attention.png'
            visualize_prototype_attention(model, image_tensor, output_path, device)

    # 目录推理
    elif args.image_dir:
        results = predict_directory(
            model, args.image_dir, transform, device,
            output_path=args.output
        )

        # 打印前10个结果
        print("\n前10个结果:")
        for r in results[:10]:
            print(f"  {r['path']}: {r['label']} ({r['confidence']:.2%})")

    else:
        parser.print_help()
        print("\n错误: 请指定 --image 或 --image_dir")


if __name__ == "__main__":
    main()
