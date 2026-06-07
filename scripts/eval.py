#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LAIGCD 评估脚本
在测试集上评估模型性能
"""

import os
import sys
import argparse
import json
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from models import build_model
from utils import get_val_dataloader, load_checkpoint
from utils.metrics import compute_metrics, print_metrics, compute_per_generator_metrics, print_per_generator_metrics
from utils.viz import plot_roc_curve, plot_confusion_matrix, confusion_matrix


def main():
    parser = argparse.ArgumentParser('LAIGCD评估')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--data_path', type=str, required=True, help='数据集根目录')
    parser.add_argument('--split', type=str, default='val', help='评估集名称 (val/test)')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--img_size', type=int, default=224, help='输入图像大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--output_dir', type=str, default=None, help='结果输出目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备')

    args = parser.parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(args.checkpoint).parent / 'eval_results'

    # 加载模型
    print(f"加载模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # 加载配置
    config_path = Path(args.checkpoint).parent / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'clip_model': 'ViT-B-32',
            'num_prototypes': 16,
            'use_freq': True,
            'freq_type': 'srm',
            'dropout': 0.1
        }

    model = build_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"模型加载完成")

    # 创建数据加载器
    dataloader = get_val_dataloader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        split=args.split
    )

    print(f"评估集: {len(dataloader.dataset)} 样本")

    # 评估
    print("\n开始评估...")
    all_probs = []
    all_labels = []
    all_paths = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits = model(images)
            probs = torch.sigmoid(logits).squeeze(1)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # 计算指标
    metrics = compute_metrics(all_labels, all_probs)
    print_metrics(metrics, prefix=args.split)

    # 保存结果
    results = {
        'checkpoint': str(args.checkpoint),
        'split': args.split,
        'num_samples': len(all_labels),
        'metrics': {k: float(v) if isinstance(v, (int, float)) else str(v)
                   for k, v in metrics.items() if k not in ['predictions', 'labels', 'probabilities']}
    }

    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    # 绘制ROC曲线
    plot_roc_curve(all_labels, all_probs, save_path=output_dir / 'roc_curve.png')

    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, (all_probs > 0.5).astype(int))
    plot_confusion_matrix(cm, ['Real', 'Fake'], save_path=output_dir / 'confusion_matrix.png')

    print(f"\n结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
