#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LAIGCD 训练结果可视化脚本
从训练checkpoint中提取历史数据并生成可视化图表
"""

import os
import sys
import argparse
import json
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from utils.viz import plot_training_curves, plot_confusion_matrix, plot_roc_curve


def extract_training_history(checkpoint_dir):
    """
    从checkpoint目录中提取训练历史

    Args:
        checkpoint_dir: checkpoint目录路径

    Returns:
        history: 包含训练历史的字典
    """
    checkpoint_dir = Path(checkpoint_dir)

    # 尝试从config.json读取配置
    config_file = checkpoint_dir / 'config.json'
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)

    # 尝试从各个checkpoint中提取历史
    history = {
        'train_losses': [],
        'val_metrics': []
    }

    # 查找所有checkpoint文件
    ckpt_files = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pth'), key=lambda x: int(x.stem.split('_')[-1]))

    # 加载每个checkpoint的指标
    for ckpt_file in ckpt_files:
        try:
            ckpt = torch.load(ckpt_file, map_location='cpu')
            if 'metrics' in ckpt:
                history['val_metrics'].append(ckpt['metrics'])
        except Exception as e:
            print(f"Warning: 无法加载 {ckpt_file}: {e}")

    # 尝试从best_model.pth获取最佳指标
    best_ckpt = checkpoint_dir / 'best_model.pth'
    if best_ckpt.exists():
        try:
            best_data = torch.load(best_ckpt, map_location='cpu')
            if 'metrics' in best_data:
                print(f"最佳模型指标: AP={best_data['metrics'].get('ap', 0):.4f}, "
                      f"Acc={best_data['metrics'].get('accuracy', 0):.4f}")
        except Exception as e:
            print(f"Warning: 无法加载最佳模型: {e}")

    return history


def plot_training_history(checkpoint_dir, output_dir=None):
    """
    绘制训练历史曲线

    Args:
        checkpoint_dir: checkpoint目录路径
        output_dir: 输出目录，默认为checkpoint_dir
    """
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir) if output_dir else checkpoint_dir

    print("提取训练历史...")
    history = extract_training_history(checkpoint_dir)

    if len(history['val_metrics']) == 0:
        print("未找到验证指标，无法绘制曲线")
        return

    # 重建train_losses（简化处理，使用val_loss近似）
    # 在实际训练中应该保存完整的loss历史
    val_losses = [m.get('loss', 0) for m in history['val_metrics']]
    history['train_losses'] = val_losses  # 临时替代

    print(f"找到 {len(history['val_metrics']) 个epoch的数据")

    # 绘制训练曲线
    plot_training_curves(
        history['train_losses'],
        history['val_metrics'],
        save_path=output_dir / 'training_curves.png'
    )

    print(f"训练曲线已保存: {output_dir / 'training_curves.png'}")


def plot_checkpoint_metrics(checkpoint_dir):
    """
    绘制各epoch的指标对比

    Args:
        checkpoint_dir: checkpoint目录路径
    """
    checkpoint_dir = Path(checkpoint_dir)

    # 收集所有checkpoint的指标
    epochs = []
    accuracies = []
    aps = []
    losses = []

    ckpt_files = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pth'),
                        key=lambda x: int(x.stem.split('_')[-1]))

    for ckpt_file in ckpt_files:
        try:
            ckpt = torch.load(ckpt_file, map_location='cpu')
            if 'metrics' in ckpt:
                epoch = ckpt.get('epoch', 0)
                epochs.append(epoch)
                accuracies.append(ckpt['metrics'].get('accuracy', 0))
                aps.append(ckpt['metrics'].get('ap', 0))
                losses.append(ckpt['metrics'].get('loss', 0))
        except Exception as e:
            continue

    if not epochs:
        print("未找到足够的指标数据")
        return

    # 绘制指标对比图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, losses, 'b-o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('验证损失')
    axes[0].grid(True)

    axes[1].plot(epochs, accuracies, 'g-o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('验证准确率')
    axes[1].grid(True)

    axes[2].plot(epochs, aps, 'r-o')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Average Precision')
    axes[2].set_title('验证AP')
    axes[2].grid(True)

    plt.tight_layout()
    save_path = checkpoint_dir / 'metrics_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"指标对比图已保存: {save_path}")
    plt.close()


def print_summary(checkpoint_dir):
    """
    打印训练结果摘要

    Args:
        checkpoint_dir: checkpoint目录路径
    """
    checkpoint_dir = Path(checkpoint_dir)

    print("\n" + "="*60)
    print("训练结果摘要")
    print("="*60)

    # 读取配置
    config_file = checkpoint_dir / 'config.json'
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        print("\n训练配置:")
        for k in ['max_samples', 'epochs', 'batch_size', 'lr', 'use_freq']:
            if k in config:
                print(f"  {k}: {config[k]}")

    # 统计checkpoint文件
    ckpt_files = list(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
    best_ckpt = checkpoint_dir / 'best_model.pth'

    print(f"\n保存的checkpoint: {len(ckpt_files)} 个")
    print(f"最佳模型: {'存在' if best_ckpt.exists() else '不存在'}")

    # 显示最佳指标
    if best_ckpt.exists():
        try:
            best_data = torch.load(best_ckpt, map_location='cpu')
            if 'metrics' in best_data:
                print(f"\n最佳验证指标:")
                print(f"  Loss: {best_data['metrics'].get('loss', 0):.4f}")
                print(f"  Accuracy: {best_data['metrics'].get('accuracy', 0):.4f}")
                print(f"  AP: {best_data['metrics'].get('ap', 0):.4f}")
        except Exception:
            pass

    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser('LAIGCD结果可视化')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='checkpoint目录路径')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（默认与checkpoint_dir相同）')
    parser.add_argument('--skip_curves', action='store_true',
                        help='跳过训练曲线绘制')
    parser.add_argument('--skip_comparison', action='store_true',
                        help='跳过指标对比图')

    args = parser.parse_args()

    # 打印摘要
    print_summary(args.checkpoint_dir)

    # 绘制训练曲线
    if not args.skip_curves:
        plot_training_history(args.checkpoint_dir, args.output_dir)

    # 绘制指标对比
    if not args.skip_comparison:
        plot_checkpoint_metrics(args.checkpoint_dir)


if __name__ == '__main__':
    main()
