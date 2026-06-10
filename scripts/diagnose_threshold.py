#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速诊断：找出AP高但Acc低的原因
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from utils.data import get_val_dataloader
from models import build_model

def main():
    print("="*60)
    print("阈值诊断工具")
    print("="*60)

    # 配置
    checkpoint_path = "checkpoints/best_model.pth"
    data_path = "data"
    batch_size = 64

    # 加载模型
    print("\n加载模型...")
    config = {
        'clip_model': 'ViT-B-32',
        'num_prototypes': 16,
        'use_freq': True,
        'freq_type': 'srm',
        'dropout': 0.1
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config).to(device)

    # 加载checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'ema_model_state_dict' in ckpt:
        model.load_state_dict(ckpt['ema_model_state_dict'])
        print("使用EMA模型权重")
    else:
        model.load_state_dict(ckpt['model_state_dict'])

    # 加载验证数据
    print("加载验证数据...")
    val_loader = get_val_dataloader(
        data_path=data_path,
        batch_size=batch_size,
        num_workers=4,
        img_size=224,
        datasets=['140k', '130k'],
        max_samples=None,
        subset_mode='balanced'
    )

    # 收集预测
    print("收集预测结果...")
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, dict):
                images = batch['image']
                labels = batch['label']
            else:
                images, labels = batch

            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu()

            all_probs.append(probs)
            all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs).numpy().squeeze()
    all_labels = torch.cat(all_labels).numpy()

    print(f"\n收集了 {len(all_labels)} 个样本")

    # 数据分布统计
    print("\n" + "="*60)
    print("数据分布")
    print("="*60)
    real_count = (all_labels == 0).sum()
    fake_count = (all_labels == 1).sum()
    print(f"真实样本: {real_count}")
    print(f"伪造样本: {fake_count}")
    print(f"比例: {real_count}:{fake_count} = {real_count/fake_count:.2f}:1")

    # 概率分布统计
    print("\n" + "="*60)
    print("概率分布统计")
    print("="*60)

    real_probs = all_probs[all_labels == 0]
    fake_probs = all_probs[all_labels == 1]

    print(f"\n真实图片的预测概率:")
    print(f"  最小值: {real_probs.min():.4f}")
    print(f"  25%分位: {np.percentile(real_probs, 25):.4f}")
    print(f"  中位数: {np.median(real_probs):.4f}")
    print(f"  75%分位: {np.percentile(real_probs, 75):.4f}")
    print(f"  最大值: {real_probs.max():.4f}")
    print(f"  平均值: {real_probs.mean():.4f}")

    print(f"\n伪造图片的预测概率:")
    print(f"  最小值: {fake_probs.min():.4f}")
    print(f"  25%分位: {np.percentile(fake_probs, 25):.4f}")
    print(f"  中位数: {np.median(fake_probs):.4f}")
    print(f"  75%分位: {np.percentile(fake_probs, 75):.4f}")
    print(f"  最大值: {fake_probs.max():.4f}")
    print(f"  平均值: {fake_probs.mean():.4f}")

    # 阈值扫描
    print("\n" + "="*60)
    print("不同阈值下的性能")
    print("="*60)

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print(f"{'阈值':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 50)

    best_threshold = 0.5
    best_acc = 0

    for th in thresholds:
        preds = (all_probs >= th).astype(int)
        acc = accuracy_score(all_labels, preds)
        prec = precision_score(all_labels, preds, zero_division=0)
        rec = recall_score(all_labels, preds, zero_division=0)
        f1 = f1_score(all_labels, preds, zero_division=0)

        print(f"{th:<8.2f} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f}")

        if acc > best_acc:
            best_acc = acc
            best_threshold = th

    # 精细搜索最优阈值
    print("\n精细搜索最优阈值...")
    best_f1 = 0
    best_th_f1 = 0.5

    for th in np.arange(0.01, 0.99, 0.01):
        preds = (all_probs >= th).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th_f1 = th

    print(f"\n最优阈值（按F1优化）: {best_th_f1:.2f}")
    print(f"  对应F1: {best_f1:.4f}")
    print(f"  对应Accuracy: {accuracy_score(all_labels, (all_probs >= best_th_f1).astype(int)):.4f}")

    # 计算AP
    from sklearn.metrics import average_precision_score
    ap = average_precision_score(all_labels, all_probs)

    print("\n" + "="*60)
    print("总结")
    print("="*60)
    print(f"AP（平均精度）: {ap:.4f}  ← 模型排序能力（很棒！）")
    print(f"默认阈值0.5的Accuracy: {accuracy_score(all_labels, (all_probs >= 0.5).astype(int)):.4f}")
    print(f"最优阈值{best_th_f1:.2f}的Accuracy: {accuracy_score(all_labels, (all_probs >= best_th_f1).astype(int)):.4f}")

    print("\n💡 结论:")
    if ap > 0.9 and best_acc > 0.8:
        print("  模型很强！只是默认阈值0.5不是最优的。")
        print(f"  建议将阈值改为 {best_th_f1:.2f} 以获得最佳性能。")
    elif ap > 0.9 and best_acc < 0.7:
        print("  模型排序能力强，但准确率低。")
        print("  可能原因：验证集不平衡，需要调整决策阈值。")
    else:
        print("  模型可能需要进一步训练。")

if __name__ == "__main__":
    main()
