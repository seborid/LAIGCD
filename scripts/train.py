#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LAIGCD 训练脚本
"""

import os
import sys
import argparse
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from timm.utils import ModelEmaV2

from models import build_model
from utils import (
    get_train_dataloader, get_val_dataloader,
    train_one_epoch, validate, save_checkpoint, load_checkpoint
)
from utils.metrics import compute_metrics, print_metrics
from utils.viz import plot_training_curves


def get_args_parser():
    """解析命令行参数"""
    parser = argparse.ArgumentParser('LAIGCD训练', add_help=False)

    # 数据集
    parser.add_argument('--data_path', type=str, required=True, help='数据集根目录')
    parser.add_argument('--img_size', type=int, default=224, help='输入图像大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')

    # 模型
    parser.add_argument('--clip_model', type=str, default='ViT-B-32', help='CLIP模型')
    parser.add_argument('--num_prototypes', type=int, default=16, help='原型数量')
    parser.add_argument('--use_freq', action='store_true', help='使用频域特征')
    parser.add_argument('--no_freq', dest='use_freq', action='store_false')
    parser.set_defaults(use_freq=True)
    parser.add_argument('--freq_type', type=str, default='srm', choices=['srm', 'dct'])
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout概率')

    # 训练
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--accumulation_steps', type=int, default=2, help='梯度累积步数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='warmup轮数')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='梯度裁剪')

    # 优化
    parser.add_argument('--use_amp', action='store_true', default=True, help='使用混合精度')
    parser.add_argument('--use_ema', action='store_true', default=True, help='使用EMA')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='EMA衰减率')
    parser.add_argument('--no_amp', dest='use_amp', action='store_false')
    parser.add_argument('--no_ema', dest='use_ema', action='store_false')

    # 输出
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='输出目录')
    parser.add_argument('--print_freq', type=int, default=50, help='日志打印频率')
    parser.add_argument('--save_freq', type=int, default=5, help='保存频率')

    # 恢复
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--eval_only', action='store_true', help='仅评估')

    # 设备
    parser.add_argument('--device', type=str, default='cuda', help='设备')

    return parser


def main(args):
    """主训练函数"""
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    config = vars(args)
    import json
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # 打印配置
    print("\n" + "="*50)
    print("训练配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("="*50 + "\n")

    # 构建模型
    print("构建模型...")
    model = build_model(config)
    model = model.to(device)
    model.print_model_info()

    # 创建数据加载器
    print("准备数据...")
    train_loader = get_train_dataloader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size
    )

    val_loader = get_val_dataloader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        split='val'
    )

    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本\n")

    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    # EMA
    ema_model = None
    if args.use_ema:
        ema_model = ModelEmaV2(model, decay=args.ema_decay)

    # 恢复训练
    start_epoch = 0
    best_ap = 0.0

    if args.resume:
        print(f"从检查点恢复: {args.resume}")
        start_epoch, ckpt_metrics = load_checkpoint(
            args.resume, model, optimizer, scheduler, ema_model
        )
        start_epoch += 1
        best_ap = ckpt_metrics.get('ap', 0.0)

    # 仅评估模式
    if args.eval_only:
        print("\n评估模式...")
        model_to_eval = ema_model.module if ema_model else model
        metrics = validate(model_to_eval, val_loader, device)
        print_metrics(metrics, prefix="验证集")
        return

    # 训练循环
    print("开始训练...\n")
    train_losses = []
    val_metrics_history = []

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # 训练一个epoch
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            device, epoch, config, ema_model
        )
        train_losses.append(train_metrics.meters['loss'][-1])

        # 验证
        model_to_eval = ema_model.module if ema_model else model
        val_metrics = validate(model_to_eval, val_loader, device)
        val_metrics_history.append(val_metrics)

        # 打印结果
        print_metrics(val_metrics, prefix=f"Epoch {epoch}")

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch} 总耗时: {epoch_time:.2f}s\n")

        # 保存最佳模型
        if val_metrics['ap'] > best_ap:
            best_ap = val_metrics['ap']
            save_path = output_dir / 'best_model.pth'
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                save_path, ema_model
            )
            print(f"保存最佳模型 (AP: {best_ap:.4f})")

        # 定期保存
        if (epoch + 1) % args.save_freq == 0:
            save_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                save_path, ema_model
            )

    # 训练结束，保存最终结果
    print("\n训练完成！")
    print(f"最佳验证AP: {best_ap:.4f}")

    # 绘制训练曲线
    if len(train_losses) > 0:
        plot_training_curves(
            train_losses, val_metrics_history,
            save_path=output_dir / 'training_curves.png'
        )


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    # 设置中文显示（可选）
    import matplotlib as mpl
    mpl.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    mpl.rcParams['axes.unicode_minus'] = False

    main(args)
