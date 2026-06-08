#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LAIGCD 训练脚本
"""

import os
import sys
import argparse
import time
import logging
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from timm.utils import ModelEmaV2


def setup_logger(output_dir, console_level=logging.INFO):
    """
    设置日志系统

    - 控制台：只输出关键信息 (INFO级别)
    - 文件：输出详细日志 (DEBUG级别)
    """
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 日志文件名（带时间戳）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = output_dir / f'train_{timestamp}.log'

    # 创建logger
    logger = logging.getLogger('LAIGCD')
    logger.setLevel(logging.DEBUG)

    # 清除已有的处理器
    logger.handlers.clear()

    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 文件处理器：记录所有详细日志
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台处理器：只显示重要信息
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"日志文件: {log_file}")
    return logger

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
    parser.add_argument('--data_path', type=str, default='data', help='数据集根目录')
    parser.add_argument('--img_size', type=int, default=224, help='输入图像大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--datasets', type=str, default='140k,130k',
                        help='使用的数据集，逗号分隔 (如: 140k,130k 或 140k 或 130k)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='小规模测试模式：限制样本数 (None=全部, train推荐1000/10000)')
    parser.add_argument('--subset_mode', type=str, default='balanced', choices=['balanced', 'random'],
                        help='小规模采样模式: balanced-保持类别平衡, random-随机采样')

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
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志系统
    logger = setup_logger(output_dir)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 保存配置
    config = vars(args)
    import json
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # 记录配置（简化输出）
    logger.info("="*50)
    logger.info("训练配置:")
    key_configs = ['max_samples', 'epochs', 'batch_size', 'lr', 'use_freq']
    for k in key_configs:
        if k in config and config[k] is not None:
            logger.info(f"  {k}: {config[k]}")
    logger.debug("完整配置:")
    for k, v in config.items():
        logger.debug(f"  {k}: {v}")
    logger.info("="*50)

    # 构建模型
    logger.info("构建模型...")
    model = build_model(config)
    model = model.to(device)

    # 记录模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.debug(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
    logger.info(f"可训练参数: {trainable_params:,}")

    # 创建数据加载器
    logger.info("准备数据...")
    datasets_list = args.datasets.split(',') if args.datasets else None

    train_loader = get_train_dataloader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        datasets=datasets_list,
        max_samples=args.max_samples,
        subset_mode=args.subset_mode
    )

    val_loader = get_val_dataloader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        datasets=datasets_list,
        max_samples=args.max_samples // 5 if args.max_samples else None,  # val约为train的1/5
        subset_mode=args.subset_mode
    )

    logger.info(f"训练集: {len(train_loader.dataset)} 样本")
    logger.info(f"验证集: {len(val_loader.dataset)} 样本")

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
        logger.info(f"从检查点恢复: {args.resume}")
        start_epoch, ckpt_metrics = load_checkpoint(
            args.resume, model, optimizer, scheduler, ema_model
        )
        start_epoch += 1
        best_ap = ckpt_metrics.get('ap', 0.0)

    # 仅评估模式
    if args.eval_only:
        logger.info("评估模式...")
        model_to_eval = ema_model.module if ema_model else model
        metrics = validate(model_to_eval, val_loader, device)
        logger.info(f"验证准确率: {metrics['accuracy']:.4f}, AP: {metrics['ap']:.4f}")
        return

    # 训练循环
    logger.info("开始训练...")
    train_losses = []
    val_metrics_history = []

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        logger.info(f"Epoch {epoch+1}/{args.epochs} 开始")

        # 训练一个epoch
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            device, epoch, config, ema_model, logger=logger
        )
        train_losses.append(train_metrics.meters['loss'][-1])

        # 验证
        model_to_eval = ema_model.module if ema_model else model
        val_metrics = validate(model_to_eval, val_loader, device, logger=logger)
        val_metrics_history.append(val_metrics)

        # 输出结果（简化）
        logger.info(f"Epoch {epoch+1} | Val Loss: {val_metrics['loss']:.4f} | "
                   f"Acc: {val_metrics['accuracy']:.4f} | AP: {val_metrics['ap']:.4f}")

        epoch_time = time.time() - epoch_start
        logger.debug(f"Epoch {epoch+1} 耗时: {epoch_time:.2f}s")

        # 保存最佳模型
        if val_metrics['ap'] > best_ap:
            best_ap = val_metrics['ap']
            save_path = output_dir / 'best_model.pth'
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                save_path, ema_model
            )
            logger.info(f"★ 保存最佳模型 (AP: {best_ap:.4f})")

        # 定期保存
        if (epoch + 1) % args.save_freq == 0:
            save_path = output_dir / f'checkpoint_epoch_{epoch+1}.pth'
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                save_path, ema_model
            )
            logger.debug(f"保存检查点: {save_path}")

    # 训练结束，保存最终结果
    logger.info("="*50)
    logger.info("训练完成！")
    logger.info(f"最佳验证AP: {best_ap:.4f}")
    logger.info(f"模型保存位置: {output_dir}/best_model.pth")
    logger.info("="*50)

    # 绘制训练曲线
    if len(train_losses) > 0:
        plot_training_curves(
            train_losses, val_metrics_history,
            save_path=output_dir / 'training_curves.png'
        )
        logger.info(f"训练曲线已保存: {output_dir}/training_curves.png")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    # 设置中文显示（可选）
    import matplotlib as mpl
    mpl.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    mpl.rcParams['axes.unicode_minus'] = False

    main(args)
