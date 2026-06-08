"""
训练和验证函数
"""

import time
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from timm.utils import ModelEmaV2
from collections import defaultdict
import numpy as np


class MetricLogger:
    """训练指标记录器"""
    def __init__(self, delimiter=" "):
        self.meters = defaultdict(list)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.meters[k].append(v)

    def add_meter(self, name, fmt=":f"):
        pass  # 兼容接口

    def log_every(self, iterable, print_freq, header):
        """日志输出"""
        for i, obj in enumerate(iterable):
            yield obj
            if i % print_freq == 0 or i == len(iterable) - 1:
                parts = [header]
                for k, v in self.meters.items():
                    if len(v) > 0:
                        parts.append(f"{k}: {v[-1]:.4f}")
                print(self.delimiter.join(parts))

    def synchronize_between_processes(self):
        """进程间同步（分布式训练用）"""
        pass

    def __str__(self):
        parts = []
        for k, v in self.meters.items():
            if len(v) > 0:
                avg = np.mean(v)
                parts.append(f"{k}: {avg:.4f}")
        return " ".join(parts)


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    epoch,
    config,
    ema_model=None
):
    """
    训练一个epoch

    Args:
        model: 模型
        dataloader: 数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 设备
        epoch: 当前epoch
        config: 配置字典
        ema_model: EMA模型（可选）

    Returns:
        metric_logger: 指标记录器
    """
    model.train()

    # 混合精度训练
    scaler = GradScaler('cuda') if config.get('use_amp', True) else None

    # 梯度累积
    accumulation_steps = config.get('accumulation_steps', 1)
    optimizer.zero_grad()

    # 指标记录
    metric_logger = MetricLogger(delimiter="  ")
    print_freq = config.get('print_freq', 50)

    header = f'Epoch: [{epoch}]'
    start_time = time.time()

    for i, batch in enumerate(metric_logger.log_every(dataloader, print_freq, header)):
        # 处理字典格式的数据
        if isinstance(batch, dict):
            images = batch['image']
            labels = batch['label']
        else:
            images, labels = batch

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 前向传播（混合精度）
        if scaler is not None:
            with autocast(device_type='cuda'):
                logits = model(images)
                loss = model.get_criterion(logits, labels)
                loss = loss / accumulation_steps  # 梯度累积
        else:
            logits = model(images)
            loss = model.get_criterion(logits, labels)
            loss = loss / accumulation_steps

        # 反向传播
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 梯度累积步数到达时更新
        if (i + 1) % accumulation_steps == 0:
            # 梯度裁剪
            if scaler is not None:
                scaler.unscale_(optimizer)
            if config.get('clip_grad_norm', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=config['clip_grad_norm']
                )

            # 更新参数
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

            # EMA更新
            if ema_model is not None:
                ema_model.update(model)

        # 记录指标
        metric_logger.update(
            loss=loss.item() * accumulation_steps,
            lr=optimizer.param_groups[0]['lr']
        )

    # 更新学习率
    if scheduler is not None:
        scheduler.step()

    # 打印统计
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch} 训练时间: {epoch_time:.2f}s")
    print(f"Averaged stats: {metric_logger}")

    return metric_logger


@torch.no_grad()
def validate(model, dataloader, device):
    """
    验证/测试

    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备

    Returns:
        metrics: 指标字典
    """
    model.eval()

    all_logits = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        # 处理字典格式的数据
        if isinstance(batch, dict):
            images = batch['image']
            labels = batch['label']
        else:
            images, labels = batch

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = model.get_criterion(logits, labels)

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        total_loss += loss.item()
        num_batches += 1

    # 合并所有批次
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # 计算指标
    probs = torch.sigmoid(all_logits).numpy()
    preds = (probs > 0.5).astype(int)
    labels_np = all_labels.numpy()

    # 基础指标
    accuracy = (preds == labels_np).mean()

    # 计算AP
    from sklearn.metrics import average_precision_score
    ap = average_precision_score(labels_np, probs)

    metrics = {
        'loss': total_loss / num_batches,
        'accuracy': accuracy,
        'ap': ap,
        'predictions': preds,
        'probabilities': probs,
        'labels': labels_np
    }

    # 详细指标
    if len(np.unique(labels_np)) > 1:
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

        try:
            metrics.update({
                'precision': precision_score(labels_np, preds, zero_division=0),
                'recall': recall_score(labels_np, preds, zero_division=0),
                'f1': f1_score(labels_np, preds, zero_division=0),
                'auc': roc_auc_score(labels_np, probs)
            })
        except:
            pass

    return metrics


def print_metrics(metrics, prefix=""):
    """打印评估指标"""
    print(f"\n{prefix}评估结果:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  AP: {metrics['ap']:.4f}")

    if 'precision' in metrics:
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, filepath, ema_model=None):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'metrics': metrics,
    }

    if ema_model is not None:
        checkpoint['ema_model_state_dict'] = ema_model.module.state_dict()

    torch.save(checkpoint, filepath)
    print(f"检查点已保存: {filepath}")


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, ema_model=None):
    """加载检查点"""
    checkpoint = torch.load(filepath, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if ema_model is not None and 'ema_model_state_dict' in checkpoint:
        ema_model.module.load_state_dict(checkpoint['ema_model_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})

    print(f"检查点已加载: {filepath} (epoch {epoch})")

    return epoch, metrics


def test_training():
    """测试训练流程"""
    print("测试训练流程...")

    # 创建简单模型和数据
    from models import build_model
    from utils.data import SimpleDataset
    from torch.utils.data import DataLoader

    config = {
        'clip_model': 'ViT-B-32',
        'num_prototypes': 8,
        'use_freq': False,  # 测试时关闭频域模块
        'use_amp': True,
        'accumulation_steps': 2,
        'print_freq': 10
    }

    try:
        model = build_model(config)
    except:
        print("  (跳过模型测试，需要open-clip-torch)")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 创建数据
    dataset = SimpleDataset(num_samples=32)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # 测试训练一个epoch
    train_one_epoch(
        model, dataloader, optimizer, scheduler,
        device, epoch=0, config=config
    )

    print("✓ 训练流程测试通过")


if __name__ == "__main__":
    test_training()
