"""
可视化工具
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    绘制混淆矩阵

    Args:
        cm: 混淆矩阵
        class_names: 类别名称列表
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax, cbar_kws={'label': '样本数'}
    )

    ax.set_xlabel('预测标签', fontsize=12)
    ax.set_ylabel('真实标签', fontsize=12)
    ax.set_title('混淆矩阵', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"混淆矩阵已保存: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_roc_curve(labels, probs, save_path=None):
    """
    绘制ROC曲线

    Args:
        labels: 真实标签
        probs: 预测概率
        save_path: 保存路径
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假正率 (FPR)', fontsize=12)
    ax.set_ylabel('真正率 (TPR)', fontsize=12)
    ax.set_title('ROC曲线', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC曲线已保存: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_prototype_attention(model, image, save_path, device='cuda'):
    """
    可视化原型注意力权重

    Args:
        model: 检测器模型
        image: 输入图像 [C, H, W] 或 [B, C, H, W]
        save_path: 保存路径
        device: 设备
    """
    model.eval()

    # 确保图像是batch格式
    if image.dim() == 3:
        image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():
        # 获取注意力权重
        attn_weights = model.get_attention_weights(image)  # [B, num_prototypes]
        attn_weights = attn_weights[0].cpu().numpy()

    # 获取预测
    probs, preds = model.predict(image)
    prob = probs[0].cpu().item()
    pred = preds[0].cpu().item()

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 显示原图
    img_np = image[0].cpu().permute(1, 2, 0).numpy()
    # 反归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)

    ax1.imshow(img_np)
    pred_label = "Fake" if pred == 1 else "Real"
    ax1.set_title(f'输入图像\n预测: {pred_label} (置信度: {prob:.2%})',
                 fontsize=12, fontweight='bold')
    ax1.axis('off')

    # 显示注意力权重
    num_prototypes = len(attn_weights)
    colors = ['#FF6B6B' if w > attn_weights.mean() else '#4ECDC4' for w in attn_weights]

    bars = ax2.bar(range(num_prototypes), attn_weights, color=colors)
    ax2.set_xlabel('原型索引', fontsize=12)
    ax2.set_ylabel('注意力权重', fontsize=12)
    ax2.set_title('原型注意力分布', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(attn_weights) * 1.1)
    ax2.axhline(y=attn_weights.mean(), color='gray', linestyle='--',
                label=f'平均值 ({attn_weights.mean():.3f})')
    ax2.legend(fontsize=10)

    # 添加数值标签
    for i, (bar, weight) in enumerate(zip(bars, attn_weights)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{weight:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"原型注意力已保存: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_training_curves(train_losses, val_metrics, save_path=None):
    """
    绘制训练曲线

    Args:
        train_losses: 训练损失列表
        val_metrics: 验证指标列表，每个元素是包含epoch、loss、accuracy、ap的字典
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = range(1, len(train_losses) + 1)

    # 训练损失
    axes[0].plot(epochs, train_losses, 'b-o', label='训练损失')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('训练损失')
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # 验证损失
    val_losses = [m['loss'] for m in val_metrics]
    axes[1].plot(epochs, val_losses, 'r-o', label='验证损失')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('验证损失')
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    # 验证准确率和AP
    ax2_twin = axes[2].twinx()

    axes[2].plot(epochs, [m['accuracy'] for m in val_metrics], 'g-o', label='准确率')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('准确率', color='g')
    axes[2].tick_params(axis='y', labelcolor='g')
    axes[2].set_title('验证指标')
    axes[2].grid(alpha=0.3)

    ax2_twin.plot(epochs, [m['ap'] for m in val_metrics], 'b-s', label='AP')
    ax2_twin.set_ylabel('AP', color='b')
    ax2_twin.tick_params(axis='y', labelcolor='b')

    # 合并图例
    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    axes[2].legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练曲线已保存: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_prototypes_embeddings(features, labels, prototypes, save_path=None):
    """
    可视化原型和特征的嵌入（使用t-SNE降维）

    Args:
        features: 样本特征 [N, D]
        labels: 样本标签 [N]
        prototypes: 原型向量 [P, D]
        save_path: 保存路径
    """
    from sklearn.manifold import TSNE

    # 合并特征和原型
    all_features = np.vstack([features, prototypes])
    all_labels = np.concatenate([labels, -np.ones(len(prototypes))])  # -1表示原型

    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    embeddings = tsne.fit_transform(all_features)

    # 分离
    n_samples = len(features)
    sample_emb = embeddings[:n_samples]
    proto_emb = embeddings[n_samples:]

    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制样本点
    real_mask = labels == 0
    fake_mask = labels == 1

    ax.scatter(sample_emb[real_mask, 0], sample_emb[real_mask, 1],
              c='green', alpha=0.5, s=30, label='真实图像')
    ax.scatter(sample_emb[fake_mask, 0], sample_emb[fake_mask, 1],
              c='red', alpha=0.5, s=30, label='伪造图像')

    # 绘制原型
    ax.scatter(proto_emb[:, 0], proto_emb[:, 1],
              c='blue', marker='*', s=200, edgecolors='black',
              linewidths=1.5, label='原型', zorder=10)

    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title('特征嵌入可视化', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"嵌入可视化已保存: {save_path}")
    else:
        plt.show()

    plt.close()


def test_visualization():
    """测试可视化函数"""
    print("测试可视化工具...")

    # 生成随机数据
    labels = np.array([0] * 50 + [1] * 50)
    probs = np.concatenate([np.random.rand(50) * 0.3, 0.7 + np.random.rand(50) * 0.3])

    # 测试ROC曲线
    plot_roc_curve(labels, probs, save_path='/tmp/test_roc.png')

    # 测试混淆矩阵
    cm = confusion_matrix(labels, (probs > 0.5).astype(int))
    plot_confusion_matrix(cm, ['Real', 'Fake'], save_path='/tmp/test_cm.png')

    # 测试训练曲线
    train_losses = [0.5, 0.4, 0.3, 0.25, 0.2]
    val_metrics = [
        {'loss': 0.45, 'accuracy': 0.85, 'ap': 0.9},
        {'loss': 0.38, 'accuracy': 0.88, 'ap': 0.92},
        {'loss': 0.32, 'accuracy': 0.90, 'ap': 0.94},
        {'loss': 0.30, 'accuracy': 0.91, 'ap': 0.95},
        {'loss': 0.28, 'accuracy': 0.92, 'ap': 0.96},
    ]
    plot_training_curves(train_losses, val_metrics, save_path='/tmp/test_curves.png')

    print("✓ 可视化工具测试通过")


if __name__ == "__main__":
    test_visualization()
