"""
评估指标和工具函数
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)


def compute_metrics(labels, probs, threshold=0.5):
    """
    计算所有评估指标

    Args:
        labels: 真实标签 [N]
        probs: 预测概率 [N]
        threshold: 分类阈值

    Returns:
        metrics: 指标字典
    """
    preds = (probs > threshold).astype(int)
    labels = np.array(labels)
    probs = np.array(probs)

    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'ap': average_precision_score(labels, probs),
    }

    # 如果有两个类别，计算更多指标
    if len(np.unique(labels)) > 1:
        try:
            metrics.update({
                'precision': precision_score(labels, preds, zero_division=0),
                'recall': recall_score(labels, preds, zero_division=0),
                'f1': f1_score(labels, preds, zero_division=0),
                'auc': roc_auc_score(labels, probs),
            })

            # 混淆矩阵
            cm = confusion_matrix(labels, preds)
            tn, fp, fn, tp = cm.ravel()
            metrics['tn'] = int(tn)
            metrics['fp'] = int(fp)
            metrics['fn'] = int(fn)
            metrics['tp'] = int(tp)
        except Exception as e:
            print(f"计算额外指标时出错: {e}")

    return metrics


def compute_per_generator_metrics(labels, probs, generator_names, threshold=0.5):
    """
    计算每个生成器的检测准确率

    Args:
        labels: 真实标签 [N]
        probs: 预测概率 [N]
        generator_names: 生成器名称列表 [N]
        threshold: 分类阈值

    Returns:
        per_gen_metrics: 每个生成器的指标
    """
    per_gen_metrics = {}

    for gen in set(generator_names):
        mask = np.array(generator_names) == gen
        gen_labels = np.array(labels)[mask]
        gen_probs = probs[mask]

        if len(gen_labels) > 0:
            gen_preds = (gen_probs > threshold).astype(int)
            per_gen_metrics[gen] = {
                'accuracy': accuracy_score(gen_labels, gen_preds),
                'ap': average_precision_score(gen_labels, gen_probs) if len(np.unique(gen_labels)) > 1 else 0.0,
                'count': len(gen_labels),
                'real_count': int((gen_labels == 0).sum()),
                'fake_count': int((gen_labels == 1).sum())
            }

    return per_gen_metrics


def print_metrics(metrics, prefix=""):
    """打印指标"""
    print(f"\n{prefix}评估结果:")

    for key, value in metrics.items():
        if key in ['tn', 'fp', 'fn', 'tp', 'real_count', 'fake_count', 'count']:
            continue
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")


def print_per_generator_metrics(per_gen_metrics):
    """打印每个生成器的指标"""
    print("\n各生成器检测准确率:")
    print(f"{'生成器':<20} {'准确率':<10} {'AP':<10} {'样本数':<10}")
    print("-" * 50)

    for gen, metrics in sorted(per_gen_metrics.items()):
        print(f"{gen:<20} {metrics['accuracy']:.4f}    {metrics['ap']:.4f}    {metrics['count']}")


def format_metrics_table(all_metrics):
    """
    将多个模型的指标格式化为表格

    Args:
        all_metrics: {模型名: metrics字典}

    Returns:
        表格字符串
    """
    lines = []
    lines.append(f"{'模型':<20} {'Accuracy':<12} {'AP':<12} {'F1':<12} {'AUC':<12}")
    lines.append("-" * 68)

    for model_name, metrics in all_metrics.items():
        acc = metrics.get('accuracy', 0)
        ap = metrics.get('ap', 0)
        f1 = metrics.get('f1', 0)
        auc = metrics.get('auc', 0)

        lines.append(f"{model_name:<20} {acc:<12.4f} {ap:<12.4f} {f1:<12.4f} {auc:<12.4f}")

    return "\n".join(lines)


def get_optimal_threshold(labels, probs, metric='f1'):
    """
    找到最优阈值（可按F1或Accuracy优化）

    Args:
        labels: 真实标签
        probs: 预测概率
        metric: 'f1' 或 'accuracy'

    Returns:
        optimal_threshold: 最优阈值
        best_score: 该阈值下的最优分数
        metrics_at_threshold: 阈值下的完整指标字典
    """
    from sklearn.metrics import f1_score, accuracy_score

    best_threshold = 0.5
    best_score = 0

    # 粗搜索
    for threshold in np.arange(0.05, 0.95, 0.05):
        preds = (probs >= threshold).astype(int)
        if metric == 'f1':
            score = f1_score(labels, preds, zero_division=0)
        else:
            score = accuracy_score(labels, preds)

        if score > best_score:
            best_score = score
            best_threshold = threshold

    # 在最优阈值附近精细搜索
    for threshold in np.arange(max(0.01, best_threshold - 0.05),
                               min(0.99, best_threshold + 0.05), 0.01):
        preds = (probs >= threshold).astype(int)
        if metric == 'f1':
            score = f1_score(labels, preds, zero_division=0)
        else:
            score = accuracy_score(labels, preds)

        if score > best_score:
            best_score = score
            best_threshold = threshold

    # 计算该阈值下的完整指标
    preds = (probs >= best_threshold).astype(int)
    metrics_at_threshold = compute_metrics(labels, probs, threshold=best_threshold)
    metrics_at_threshold['best_f1'] = best_score if metric == 'f1' else None

    return best_threshold, best_score, metrics_at_threshold


def test_metrics():
    """测试指标函数"""
    print("测试评估指标...")

    # 生成随机数据
    np.random.seed(42)
    n = 100
    labels = np.random.randint(0, 2, n)
    probs = np.random.rand(n)

    # 测试基础指标
    metrics = compute_metrics(labels, probs)
    print_metrics(metrics, prefix="测试")

    # 测试per-generator指标
    generator_names = ['SDXL'] * 50 + ['Midjourney'] * 30 + ['DALL-E'] * 20
    per_gen_metrics = compute_per_generator_metrics(labels, probs, generator_names)
    print_per_generator_metrics(per_gen_metrics)

    # 测试最优阈值
    optimal_threshold = get_optimal_threshold(labels, probs)
    print(f"\n最优阈值: {optimal_threshold:.2f}")

    print("✓ 评估指标测试通过")


if __name__ == "__main__":
    test_metrics()
