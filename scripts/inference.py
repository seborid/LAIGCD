#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LAIGCD 推理脚本
默认行为：评估 test/fake 与 test/real 中的全部图片
"""

import sys
import argparse
import json
from pathlib import Path

from PIL import Image

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torchvision import transforms

from models import build_model
from utils.metrics import compute_metrics, get_optimal_threshold


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def get_inference_transform(img_size=224):
    """获取推理数据变换"""
    return transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_model(checkpoint_path, device="cuda"):
    """加载模型"""
    print(f"加载模型: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config_path = Path(checkpoint_path).parent / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {
            "clip_model": "ViT-B-32",
            "num_prototypes": 16,
            "use_freq": True,
            "freq_type": "srm",
            "dropout": 0.1,
        }

    model = build_model(config)

    state_dict_key = "ema_model_state_dict" if "ema_model_state_dict" in checkpoint else "model_state_dict"
    model.load_state_dict(checkpoint[state_dict_key])
    model = model.to(device)
    model.eval()

    print(f"模型加载完成 (epoch {checkpoint.get('epoch', '?')})")
    print(f"加载权重: {'EMA' if state_dict_key == 'ema_model_state_dict' else '原始模型'}")
    if "metrics" in checkpoint:
        print(f"检查点指标: AP={checkpoint['metrics'].get('ap', 0):.4f}")

    return model, config, checkpoint


def resolve_threshold(checkpoint, manual_threshold=None):
    """确定推理使用的阈值，优先使用验证集阈值"""
    if manual_threshold is not None:
        return float(manual_threshold), "manual_argument"

    metrics = checkpoint.get("metrics", {})

    if metrics.get("optimal_threshold") is not None:
        return float(metrics["optimal_threshold"]), "checkpoint.metrics.optimal_threshold"

    probs = metrics.get("probabilities")
    labels = metrics.get("labels")
    if probs is not None and labels is not None:
        probs = np.asarray(probs).reshape(-1)
        labels = np.asarray(labels).reshape(-1)
        threshold, _, _ = get_optimal_threshold(labels, probs, metric="f1")
        return float(threshold), "checkpoint.validation_metrics_recomputed_f1"

    stored_threshold = metrics.get("threshold")
    if stored_threshold is not None:
        return float(stored_threshold), "checkpoint.metrics.threshold"

    return 0.5, "fallback_default_0.5"


def collect_images(image_dir):
    """收集目录下所有支持的图像文件"""
    image_dir = Path(image_dir)
    image_paths = [
        path for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(image_paths)


def classify_probability(fake_probability, threshold):
    """根据阈值将伪造概率转为类别与置信度"""
    pred = 1 if fake_probability >= threshold else 0
    confidence = fake_probability if pred == 1 else 1 - fake_probability
    return pred, confidence


def predict_fake_probability(model, image_path, transform, device="cuda"):
    """预测单张图像的伪造概率"""
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).squeeze(1)

    return probs[0].item()


def predict_single_image(model, image_path, transform, device="cuda", threshold=0.5):
    """预测单张图像"""
    image_path = Path(image_path)
    fake_probability = predict_fake_probability(model, image_path, transform, device)
    pred, confidence = classify_probability(fake_probability, threshold)

    return {
        "path": str(image_path),
        "filename": image_path.name,
        "label": "Fake" if pred == 1 else "Real",
        "confidence": confidence,
        "fake_probability": fake_probability,
        "threshold": threshold,
    }


def predict_directory(model, image_dir, transform, device="cuda", output_path=None, threshold=0.5):
    """预测目录中的所有图像"""
    image_dir = Path(image_dir)
    image_paths = collect_images(image_dir)

    if not image_paths:
        print(f"在 {image_dir} 中未找到图像")
        return []

    print(f"找到 {len(image_paths)} 张图像")

    results = []
    for i, img_path in enumerate(image_paths, start=1):
        result = predict_single_image(model, img_path, transform, device, threshold=threshold)
        results.append(result)

        if i % 100 == 0 or i == len(image_paths):
            print(f"已处理 {i}/{len(image_paths)}")

    fake_count = sum(1 for r in results if r["label"] == "Fake")
    real_count = len(results) - fake_count

    print("\n批量推理结果:")
    print(f"  总计: {len(results)}")
    print(f"  真实: {real_count} ({real_count / len(results) * 100:.1f}%)")
    print(f"  伪造: {fake_count} ({fake_count / len(results) * 100:.1f}%)")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"结果已保存: {output_path}")

    return results


def evaluate_test_set(
    model,
    test_dir,
    transform,
    device="cuda",
    threshold=0.5,
    threshold_source="manual_argument",
    output_path=None,
):
    """评估 test/real 与 test/fake 目录"""
    test_dir = Path(test_dir)
    class_dirs = {
        "real": {"label": 0, "path": test_dir / "real"},
        "fake": {"label": 1, "path": test_dir / "fake"},
    }

    samples = []
    for class_name, meta in class_dirs.items():
        class_path = meta["path"]
        if not class_path.exists():
            raise FileNotFoundError(f"未找到测试目录: {class_path}")

        class_images = collect_images(class_path)
        print(f"{class_name}: {len(class_images)} 张")
        for image_path in class_images:
            samples.append({
                "path": image_path,
                "true_label": meta["label"],
                "true_label_name": "Fake" if meta["label"] == 1 else "Real",
            })

    if not samples:
        raise RuntimeError(f"在 {test_dir} 下未找到可评估的图像")

    print(f"\n开始评估测试集，共 {len(samples)} 张图像")

    all_labels = []
    all_probs = []

    for i, sample in enumerate(samples, start=1):
        fake_probability = predict_fake_probability(model, sample["path"], transform, device)
        all_labels.append(sample["true_label"])
        all_probs.append(fake_probability)

        if i % 100 == 0 or i == len(samples):
            print(f"已处理 {i}/{len(samples)}")

    metrics = compute_metrics(all_labels, all_probs, threshold=threshold)
    real_count = sum(1 for x in all_labels if x == 0)
    fake_count = len(all_labels) - real_count

    results = []
    misclassified = []
    for sample, fake_probability in zip(samples, all_probs):
        pred_label, confidence = classify_probability(fake_probability, threshold)
        pred_label_name = "Fake" if pred_label == 1 else "Real"
        rel_path = sample["path"].relative_to(test_dir)

        row = {
            "path": str(sample["path"]),
            "relative_path": str(rel_path),
            "filename": sample["path"].name,
            "true_label": sample["true_label"],
            "true_label_name": sample["true_label_name"],
            "pred_label": pred_label,
            "pred_label_name": pred_label_name,
            "confidence": confidence,
            "fake_probability": fake_probability,
            "correct": pred_label == sample["true_label"],
        }
        results.append(row)
        if not row["correct"]:
            misclassified.append(row)

    print("\n测试集评估结果:")
    print(f"  样本总数: {len(all_labels)}")
    print(f"  Real: {real_count}")
    print(f"  Fake: {fake_count}")
    print(f"  使用阈值: {threshold:.2f}")
    print(f"  阈值来源: {threshold_source}")
    print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"  Precision: {metrics.get('precision', 0):.4f}")
    print(f"  Recall: {metrics.get('recall', 0):.4f}")
    print(f"  F1: {metrics.get('f1', 0):.4f}")
    print(f"  AP: {metrics.get('ap', 0):.4f}")
    if "auc" in metrics:
        print(f"  AUC: {metrics['auc']:.4f}")
    if {"tn", "fp", "fn", "tp"} <= metrics.keys():
        print(f"  TN/FP/FN/TP: {metrics['tn']}/{metrics['fp']}/{metrics['fn']}/{metrics['tp']}")

    misclassified.sort(key=lambda item: item["confidence"], reverse=True)
    print(f"\n误分类样本: {len(misclassified)}")
    if misclassified:
        for item in misclassified:
            print(
                "  "
                f"{item['relative_path']} | "
                f"GT={item['true_label_name']} | "
                f"Pred={item['pred_label_name']} | "
                f"Conf={item['confidence']:.2%} | "
                f"P(fake)={item['fake_probability']:.4f}"
            )
    else:
        print("  无")

    summary = {
        "test_dir": str(test_dir),
        "num_samples": len(all_labels),
        "real_count": real_count,
        "fake_count": fake_count,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "metrics": {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in metrics.items()
        },
        "misclassified_count": len(misclassified),
        "misclassified": misclassified,
        "results": results,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n评估结果已保存: {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser("LAIGCD推理")
    parser.add_argument("image", type=str, nargs="?", help="单张图像路径")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/full_run/best_model.pth",
        help="模型检查点路径 (默认: checkpoints/full_run/best_model.pth)",
    )
    parser.add_argument("--image_dir", type=str, help="图像目录路径")
    parser.add_argument("--test_dir", type=str, default="test", help="测试集目录，需包含 fake/ 与 real/")
    parser.add_argument("--output", type=str, help="输出结果文件路径")
    parser.add_argument("--img_size", type=int, default=224, help="输入图像大小")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fake 判定阈值；默认优先使用 checkpoint 对应验证集最优阈值",
    )
    parser.add_argument("--show_attention", action="store_true", help="显示原型注意力")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model, _, checkpoint = load_model(args.checkpoint, device)
    transform = get_inference_transform(args.img_size)
    threshold, threshold_source = resolve_threshold(checkpoint, args.threshold)

    print(f"使用阈值: {threshold:.2f} ({threshold_source})")

    if args.image:
        result = predict_single_image(model, args.image, transform, device, threshold=threshold)

        print("\n" + "=" * 50)
        print(f"图像: {result['path']}")
        print(f"预测: {result['label']}")
        print(f"置信度: {result['confidence']:.2%}")
        print(f"伪造概率: {result['fake_probability']:.4f}")
        print("=" * 50)

        if args.show_attention:
            from utils.viz import visualize_prototype_attention

            image = Image.open(args.image).convert("RGB")
            image_tensor = transform(image)
            output_path = Path(args.image).stem + "_attention.png"
            visualize_prototype_attention(model, image_tensor, output_path, device)
        return

    if args.image_dir:
        results = predict_directory(
            model,
            args.image_dir,
            transform,
            device,
            output_path=args.output,
            threshold=threshold,
        )

        print("\n前10个结果:")
        for result in results[:10]:
            print(f"  {result['path']}: {result['label']} ({result['confidence']:.2%})")
        return

    evaluate_test_set(
        model,
        args.test_dir,
        transform,
        device,
        threshold=threshold,
        threshold_source=threshold_source,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
