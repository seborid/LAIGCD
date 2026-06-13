#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LAIGCD 检测服务模块
负责模型加载和图片检测逻辑
"""

import sys
from pathlib import Path
import json
from typing import Optional, Dict, Any

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import build_model
from utils.viz import denormalize_image, tensor_to_heatmap, create_overlay


class DetectionService:
    """
    检测服务类

    负责加载检测模型并提供检测接口
    """

    def __init__(
        self,
        checkpoint_path: str = "./backend/best_model.pth",
        device: str = "cuda",
        img_size: int = 224,
    ):
        """
        初始化检测服务

        Args:
            checkpoint_path: 模型检查点路径
            device: 运行设备
            img_size: 输入图像大小
        """
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.model = None
        self.checkpoint = None
        self.config = None
        self.transform = None

        # 懒加载模型
        self._load_model()

    def _load_model(self):
        """加载模型"""
        print(f"正在加载模型: {self.checkpoint_path}")
        print(f"使用设备: {self.device}")

        # 加载检查点
        checkpoint_path = Path(self.checkpoint_path)
        if not checkpoint_path.is_absolute():
            # 相对于项目根目录
            project_root = Path(__file__).parent.parent
            checkpoint_path = project_root / self.checkpoint_path

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")

        self.checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # 加载配置
        config_path = checkpoint_path.parent / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                self.config = json.load(f)
        else:
            self.config = {
                "clip_model": "ViT-B/32",
                "num_prototypes": 16,
                "use_freq": True,
                "freq_type": "srm",
                "dropout": 0.1,
            }

        # 构建模型
        self.model = build_model(self.config)

        # 加载权重
        state_dict_key = "ema_model_state_dict" if "ema_model_state_dict" in self.checkpoint else "model_state_dict"
        self.model.load_state_dict(self.checkpoint[state_dict_key])
        self.model = self.model.to(self.device)
        self.model.eval()

        # 数据变换
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size + 32),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # 获取阈值
        self.default_threshold = 0.68

        print(f"✓ 模型加载完成")
        print(f"  检查点: {checkpoint_path}")
        print(f"  Epoch: {self.checkpoint.get('epoch', '?')}")
        print(f"  默认阈值: {self.default_threshold:.4f}")
        if "metrics" in self.checkpoint:
            metrics = self.checkpoint["metrics"]
            print(f"  检查点指标: AP={metrics.get('ap', 0):.4f}")

    def detect(
        self,
        image: Image.Image,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        检测图片

        Args:
            image: PIL Image对象
            threshold: 可选的检测阈值

        Returns:
            dict: 包含检测结果和热力图
        """
        if self.model is None:
            self._load_model()

        # 使用默认阈值或指定的阈值
        if threshold is None:
            threshold = self.default_threshold

        # 数据预处理
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 生成解释结果（包含热力图）
        result = self.model.explain(image_tensor)

        # 反归一化原图
        original = denormalize_image(image_tensor)

        # 生成空域热力图叠加图
        spatial_heatmap = tensor_to_heatmap(result['spatial_heatmap'])
        spatial_overlay = create_overlay(original, spatial_heatmap, cmap='jet', alpha=0.45)

        # 生成频域热力图叠加图
        frequency_overlay = None
        if result['frequency_heatmap'] is not None:
            frequency_heatmap = tensor_to_heatmap(result['frequency_heatmap'])
            frequency_overlay = create_overlay(original, frequency_heatmap, cmap='magma', alpha=0.45)

        # 获取预测概率，并按当前阈值重新计算分类结果
        fake_probability = float(result['probabilities'][0].item())
        pred = 1 if fake_probability >= threshold else 0

        # 计算置信度
        if pred == 1:  # Fake
            confidence = fake_probability
            prediction = "Fake"
        else:  # Real
            confidence = 1 - fake_probability
            prediction = "Real"

        return {
            "prediction": prediction,
            "confidence": confidence,
            "fake_probability": fake_probability,
            "threshold": threshold,
            "spatial_overlay": spatial_overlay,
            "frequency_overlay": frequency_overlay,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "device": str(self.device),
            "img_size": self.img_size,
            "default_threshold": self.default_threshold,
            "config": self.config,
        }
