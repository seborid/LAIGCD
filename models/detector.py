"""
LAIGCD 轻量级AIGC检测器
整合CLIP特征、频域特征和原型学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .freq_module import FreqModule, DCTModule
from .prototype import PrototypeModule, SimpleClassifier


class LightweightAIGCDetector(nn.Module):
    """
    轻量级AI生成内容检测器

    架构：
        CLIP ViT-B/32 (冻结) → 空域特征 [512]
        FreqModule (可训练) → 频域特征 [128]
        拼接 → [640]
        PrototypeModule → 分类输出 [1]

    Args:
        clip_model: CLIP模型名称 (默认 ViT-B-32)
        clip_pretrained: 预训练权重
        num_prototypes: 原型数量
        freq_dim: 频域特征维度
        use_freq: 是否使用频域特征
        freq_type: 频域模块类型 ('srm' 或 'dct')
        dropout: Dropout概率
        use_simple_classifier: 是否使用简单分类器（对比实验用）
    """
    def __init__(
        self,
        clip_model='ViT-B-32',
        clip_pretrained='laion2b_s34b_b79k',
        num_prototypes=16,
        freq_dim=128,
        use_freq=True,
        freq_type='srm',
        dropout=0.1,
        use_simple_classifier=False
    ):
        super().__init__()

        self.use_freq = use_freq
        self.num_prototypes = num_prototypes
        self.clip_model_name = clip_model

        # 1. 加载CLIP模型
        try:
            import open_clip
            self.clip, _, _ = open_clip.create_model_and_transforms(
                clip_model,
                pretrained=clip_pretrained
            )
        except ImportError:
            raise ImportError("请安装 open-clip-torch: pip install open-clip-torch")

        # 获取CLIP输出维度
        self.clip_dim = self.clip.visual.output_dim  # ViT-B-32: 512

        # 冻结CLIP参数
        for param in self.clip.parameters():
            param.requires_grad = False
        self.clip.eval()

        # 2. 频域分支（可选）
        if use_freq:
            if freq_type == 'srm':
                self.freq_module = FreqModule(output_dim=freq_dim)
            elif freq_type == 'dct':
                self.freq_module = DCTModule(output_dim=freq_dim)
            else:
                raise ValueError(f"未知的频域模块类型: {freq_type}")

            input_dim = self.clip_dim + freq_dim
        else:
            input_dim = self.clip_dim

        # 3. 分类模块
        if use_simple_classifier:
            self.classifier = SimpleClassifier(
                input_dim=input_dim,
                dropout=dropout
            )
        else:
            self.classifier = PrototypeModule(
                input_dim=input_dim,
                num_prototypes=num_prototypes,
                dropout=dropout
            )

        # 损失函数
        self.criterion = nn.BCEWithLogitsLoss()

    @staticmethod
    def _normalize_heatmap(heatmap):
        """将热力图逐样本归一化到[0, 1]。"""
        flat = heatmap.flatten(1)
        min_vals = flat.min(dim=1, keepdim=True).values.view(-1, 1, 1)
        max_vals = flat.max(dim=1, keepdim=True).values.view(-1, 1, 1)
        return (heatmap - min_vals) / (max_vals - min_vals + 1e-6)

    def extract_clip_features(self, x, enable_grad=False):
        """
        提取CLIP特征。

        Args:
            x: [B, 3, H, W] 输入图像
            enable_grad: 是否保留梯度用于解释

        Returns:
            [B, clip_dim] CLIP全局特征
        """
        if enable_grad:
            clip_features = self.clip.visual(x)
        else:
            with torch.no_grad():
                clip_features = self.clip.visual(x)

        if clip_features.dim() == 3:
            clip_features = clip_features.mean(dim=1)

        return clip_features

    def extract_features(self, x, enable_clip_grad=False):
        """
        提取融合特征。

        Args:
            x: [B, 3, H, W] 输入图像
            enable_clip_grad: 是否保留CLIP梯度

        Returns:
            features: [B, D] 融合特征
            clip_features: [B, clip_dim]
            freq_features: [B, freq_dim] 或 None
        """
        clip_features = self.extract_clip_features(x, enable_grad=enable_clip_grad)

        freq_features = None
        if self.use_freq:
            freq_features = self.freq_module(x)
            features = torch.cat([clip_features, freq_features], dim=1)
        else:
            features = clip_features

        return features, clip_features, freq_features

    def forward(self, x, return_attn=False):
        """
        前向传播

        Args:
            x: [B, 3, H, W] 输入图像
            return_attn: 是否返回注意力权重

        Returns:
            logits: [B, 1] 分类logits
            attn_weights: [B, num_prototypes] 注意力权重（可选）
        """
        features, _, _ = self.extract_features(x, enable_clip_grad=False)

        # 分类
        if isinstance(self.classifier, PrototypeModule):
            logits = self.classifier(features, return_attn=return_attn)
            if return_attn:
                logits, attn_weights = logits
                return logits, attn_weights
            return logits
        else:
            return self.classifier(features)

    def get_criterion(self, outputs, targets):
        """
        计算损失

        Args:
            outputs: [B, 1] 模型输出
            targets: [B] 目标标签 (0=real, 1=fake)

        Returns:
            loss: 损失值
        """
        targets = targets.float()
        return self.criterion(outputs.squeeze(), targets)

    def get_attention_weights(self, x):
        """
        获取原型注意力权重（用于可视化）

        Args:
            x: [B, 3, H, W] 输入图像

        Returns:
            attn_weights: [B, num_prototypes] 注意力权重
        """
        if isinstance(self.classifier, PrototypeModule):
            with torch.no_grad():
                _, attn_weights = self.forward(x, return_attn=True)
            return attn_weights
        else:
            raise NotImplementedError("简单分类器不支持注意力权重")

    def explain(self, x):
        """
        生成预测及空域/频域热力图。

        Args:
            x: [B, 3, H, W] 输入图像

        Returns:
            dict:
                logits: [B, 1]
                probabilities: [B]
                predictions: [B]
                prototype_attention: [B, P] 或 None
                spatial_heatmap: [B, H, W]
                frequency_heatmap: [B, H, W] 或 None
        """
        self.eval()

        input_tensor = x.detach().clone()
        input_tensor.requires_grad_(True)

        features, _, _ = self.extract_features(input_tensor, enable_clip_grad=True)

        if isinstance(self.classifier, PrototypeModule):
            logits, attn_weights = self.classifier(features, return_attn=True)
        else:
            logits = self.classifier(features)
            attn_weights = None

        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs > 0.5).long()

        target = logits.sum()
        grads = torch.autograd.grad(
            outputs=target,
            inputs=input_tensor,
            retain_graph=False,
            create_graph=False,
            allow_unused=False
        )[0]

        spatial_heatmap = grads.abs().mean(dim=1)
        spatial_heatmap = self._normalize_heatmap(spatial_heatmap)

        if attn_weights is not None:
            topk = min(3, attn_weights.shape[1])
            gate = torch.topk(attn_weights, k=topk, dim=1).values.mean(dim=1)
            spatial_heatmap = spatial_heatmap * gate.view(-1, 1, 1)
            spatial_heatmap = self._normalize_heatmap(spatial_heatmap)

        frequency_heatmap = None
        if self.use_freq and hasattr(self.freq_module, 'get_frequency_heatmap'):
            with torch.no_grad():
                frequency_heatmap = self.freq_module.get_frequency_heatmap(x.detach())

        return {
            'logits': logits.detach(),
            'probabilities': probs.detach(),
            'predictions': preds.detach(),
            'prototype_attention': attn_weights.detach() if attn_weights is not None else None,
            'spatial_heatmap': spatial_heatmap.detach(),
            'frequency_heatmap': frequency_heatmap.detach() if frequency_heatmap is not None else None
        }

    def get_prototypes(self):
        """获取原型向量"""
        if isinstance(self.classifier, PrototypeModule):
            return self.classifier.get_prototypes()
        else:
            raise NotImplementedError("简单分类器没有原型向量")

    @torch.no_grad()
    def predict(self, x):
        """
        预测

        Args:
            x: [B, 3, H, W] 输入图像

        Returns:
            probs: [B] 伪造概率
            preds: [B] 预测标签 (0=real, 1=fake)
        """
        self.eval()
        logits = self.forward(x)
        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs > 0.5).long()
        return probs, preds

    def print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        clip_params = sum(p.numel() for p in self.clip.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\n=== 模型信息 ===")
        print(f"CLIP模型: {self.clip_model_name}")
        print(f"CLIP维度: {self.clip_dim}")
        print(f"使用频域特征: {self.use_freq}")
        print(f"原型数量: {self.num_prototypes}")
        print(f"\n参数统计:")
        print(f"  总参数量: {total_params:,}")
        print(f"  CLIP参数 (冻结): {clip_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  可训练比例: {trainable_params / total_params * 100:.2f}%")
        print("==================\n")


def build_model(config):
    """
    构建模型的工厂函数

    Args:
        config: 配置字典

    Returns:
        model: LAIGCD检测器
    """
    return LightweightAIGCDetector(
        clip_model=config.get('clip_model', 'ViT-B-32'),
        clip_pretrained=config.get('clip_pretrained', 'laion2b_s34b_b79k'),
        num_prototypes=config.get('num_prototypes', 16),
        freq_dim=config.get('freq_dim', 128),
        use_freq=config.get('use_freq', True),
        freq_type=config.get('freq_type', 'srm'),
        dropout=config.get('dropout', 0.1),
        use_simple_classifier=config.get('use_simple_classifier', False)
    )


def test_detector():
    """测试检测器"""
    print("测试 LightweightAIGCDetector...")

    # 创建模型
    model = LightweightAIGCDetector(
        clip_model='ViT-B-32',
        num_prototypes=16,
        use_freq=True,
        freq_type='srm'
    )

    # 打印模型信息
    model.print_model_info()

    # 测试前向传播
    x = torch.randn(2, 3, 224, 224)
    logits, attn = model(x, return_attn=True)

    print(f"输入形状: {x.shape}")
    print(f"输出logits形状: {logits.shape}")
    print(f"注意力权重形状: {attn.shape}")

    # 测试预测
    probs, preds = model.predict(x)
    print(f"预测概率: {probs}")
    print(f"预测标签: {preds}")

    # 测试损失计算
    targets = torch.tensor([0, 1])
    loss = model.get_criterion(logits, targets)
    print(f"损失值: {loss.item():.4f}")

    print("✓ LightweightAIGCDetector 测试通过")


if __name__ == "__main__":
    test_detector()
