"""
原型学习模块
借鉴GAPL项目，使用可学习的原型和交叉注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeModule(nn.Module):
    """
    原型学习模块
    使用可学习的原型向量作为中间表示，通过交叉注意力与输入特征交互

    Args:
        input_dim: 输入特征维度
        num_prototypes: 原型数量
        num_heads: 交叉注意力的头数
        dropout: Dropout概率
    """
    def __init__(self, input_dim=640, num_prototypes=16, num_heads=4, dropout=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.num_prototypes = num_prototypes
        self.num_heads = num_heads

        # 可学习的原型向量
        self.prototypes = nn.Parameter(torch.Tensor(num_prototypes, input_dim))
        nn.init.xavier_uniform_(self.prototypes)

        # 交叉注意力机制
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, input_dim),
            nn.Dropout(dropout)
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x, return_attn=False):
        """
        前向传播

        Args:
            x: [B, D] 或 [B, N, D] 输入特征
            return_attn: 是否返回注意力权重

        Returns:
            logits: [B, 1] 分类logits
            attn_weights: [B, num_prototypes] 原型注意力权重（可选）
        """
        # 确保输入是3D张量
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, D]

        B, N, D = x.shape

        # 扩展原型到batch维度
        prototypes = self.prototypes.unsqueeze(0).expand(B, -1, -1)  # [B, P, D]

        # 交叉注意力：query=输入特征, key/value=原型
        attn_out, attn_weights = self.cross_attn(
            query=x,              # [B, N, D]
            key=prototypes,       # [B, P, D]
            value=prototypes,     # [B, P, D]
            need_weights=True,
            average_attn_weights=False  # 保留每个头的权重
        )

        # 残差连接 + 层归一化
        x = self.norm1(x + attn_out)  # [B, N, D]

        # 前馈网络 + 残差
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)  # [B, N, D]

        # 聚合（平均池化）
        x = x.mean(dim=1)  # [B, D]

        # 分类
        logits = self.classifier(x)  # [B, 1]

        if return_attn:
            # attn_weights: [B, num_heads, N, P]
            # 聚合heads和patches，得到每个原型的平均注意力
            attn_weights = attn_weights.mean(dim=(1, 2))  # [B, P]
            return logits, attn_weights

        return logits

    def get_prototypes(self):
        """获取原型向量"""
        return self.prototypes.detach()


class SimpleClassifier(nn.Module):
    """
    简单分类器（作为对比，不使用原型学习）
    """
    def __init__(self, input_dim=640, hidden_dim=256, dropout=0.1):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        """
        Args:
            x: [B, D] 输入特征
        Returns:
            logits: [B, 1] 分类logits
        """
        return self.classifier(x)


def test_prototype_module():
    """测试原型模块"""
    print("测试 PrototypeModule...")

    # 创建模块
    proto_module = PrototypeModule(
        input_dim=640,
        num_prototypes=16,
        num_heads=4
    )

    # 测试前向传播（2D输入）
    x_2d = torch.randn(4, 640)
    out_2d, attn = proto_module(x_2d, return_attn=True)

    print(f"  2D输入形状: {x_2d.shape}")
    print(f"  输出形状: {out_2d.shape}")
    print(f"  注意力权重形状: {attn.shape}")

    # 测试前向传播（3D输入）
    x_3d = torch.randn(4, 10, 640)
    out_3d = proto_module(x_3d)

    print(f"  3D输入形状: {x_3d.shape}")
    print(f"  输出形状: {out_3d.shape}")

    # 计算参数量
    total_params = sum(p.numel() for p in proto_module.parameters())
    trainable_params = sum(p.numel() for p in proto_module.parameters() if p.requires_grad)

    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # 获取原型
    prototypes = proto_module.get_prototypes()
    print(f"  原型形状: {prototypes.shape}")

    print("✓ PrototypeModule 测试通过")


def test_simple_classifier():
    """测试简单分类器"""
    print("测试 SimpleClassifier...")

    classifier = SimpleClassifier(input_dim=640)

    x = torch.randn(4, 640)
    out = classifier(x)

    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {out.shape}")
    print(f"  参数量: {sum(p.numel() for p in classifier.parameters()):,}")

    print("✓ SimpleClassifier 测试通过")


if __name__ == "__main__":
    test_prototype_module()
    print()
    test_simple_classifier()
