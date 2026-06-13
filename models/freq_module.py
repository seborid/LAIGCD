"""
频域特征提取模块
参考IAPL项目，使用SRM滤波器提取图像频域特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SRMConv(nn.Module):
    """
    SRM (Spatial Rich Model) 高频滤波器
    包含30个预定义的高通滤波器，用于检测图像篡改痕迹
    """
    def __init__(self):
        super().__init__()
        # 构建30个SRM滤波器
        self.filters = self._build_srm_filters()

        # 创建卷积层（权重不可训练）
        self.conv = nn.Conv2d(3, 30, kernel_size=5, padding=2, bias=False)
        self.conv.weight = nn.Parameter(self.filters, requires_grad=False)

    def _build_srm_filters(self):
        """构建30个SRM滤波器"""
        filters = torch.zeros(30, 3, 5, 5)

        # 基础高通滤波器模板
        # 这里使用简化的滤波器设计
        # 实际应用中可以使用SRM论文中的30个标准滤波器

        # 1-4: 基本拉普拉斯变体 (4个)
        for i in range(4):
            filter_2d = torch.zeros(5, 5)
            filter_2d[2, 2] = -4.0
            for j in range(5):
                for k in range(5):
                    if j == 2 and k == 2:
                        continue
                    if (j + k) % 2 == i % 2:  # 创建不同模式
                        filter_2d[j, k] = 1.0
            filter_2d = filter_2d / filter_2d.abs().sum()
            for c in range(3):
                filters[i, c] = filter_2d

        # 5-30: 各种方向和尺度的高通滤波器 (26个)
        for i in range(4, 30):
            filter_2d = torch.randn(5, 5) * 0.3

            # 确保是高通滤波器（中心为负，周围为正）
            filter_2d[2, 2] = -torch.abs(filter_2d).sum() - 1.0

            # 归一化
            filter_2d = filter_2d - filter_2d.mean()
            filter_2d = filter_2d / (filter_2d.std() + 1e-6)

            # RGB三通道使用相同的滤波器
            for c in range(3):
                filters[i, c] = filter_2d.clone()

        return filters

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] 输入图像
        Returns:
            [B, 30, H, W] 频域特征
        """
        return self.conv(x)


class FreqModule(nn.Module):
    """
    频域特征编码器
    使用SRM滤波器提取高频特征，然后通过轻量CNN编码
    """
    def __init__(self, output_dim=128):
        super().__init__()

        # SRM高频滤波
        self.hpf = SRMConv()

        # 轻量编码器
        self.encoder = nn.Sequential(
            # 第一阶段：浅层特征
            nn.Conv2d(30, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 第二阶段：中层特征
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),

            # 第三阶段：深层特征
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 全局池化
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            # 投影层
            nn.Linear(64, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    @staticmethod
    def _normalize_heatmap(heatmap):
        """将热力图逐样本归一化到[0, 1]。"""
        flat = heatmap.flatten(1)
        min_vals = flat.min(dim=1, keepdim=True).values.view(-1, 1, 1)
        max_vals = flat.max(dim=1, keepdim=True).values.view(-1, 1, 1)
        return (heatmap - min_vals) / (max_vals - min_vals + 1e-6)

    def apply_hpf(self, x):
        """应用SRM高通滤波器。"""
        return self.hpf(x)

    def encode_filtered(self, filtered_x):
        """对滤波响应进行编码。"""
        return self.encoder(filtered_x)

    def get_frequency_heatmap(self, x):
        """
        生成频域异常热力图。

        Args:
            x: [B, 3, H, W] 输入图像

        Returns:
            [B, H, W] 归一化频域热力图
        """
        filtered = self.apply_hpf(x)
        heatmap = filtered.abs().mean(dim=1)
        return self._normalize_heatmap(heatmap)

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] 输入图像
        Returns:
            [B, output_dim] 频域特征向量
        """
        # SRM滤波
        x = self.apply_hpf(x)  # [B, 30, H, W]

        # 编码
        x = self.encode_filtered(x)  # [B, output_dim]

        return x


class DCTModule(nn.Module):
    """
    DCT频域特征提取模块（备选方案）
    使用离散余弦变换提取频域特征
    """
    def __init__(self, output_dim=128, window_size=8):
        super().__init__()
        self.window_size = window_size

        # 轻量编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, output_dim)
        )

    @staticmethod
    def _normalize_heatmap(heatmap):
        """将热力图逐样本归一化到[0, 1]。"""
        flat = heatmap.flatten(1)
        min_vals = flat.min(dim=1, keepdim=True).values.view(-1, 1, 1)
        max_vals = flat.max(dim=1, keepdim=True).values.view(-1, 1, 1)
        return (heatmap - min_vals) / (max_vals - min_vals + 1e-6)

    def _dct_transform(self, x):
        """简化的DCT变换（实际项目中建议使用pytorch-wavelets）"""
        # 这里使用FFT的实部作为DCT的近似
        # 实际应用中应使用专门的DCT实现
        fft = torch.fft.rfft2(x, norm='ortho')
        magnitude = torch.abs(fft)
        return magnitude

    def get_frequency_heatmap(self, x):
        """
        生成DCT频域热力图。

        Args:
            x: [B, 3, H, W] 输入图像

        Returns:
            [B, H, W] 归一化频域热力图
        """
        magnitude = self._dct_transform(x)
        heatmap = magnitude.mean(dim=1)
        heatmap = F.interpolate(
            heatmap.unsqueeze(1),
            size=x.shape[-2:],
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        return self._normalize_heatmap(heatmap)

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] 输入图像
        Returns:
            [B, output_dim] 频域特征向量
        """
        # DCT变换
        x = self._dct_transform(x)  # [B, 3, H, W/2+1]

        # 编码
        x = self.encoder(x)  # [B, output_dim]

        return x


def test_freq_module():
    """测试频域模块"""
    print("测试 FreqModule...")

    # 创建模块
    freq_module = FreqModule(output_dim=128)

    # 测试前向传播
    x = torch.randn(2, 3, 224, 224)
    out = freq_module(x)

    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {out.shape}")
    print(f"  参数数量: {sum(p.numel() for p in freq_module.parameters()):,}")

    # 计算可训练参数
    trainable = sum(p.numel() for p in freq_module.parameters() if p.requires_grad)
    print(f"  可训练参数: {trainable:,}")

    print("✓ FreqModule 测试通过")


if __name__ == "__main__":
    test_freq_module()
