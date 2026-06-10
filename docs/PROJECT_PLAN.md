# LAIGCD: Lightweight AI-Generated Content Detector

## 项目概述

**目标**：构建一个轻量级的AI生成图像检测模型，数据需求 < 100GB，可在4070Ti上训练

**技术路线**：CLIP空域特征 + 频域分析 + 原型学习（借鉴GAPL核心思想）

**预期效果**：
- 在多种生成器（SD, MJ, DALL-E等）上达到 > 85% 准确率
- 训练时间 < 8小时（4070Ti）
- 推理速度 > 30 FPS
- 可训练参数 < 1M

---

## 一、GAPL与IAPL核心思想对比

| 特性 | GAPL | IAPG | 本方案 |
|------|------|------|--------|
| **骨干网络** | CLIP ViT-L/14 + LoRA | CLIP ViT-L/14 + Adapter | CLIP ViT-B/32 (冻结) |
| **核心机制** | 原型学习 + 交叉注意力 | 图像自适应提示学习 | 原型学习 + 频域特征 |
| **频域分析** | ❌ | ✅ DCT条件模块 | ✅ 轻量DCT+SRM |
| **可训练参数** | ~500K + 原型 | ~6M + Prompt | ~800K |
| **原型数量** | 64个 | - | 16个 |
| **显存占用** | ~6GB | ~12GB | ~4GB |

---

## 二、系统架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                        输入图像 [3, 224, 224]                      │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                              ▼
    ┌──────────────────────┐      ┌──────────────────────┐
    │    空域特征分支      │      │    频域特征分支      │
    │                      │      │                      │
    │  CLIP ViT-B/32       │      │  轻量DCT+SRM         │
    │  (冻结，预训练)      │      │  (可训练，~1M参数)   │
    │                      │      │                      │
    │  输出: [B, 512]      │      │  输出: [B, 128]      │
    └──────────────────────┘      └──────────────────────┘
                │                              │
                └──────────────┬───────────────┘
                               ▼
                    ┌──────────────────────┐
                    │    特征融合          │
                    │  concat + proj       │
                    │  [B, 640]            │
                    └──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    原型学习模块      │
                    │  (借鉴GAPL)         │
                    │                      │
                    │  16个可学习原型      │
                    │  交叉注意力(4头)     │
                    │  残差连接+层归一化   │
                    └──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  分类头 [B, 1]       │
                    │  真实(0) / 伪造(1)   │
                    └──────────────────────┘
```

---

## 三、核心模块设计

### 3.1 原型学习模块（借鉴GAPL）

```python
class PrototypeModule(nn.Module):
    """轻量原型学习模块"""
    def __init__(self, input_dim=640, num_prototypes=16, num_heads=4, dropout=0.1):
        super().__init__()
        # 可学习的原型向量
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, input_dim))
        nn.init.xavier_uniform_(self.prototypes)

        # 交叉注意力机制
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )

        # 层归一化
        self.norm = nn.LayerNorm(input_dim)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: [B, D] 或 [B, N, D]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, D]

        B, N, D = x.shape
        prototypes = self.prototypes.unsqueeze(0).expand(B, -1, -1)  # [B, P, D]

        # 交叉注意力：query=特征, key/value=原型
        attn_out, attn_weights = self.cross_attn(
            query=x,
            key=prototypes,
            value=prototypes,
            need_weights=True
        )

        # 残差连接 + 层归一化
        x = self.norm(x + attn_out)

        # 聚合
        x = x.mean(dim=1)  # [B, D]

        # 分类
        logits = self.classifier(x)  # [B, 1]
        return logits, attn_weights
```

### 3.2 频域特征模块（借鉴IAPL，简化版）

```python
class SRMConv(nn.Module):
    """SRM高频滤波器（30个滤波器）"""
    def __init__(self):
        super().__init__()
        # 30个预定义的高通滤波器
        self.filters = self._build_filters()
        self.conv = nn.Conv2d(3, 30, 5, padding=2, bias=False)
        self.conv.weight = nn.Parameter(self.filters, requires_grad=False)

    def _build_filters(self):
        # 参考IAPL的srm.py实现
        filters = torch.zeros(30, 3, 5, 5)
        # ... 滤波器定义 ...
        return filters

    def forward(self, x):
        return self.conv(x)


class FreqModule(nn.Module):
    """轻量频域特征提取模块"""
    def __init__(self, output_dim=128):
        super().__init__()
        # SRM高频滤波
        self.hpf = SRMConv()

        # 轻量编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(30, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # x: [B, 3, H, W]
        x = self.hpf(x)  # [B, 30, H, W]
        x = self.encoder(x)  # [B, 128]
        return x
```

### 3.3 完整模型架构

```python
class LightweightAIGCDetector(nn.Module):
    """轻量AIGC检测器"""
    def __init__(self,
                 clip_model='ViT-B-32',
                 clip_pretrained='laion2b_s34b_b79k',
                 num_prototypes=16,
                 freq_dim=128,
                 use_freq=True,
                 dropout=0.1):
        super().__init__()

        self.use_freq = use_freq
        self.num_prototypes = num_prototypes

        # 1. 空域分支：CLIP特征提取（冻结）
        import open_clip
        self.clip, _, _ = open_clip.create_model_and_transforms(
            clip_model, pretrained=clip_pretrained
        )

        # 冻结CLIP参数
        for param in self.clip.parameters():
            param.requires_grad = False
        self.clip.eval()

        # 获取CLIP输出维度
        clip_dim = self.clip.visual.output_dim  # ViT-B/32: 512

        # 2. 频域分支（可选）
        if use_freq:
            self.freq_module = FreqModule(freq_dim)
            input_dim = clip_dim + freq_dim
        else:
            input_dim = clip_dim

        # 3. 原型学习模块
        self.prototype = PrototypeModule(
            input_dim=input_dim,
            num_prototypes=num_prototypes,
            dropout=dropout
        )

        # 损失函数
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x, return_attn=False):
        # CLIP特征提取
        with torch.no_grad():
            clip_features = self.clip.visual(x)  # [B, 512]
            # 确保是全局特征
            if clip_features.dim() == 3:
                clip_features = clip_features.mean(dim=1)

        # 频域特征
        if self.use_freq:
            freq_features = self.freq_module(x)  # [B, 128]
            features = torch.cat([clip_features, freq_features], dim=1)  # [B, 640]
        else:
            features = clip_features

        # 原型分类
        logits, attn_weights = self.prototype(features)

        if return_attn:
            return logits, attn_weights
        return logits

    def get_criterion(self, outputs, targets):
        return self.criterion(outputs.squeeze(), targets.float())

    @torch.no_grad()
    def get_attention_weights(self, x):
        """获取原型注意力权重，用于可视化"""
        _, attn_weights = self.forward(x, return_attn=True)
        return attn_weights  # [B, 1, num_prototypes]
```

---

## 四、数据集策略（100GB限制）

### 4.1 推荐数据集组合

| 数据集 | 大小 | 内容 | 用途 |
|--------|------|------|------|
| **GenImage** | ~30GB | SDv1.4/SDXL/MJ生成图像 | 主要训练数据 |
| **CNNSpot** | ~15GB | ProGAN/StyleGAN等 | 传统GAN检测 |
| **Real Images** | ~30GB | ImageNet/COCO子集 | 真实图像 |
| **DiffusionDB** | ~10GB | Stable Diffusion生成 | 扩充数据 |
| **验证集** | ~8GB | 各种生成器混合 | 验证 |
| **测试集** | ~7GB | 未见过的生成器 | 最终测试 |

**总计：~100GB**

### 4.2 数据组织结构

```
data/
├── train/
│   ├── real/
│   │   ├── imagenet_10k/
│   │   └── coco_5k/
│   └── fake/
│       ├── sdv14_10k/
│       ├── sdxl_5k/
│       ├── midjourney_3k/
│       ├── progan_2k/
│       └── stylegan_2k/
├── val/
│   ├── real/
│   │   └── imagenet_val/
│   └── fake/
│       └── mixed_gan_val/
└── test/
    ├── real/
    │   └── real_test/
    └── fake/
        ├── unseen_gan/
        └── unseen_diffusion/
```

### 4.3 数据增强

```python
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

---

## 五、训练配置（4070Ti优化）

### 5.1 超参数配置

```python
config = {
    # 模型配置
    'clip_model': 'ViT-B-32',
    'clip_pretrained': 'laion2b_s34b_b79k',
    'num_prototypes': 16,
    'use_freq': True,
    'freq_dim': 128,
    'dropout': 0.1,

    # 训练配置
    'batch_size': 16,           # 单卡4070Ti推荐
    'accumulation_steps': 2,    # 等效batch=32
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'epochs': 30,

    # 优化配置
    'use_amp': True,            # 混合精度训练
    'use_ema': True,            # 指数移动平均
    'ema_decay': 0.9999,
    'warmup_epochs': 3,

    # 损失配置
    'label_smoothing': 0.1,

    # 保存配置
    'save_freq': 5,
    'eval_freq': 1,
    'print_freq': 50,
}
```

### 5.2 训练策略

```python
# 优化器
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config['learning_rate'],
    weight_decay=config['weight_decay']
)

# 学习率调度器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config['epochs'],
    eta_min=1e-6
)

# 混合精度训练
scaler = torch.cuda.amp.GradScaler()

# EMA
from timm.utils import ModelEmaV2
model_ema = ModelEmaV2(model, decay=config['ema_decay'])
```

### 5.3 显存优化技巧

1. **梯度累积**：accumulation_steps=2，等效batch=32
2. **混合精度**：use_amp=True，节省40%显存
3. **CLIP冻结**：不计算CLIP的梯度
4. **梯度检查点**（可选）：频域模块使用gradient checkpointing

---

## 六、实现路线图

### 阶段0：环境准备（0.5天）

- [ ] 创建项目目录结构
- [ ] 安装依赖包
- [ ] 下载预训练CLIP模型
- [ ] 准备数据集

```bash
# 项目结构
LAIGCD/
├── data/                   # 数据集
├── models/                 # 模型定义
│   ├── __init__.py
│   ├── prototype.py       # 原型模块
│   ├── freq_module.py     # 频域模块
│   └── detector.py        # 完整检测器
├── utils/                  # 工具函数
│   ├── data.py            # 数据加载
│   ├── train.py           # 训练函数
│   └── metrics.py         # 评估指标
├── scripts/                # 脚本
│   ├── train.sh
│   └── eval.sh
├── checkpoints/            # 模型保存
└── results/                # 结果输出
```

### 阶段1：核心模块实现（1天）

- [ ] 实现`SRMConv`（参考IAPL的srm.py）
- [ ] 实现`FreqModule`
- [ ] 实现`PrototypeModule`
- [ ] 实现`LightweightAIGCDetector`

### 阶段2：数据加载器（0.5天）

- [ ] 实现`AIGCDataset`
- [ ] 实现数据增强
- [ ] 实现数据采样器（平衡真实/伪造样本）

### 阶段3：训练流程（1天）

- [ ] 实现`train_one_epoch`
- [ ] 实现`validate`
- [ ] 实现EMA、混合精度、梯度累积
- [ ] 实现训练脚本

### 阶段4：实验与调优（2天）

- [ ] 基线训练
- [ ] 超参数搜索（原型数量、学习率等）
- [ ] 消融实验（频域模块的作用）
- [ ] 错误分析

### 阶段5：评估与部署（0.5天）

- [ ] 测试集评估
- [ ] 可视化原型注意力
- [ ] 推理脚本
- [ ] 性能优化

**总计：5天**

---

## 七、预期性能

| 指标 | 目标值 |
|------|--------|
| 可训练参数 | ~800K |
| 显存占用 | ~4GB (训练), ~2GB (推理) |
| 训练时间 | ~6小时 (4070Ti, 30 epochs) |
| 推理速度 | ~50 FPS |
| 检测准确率 | > 85% (混合测试集) |
| 模型大小 | ~50 MB |

---

## 八、核心优势

1. **轻量化**：只有~800K可训练参数，模型小易于部署
2. **高效**：CLIP冻结，只训练原型和频域模块
3. **可解释**：原型注意力可视化，理解检测依据
4. **泛化**：结合空域+频域特征，对未见生成器有更好泛化
5. **实用**：100GB数据即可训练，4070Ti单卡可跑

---

## 九、风险与应对

| 风险 | 应对措施 |
|------|----------|
| 数据不足 | 使用数据增强；考虑在线生成 |
| 过拟合 | 增加dropout；早停；更多正则化 |
| 频域特征无效 | 消融实验验证；尝试其他频域方法 |
| 显存不足 | 减小batch_size；使用梯度检查点 |
| 效果不佳 | 分析混淆矩阵；调整原型数量；尝试双分支融合 |

---

## 十、后续优化方向

1. **知识蒸馏**：从大模型（ViT-L/14）蒸馏到小模型
2. **原型动态更新**：根据新数据更新原型
3. **多尺度特征**：融合CLIP的多层特征
4. **自适应原型**：不同生成器使用不同原型子集
5. **时序扩展**：扩展到视频检测

---

**开始执行！** 🚀
