# LAIGCD 技术设计文档

## 一、模块详细设计

### 1.1 SRM高频滤波器

SRM (Spatial Rich Model) 是图像取证中常用的特征提取方法，包含30个高通滤波器用于检测图像篡改痕迹。

```python
class SRMConv(nn.Module):
    """SRM高频滤波器 - 30个预定义滤波器"""
    def __init__(self):
        super().__init__()

        # 30个高通滤波器（来自图像取证领域）
        # 包括：基本的拉普拉斯、定向梯度、高斯差分等
        self.filters = self._build_srm_filters()

        # 创建卷积层（权重不可训练）
        self.conv = nn.Conv2d(3, 30, kernel_size=5, padding=2, bias=False)
        self.conv.weight = nn.Parameter(self.filters, requires_grad=False)

    def _build_srm_filters(self):
        """构建30个SRM滤波器"""
        import numpy as np

        # 基础滤波器类型
        # 1. 拉普拉斯滤波器 (4个方向)
        laplacian_1 = np.array([[0, 0, 1, 0, 0],
                                [0, 1, 2, 1, 0],
                                [1, 2, -16, 2, 1],
                                [0, 1, 2, 1, 0],
                                [0, 0, 1, 0, 0]], dtype=np.float32)

        # 2. 定向梯度滤波器
        # ... 其他29个滤波器定义 ...

        # 简化实现：使用随机初始化的高通滤波器
        # 实际项目中应使用论文中的30个标准滤波器
        filters = torch.zeros(30, 3, 5, 5)
        for i in range(30):
            # RGB三通道使用相同的滤波器
            filter_2d = torch.randn(5, 5) * 0.1
            filter_2d[2, 2] = -4.0  # 中心负值
            filter_2d = filter_2d - filter_2d.mean()  # 零均值
            for c in range(3):
                filters[i, c] = filter_2d

        return filters

    def forward(self, x):
        return self.conv(x)
```

### 1.2 频域编码器详细设计

```python
class FreqEncoder(nn.Module):
    """频域特征编码器"""
    def __init__(self, in_channels=30, hidden_dim=128, output_dim=128):
        super().__init__()

        # 第一阶段：浅层特征
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 第二阶段：中层特征
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        # 第三阶段：深层特征
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 全局池化和投影
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: [B, 30, H, W]
        x = self.stage1(x)  # [B, 32, H/2, W/2]
        x = self.stage2(x)  # [B, 64, H/4, W/4]
        x = self.stage3(x)  # [B, 64, H/8, W/8]
        x = self.pool(x)    # [B, 64, 1, 1]
        x = self.proj(x)    # [B, output_dim]
        return x
```

### 1.3 原型模块详细设计

```python
class PrototypeModule(nn.Module):
    """原型学习模块 - 借鉴GAPL"""
    def __init__(self, input_dim=640, num_prototypes=16, num_heads=4, dropout=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.num_prototypes = num_prototypes

        # 可学习的原型向量
        self.prototypes = nn.Parameter(torch.Tensor(num_prototypes, input_dim))
        nn.init.xavier_uniform_(self.prototypes)

        # 交叉注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(input_dim)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, input_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(input_dim)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x, return_attn=False):
        """
        Args:
            x: [B, D] 或 [B, N, D]
        Returns:
            logits: [B, 1]
            attn_weights: [B, num_heads, N, num_prototypes]
        """
        # 确保输入是3D
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, D]

        B, N, D = x.shape

        # 扩展原型
        prototypes = self.prototypes.unsqueeze(0).expand(B, -1, -1)  # [B, P, D]

        # 交叉注意力
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
            # 聚合heads和patches
            attn_weights = attn_weights.mean(dim=(1, 2))  # [B, P]
            return logits, attn_weights

        return logits
```

---

## 二、损失函数设计

### 2.1 基础损失

```python
class DetectionLoss(nn.Module):
    """检测损失函数"""
    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, 1]
            targets: [B]
        """
        targets = targets.float()
        if self.label_smoothing > 0:
            # Label smoothing
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        return self.bce(logits.squeeze(), targets)
```

### 2.2 对比学习损失（可选）

```python
class ContrastiveLoss(nn.Module):
    """对比学习损失 - 让真假图像特征分离"""
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: [B, D] (L2归一化前)
            labels: [B] (0=real, 1=fake)
        """
        # L2归一化
        features = F.normalize(features, dim=1)  # [B, D]

        # 分离真假
        real_mask = (labels == 0)
        fake_mask = (labels == 1)

        real_feat = features[real_mask]  # [N_real, D]
        fake_feat = features[fake_mask]  # [N_fake, D]

        if real_feat.size(0) == 0 or fake_feat.size(0) == 0:
            return torch.tensor(0.0, device=features.device)

        # 真实样本之间的相似度（应该高）
        sim_real = torch.mm(real_feat, real_feat.t()) / self.temperature  # [N_real, N_real]

        # 真假样本之间的相似度（应该低）
        sim_cross = torch.mm(real_feat, fake_feat.t()) / self.temperature  # [N_real, N_fake]

        # InfoNCE loss
        # 真实样本应该与真实样本相似，与假样本不相似
        pos_sim = sim_real.diag()  # 正样本对（自己）
        neg_sim = torch.logsumexp(sim_cross, dim=1)  # 负样本

        loss = -pos_sim.mean() + neg_sim.mean()
        return loss
```

---

## 三、训练流程详细设计

### 3.1 训练步骤

```python
def train_one_epoch(model, dataloader, optimizer, scheduler,
                    scaler, ema, config, device, epoch):
    """训练一个epoch"""
    model.train()
    ema_model = ema if ema is not None else None

    # 梯度累积
    accumulation_steps = config.get('accumulation_steps', 1)
    optimizer.zero_grad()

    metric_logger = MetricLogger(delimiter="  ")

    for i, (images, labels) in enumerate(metric_logger.log_every(dataloader, 50, f"Epoch {epoch}")):
        images = images.to(device)
        labels = labels.to(device)

        # 混合精度前向传播
        with torch.cuda.amp.autocast():
            logits, attn_weights = model(images, return_attn=True)
            loss = model.get_criterion(logits, labels)
            loss = loss / accumulation_steps  # 梯度累积

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度累积步数到达时更新
        if (i + 1) % accumulation_steps == 0:
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 更新参数
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # EMA更新
            if ema_model is not None:
                ema_model.update(model)

        # 日志
        metric_logger.update(loss=loss.item() * accumulation_steps)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])

    scheduler.step()
    return metric_logger
```

### 3.2 验证流程

```python
@torch.no_grad()
def validate(model, dataloader, device):
    """验证"""
    model.eval()

    all_logits = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        all_logits.append(logits)
        all_labels.append(labels)

    all_logits = torch.cat(all_logits).cpu()
    all_labels = torch.cat(all_labels).cpu()

    # 计算指标
    probs = torch.sigmoid(all_logits).numpy()
    preds = (probs > 0.5).astype(int)

    acc = accuracy_score(all_labels.numpy(), preds)
    ap = average_precision_score(all_labels.numpy(), probs)

    return {
        'accuracy': acc,
        'ap': ap,
        'predictions': preds,
        'probabilities': probs,
        'labels': all_labels.numpy()
    }
```

---

## 四、数据增强策略

### 4.1 基础增强

```python
from torchvision import transforms

def get_train_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_val_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
```

### 4.2 高级增强（可选）

```python
def get_advanced_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(15),

        # 高级颜色变换
        transforms.ColorJitter(0.2, 0.2, 0.1, 0.05),

        # 高斯模糊（模拟JPEG压缩）
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        ], p=0.3),

        # 随机锐化
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),

        # 随机 Posterize（降低位深度）
        transforms.RandomPosterize(bits=6, p=0.2),

        # 随机 JPEG 压缩
        transforms.RandomApply([
            JPEGCompression(quality=(70, 95))
        ], p=0.3),

        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
```

---

## 五、评估指标

### 5.1 基础指标

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, average_precision_score, roc_auc_score,
    confusion_matrix
)

def compute_metrics(labels, probs, threshold=0.5):
    """计算所有评估指标"""
    preds = (probs > threshold).astype(int)

    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
        'ap': average_precision_score(labels, probs),
        'auc': roc_auc_score(labels, probs)
    }

    # 混淆矩阵
    cm = confusion_matrix(labels, preds)
    metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()

    return metrics
```

### 5.2 生成器级别指标

```python
def compute_per_generator_metrics(labels, probs, generator_names):
    """计算每个生成器的检测准确率"""
    metrics = {}

    for gen in set(generator_names):
        mask = np.array(generator_names) == gen
        gen_labels = np.array(labels)[mask]
        gen_probs = probs[mask]
        gen_preds = (gen_probs > 0.5).astype(int)

        if len(gen_labels) > 0:
            metrics[gen] = {
                'accuracy': accuracy_score(gen_labels, gen_preds),
                'ap': average_precision_score(gen_labels, gen_probs),
                'count': len(gen_labels)
            }

    return metrics
```

---

## 六、可视化工具

### 6.1 原型注意力可视化

```python
def visualize_prototype_attention(model, image, save_path):
    """可视化原型注意力权重"""
    model.eval()

    with torch.no_grad():
        image = image.unsqueeze(0).cuda()
        logits, attn_weights = model(image, return_attn=True)

        # attn_weights: [B, num_prototypes]
        attn_weights = attn_weights[0].cpu().numpy()

    # 绘制
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 原图
    ax1.imshow(image.cpu().squeeze().permute(1, 2, 0))
    ax1.set_title('Input Image')
    ax1.axis('off')

    # 注意力权重
    ax2.bar(range(len(attn_weights)), attn_weights)
    ax2.set_xlabel('Prototype Index')
    ax2.set_ylabel('Attention Weight')
    ax2.set_title('Prototype Attention Weights')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
```

### 6.2 混淆矩阵可视化

```python
def plot_confusion_matrix(cm, class_names, save_path):
    """绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=ax
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
```

---

## 七、模型压缩与优化

### 7.1 模型量化

```python
def quantize_model(model_path, output_path):
    """量化模型以减小大小"""
    import torch.quantization as quant

    # 加载模型
    model = torch.load(model_path)
    model.eval()

    # 准备量化
    model.qconfig = quant.get_default_qconfig('fbgemm')
    quant.prepare(model, inplace=True)

    # 校准（使用验证集）
    # ... calibrate_with_validation_set ...

    # 转换
    quant.convert(model, inplace=True)

    # 保存
    torch.save(model.state_dict(), output_path)
```

### 7.2 模型剪枝

```python
def prune_model(model, sparsity=0.3):
    """剪枝模型参数"""
    import torch.nn.utils.prune as prune

    # 全局剪枝
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            parameters_to_prune.append((module, 'weight'))

    # 全局非结构化剪枝
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity
    )

    return model
```

---

## 八、推理优化

### 8.1 批量推理

```python
def batch_inference(model, image_paths, batch_size=32, device='cuda'):
    """批量推理"""
    model.eval()

    results = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []

        for path in batch_paths:
            img = Image.open(path).convert('RGB')
            img = val_transform(img)
            batch_images.append(img)

        batch_tensor = torch.stack(batch_images).to(device)

        with torch.no_grad():
            logits = model(batch_tensor)
            probs = torch.sigmoid(logits).squeeze()

        for path, prob in zip(batch_paths, probs.cpu().numpy()):
            results.append({
                'path': path,
                'is_fake': prob > 0.5,
                'confidence': prob,
                'label': 'Fake' if prob > 0.5 else 'Real'
            })

    return results
```

### 8.2 ONNX导出

```python
def export_to_onnx(model, onnx_path, input_size=(1, 3, 224, 224)):
    """导出到ONNX格式"""
    model.eval()

    dummy_input = torch.randn(*input_size).cuda()

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        input_names=['image'],
        output_names=['logits'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )
```

---

## 九、实验记录模板

```markdown
# 实验记录

## 实验 001: 基线模型

### 配置
- CLIP模型: ViT-B/32
- 原型数量: 16
- 使用频域: True
- Batch size: 16
- 学习率: 1e-4
- Epochs: 30

### 结果
- 验证集准确率: 87.3%
- 测试集准确率: 85.2%
- 训练时间: 5.5小时

### 观察与改进
- [ ] 原型注意力可视化
- [ ] 尝试增加原型数量到32
- [ ] 尝试添加对比学习损失

## 实验 002: 增加原型数量
...
```

---

**文档版本**: v1.0
**最后更新**: 2025-06-07
