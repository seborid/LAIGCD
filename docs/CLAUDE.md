# Deepfake检测项目

## 项目概述

这是一个专注于**AI生成图像检测（Deepfake检测）**的课程项目。目标是构建一个轻量级检测器，使用深度学习技术区分真实图像和AI生成图像。

**课程要求：**
- 运用神经网络、强化学习等课堂所学技术解决具体问题
- 必须提供现场运行演示
- 具有原创性（可参考开源代码但需明确说明贡献部分）

---

## 项目状态总结

### 主项目：LAIGCD (轻量级AI生成内容检测器)

**状态**: 🟢 实现完成，已训练，待测试

**位置**: `/home/seborid/deepfake/LAIGCD/`

**已完成：**
- ✅ 核心模型已实现（检测器、频域模块、原型学习模块）
- ✅ 训练脚本已创建（`train.py`, `train.sh`, `eval.py`, `inference.py`）
- ✅ 模型训练成功 - 检查点已保存：`best_model.pth` (1.2GB)
- ✅ CIFake数据集已下载并组织好
- ✅ 文档完整（PROJECT_PLAN.md, TODO.md, TECHNICAL_DESIGN.md等）

**待完成：**
- ⏳ 在测试集上评估模型性能
- ⏳ 生成性能指标和混淆矩阵
- ⏳ 制作演示界面（用于答辩展示）
- ⏳ Web界面（可选，但推荐用于演示）

---

## 技术架构

```
输入图像 [3, 224, 224]
    ↓
    ├── CLIP ViT-B/32 (冻结) → 空域特征 [512]
    │
    └── 频域模块 (可训练) → 频域特征 [128]
         └── SRM卷积 + 轻量CNN
    ↓
特征融合 [640]
    ↓
原型学习模块
    └── 16个可学习原型
    └── 交叉注意力机制
    ↓
二分类器 → 真实(0) / 伪造(1)
```

**核心特性：**
- **轻量化**: 仅~80万可训练参数
- **高效**: CLIP冻结，只训练原型和频域模块
- **可解释**: 原型注意力可视化
- **快速**: >30 FPS推理速度

---

## 目录结构

```
deepfake/
├── LAIGCD/                    # 主项目（正在开发）
│   ├── models/                # 模型实现
│   │   ├── detector.py       # 完整检测器
│   │   ├── freq_module.py   # 频域特征提取
│   │   └── prototype.py      # 原型学习
│   ├── scripts/              # 训练/评估脚本
│   │   ├── train.py
│   │   ├── eval.py
│   │   └── inference.py
│   ├── checkpoints/          # 训练好的模型
│   │   └── best_model.pth   # 1.2GB训练好的检查点
│   ├── data/                 # 数据集
│   │   └── cifake/          # CIFake数据集（训练/测试）
│   ├── PROJECT_PLAN.md       # 项目计划书
│   ├── TODO.md               # 实现清单
│   └── requirements.txt
│
├── GAPL/                     # 参考：原型学习实现
│   └── checkpoint.pt        # 1.2GB（参考用）
│
├── GenImage/                 # 参考：GenImage基准数据集
│   ├── detector_codes/       # 检测方法
│   ├── generator_codes/      # 生成器实现
│   └── index.html           # Web演示
│
├── IAPL/                     # 参考：图像增强
│   └── augmix.py            # AugMix实现
│
└── FakeVLM/                  # 参考：Fake VLM检测
    └── train.py
```

---

## 快速开始

### 运行推理

```bash
cd /home/seborid/deepfake/LAIGCD

# 测试单张图片
python scripts/inference.py \
    --image_path /path/to/image.jpg \
    --checkpoint checkpoints/best_model.pth
```

### 评估模型

```bash
# 在测试集上评估
python scripts/eval.py \
    --checkpoint checkpoints/best_model.pth \
    --data_dir data/cifake/test
```

### 训练新模型（如需要）

```bash
bash scripts/train.sh
```

---

## 参考项目说明

1. **GAPL** - 原型学习 + 交叉注意力机制
2. **IAPL** - 频域分析 + DCT条件模块
3. **GenImage** - 百万级AIGC检测基准数据集

**重要说明**：这些是参考实现。本项目（LAIGCD）结合了两者思路并做了轻量化改进。

---

## 两阶段可解释性设计

**详见**: [LAIGCD/EXPLAINABILITY_DESIGN.md](LAIGCD/EXPLAINABILITY_DESIGN.md)

**核心思想**：检测器 + 解释器分离

**第一阶段**：轻量化检测器 + 热力图生成
- 输入：人脸图像
- 输出：真假判断 + 置信度 + 空域热力图 + 频域热力图

**第二阶段**：预训练VLM生成自然语言解释
- 输入：原图 + 两张热力图（3张图像）
- 输出：可解释性文字说明
- 技术：FakeVLM零样本/少样本推理（无需重新训练）

---

## 下一步工作（按优先级）

1. **实现第一阶段** - 修改detector添加热力图输出（空域+频域）
2. **测试FakeVLM零样本** - 验证第二阶段可行性
3. **创建演示界面** - 集成两个阶段的完整系统
4. **准备答辩材料** - PPT + 现场演示

---

## 硬件要求

- **GPU**: NVIDIA RTX 4070Ti（或类似）
- **显存**: 训练~4GB，推理~2GB
- **训练时间**: 约6小时（30轮）
- **数据量**: CIFake数据集约数GB

---

## 演示建议

1. **实时检测**: 上传图片显示真假预测
2. **注意力可视化**: 显示哪些原型被激活
3. **频域分析**: 对比真实图和假图的频域特征
4. **批量测试**: 测试多个不同生成器的图片

---

## 备注

- 创建时间: 2026年6月
- 课程: 深度学习课程项目
- 目标: 展示对神经网络和AIGC检测的理解

**最后更新**: 2026-06-08
