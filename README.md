# LAIGCD: Lightweight AI-Generated Content Detector

## 项目简介

LAIGCD是一个轻量级的AI生成图像检测系统，旨在在小数据集（<100GB）和单卡（如4070Ti）条件下实现高效的深度伪造检测。

## 核心特性

- ✅ **轻量化**: 仅~800K可训练参数
- ✅ **高效**: 4070Ti单卡6小时完成训练
- ✅ **准确**: 多种生成器检测准确率>85%
- ✅ **可解释**: 原型注意力可视化
- ✅ **实用**: 支持批量推理和API部署

## 技术架构

```
CLIP空域特征 (冻结) + 频域特征 (可训练) → 原型学习模块 → 二分类输出
```

## 快速开始

### 安装依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install open-clip-torch transformers pytorch-wavelets scikit-learn matplotlib tqdm
```

### 数据准备

```bash
# 数据集结构
data/
├── train/
│   ├── real/  # 真实图像
│   └── fake/  # AI生成图像
└── val/
    ├── real/
    └── fake/
```

### 训练

```bash
bash scripts/train.sh
```

### 推理

```bash
python inference.py --image_path test.jpg --checkpoint checkpoints/best.pth
```

## 项目结构

```
LAIGCD/
├── PROJECT_PLAN.md          # 项目计划
├── TECHNICAL_DESIGN.md      # 技术设计
├── TODO.md                  # 实现检查清单
├── models/                  # 模型定义
│   ├── __init__.py
│   ├── freq_module.py       # 频域特征模块
│   ├── prototype.py         # 原型学习模块
│   └── detector.py          # 完整检测器
├── utils/                   # 工具函数
│   ├── __init__.py
│   ├── data.py              # 数据加载
│   ├── train.py             # 训练函数
│   ├── metrics.py           # 评估指标
│   └── viz.py               # 可视化工具
├── scripts/                 # 运行脚本
│   ├── train.sh
│   ├── eval.sh
│   └── inference.sh
├── data/                    # 数据集目录
├── checkpoints/             # 模型检查点
└── results/                 # 实验结果
```

## 性能指标

| 指标 | 目标值 |
|------|--------|
| 可训练参数 | ~800K |
| 显存占用 | ~4GB (训练) |
| 训练时间 | ~6小时 (4070Ti) |
| 推理速度 | ~50 FPS |
| 检测准确率 | >85% |

## 文档导航

1. **[PROJECT_PLAN.md](PROJECT_PLAN.md)** - 项目计划与路线图
2. **[TECHNICAL_DESIGN.md](TECHNICAL_DESIGN.md)** - 详细技术设计
3. **[TODO.md](TODO.md)** - 实现检查清单

## 核心设计思想

本项目借鉴了以下两个优秀开源项目的核心思想：

- **[GAPL](https://github.com/AbyssLumine/GAPL)**: 原型学习 + 交叉注意力机制
- **[IAPL](https://github.com/xxx/IAPL)**: 频域特征分析 + DCT条件模块

## 致谢

- CLIP模型来自 [OpenAI](https://openai.com/research/clip)
- 预训练权重来自 [LAION](https://laion.ai/blog/laion-5b/)

## License

MIT License
