# LAIGCD: 基于空域语义分析和频域分析的轻量化、可解释性 AIGC 人脸检测系统

## 项目简介

LAIGCD 是一个面向 **AIGC / Deepfake 人脸检测** 的两阶段系统，目标是在较低训练成本下完成：

- **第一阶段：真假判断**
  - 输入人脸图像
  - 输出真实 / 伪造分类结果
- **第二阶段：多模态可解释性**
  - 结合原图、空域热力图、频域响应图、原型激活等信息
  - 生成可视化和自然语言解释

当前仓库的主要实现集中在第一阶段，第二阶段已有设计文档，尚未形成完整可运行主链路。

## 题目定义

正式题目：

**基于空域语义分析和频域分析的轻量化、可解释性 AIGC 人脸检测系统**

关键词：

- 空域语义分析
- 频域分析
- 轻量化
- 可解释性
- AIGC 人脸检测

## 技术路线

第一阶段的主干模型为：

```text
输入人脸图像
  -> CLIP ViT-B/32 空域语义特征（冻结）
  -> SRM 频域特征分支（可训练）
  -> 特征融合
  -> Prototype Module 原型学习分类头
  -> 真 / 伪判断
```

第二阶段的目标链路为：

```text
原图 + 空域热力图 + 频域热力图 + 原型激活
  -> 多模态解释模块
  -> 输出自然语言解释与可视化说明
```

## 当前进展

### 已完成

- 第一阶段检测模型实现
- 双数据集加载与统一封装
- 训练、验证、推理脚本
- 日志、checkpoint、结果可视化
- 原型注意力相关可解释性基础能力
- 本地数据下载与多轮训练产物沉淀

### 进行中

- 验证阶段阈值校准
- 排查 `AP` 高但 `Accuracy` 接近 `0.5` 的原因
- 让第一阶段输出稳定服务于第二阶段解释模块

### 未完成

- 第二阶段多模态解释主链路
- 自动化测试
- API / Web 部署
- ONNX / TensorRT / 量化优化
- 完整消融实验与正式报告

## 当前实验状态

现有训练结果表明：

- 小样本 `quick_run` 仅验证了训练链路可运行
- `standard_run` 与 `full_run` 的 `AP` 已经很高
- 但默认阈值下 `Accuracy` 仍异常接近 `0.5`

这说明当前最优先的问题不是“模型完全没学到”，而是：

- 决策阈值可能未校准
- 概率分布可能整体偏移
- 评估脚本或评估口径需要继续核查

## 数据集

当前实现主要使用两个 Kaggle 数据集：

- `140k-real-and-fake-faces`
- `130k-real-vs-fake-face`

本项目不是强依赖统一的 `train/val/test` 简单目录，而是已经在 `utils/data.py` 中适配了这两套真实数据结构。

数据下载脚本：

```bash
python3 datadownload.py
```

## 安装依赖

建议使用 Python 3.10+。

```bash
pip install -r requirements.txt
```

如需单独安装 PyTorch，可根据 CUDA 版本自行选择官方源。

## 训练

标准训练入口：

```bash
python3 scripts/train.py --data_path data --output_dir checkpoints/run1
```

优化训练脚本：

```bash
bash scripts/train_optimized.sh quick
bash scripts/train_optimized.sh standard
bash scripts/train_optimized.sh full
```

说明：

- `quick`：快速验证代码和收敛趋势
- `standard`：中规模实验
- `full`：全量训练

## 推理

单图推理：

```bash
python3 scripts/inference.py \
  --checkpoint checkpoints/full_run/best_model.pth \
  --image path/to/image.jpg
```

目录批量推理：

```bash
python3 scripts/inference.py \
  --checkpoint checkpoints/full_run/best_model.pth \
  --image_dir path/to/images \
  --output results.json
```

## 评估与诊断

评估入口：

```bash
python3 scripts/eval.py --checkpoint checkpoints/full_run/best_model.pth --data_path data
```

阈值诊断：

```bash
python3 scripts/diagnose_threshold.py
```

结果可视化：

```bash
python3 scripts/visualize_results.py --checkpoint_dir checkpoints/full_run
```

## 项目结构

```text
LAIGCD/
├── README.md
├── PROJECT_MEMORY.md
├── PROJECT_PLAN.md
├── TECHNICAL_DESIGN.md
├── EXPLAINABILITY_DESIGN.md
├── EXPLAINABILITY_ANALYSIS.md
├── INNOVATION_IDEAS.md
├── models/
│   ├── detector.py
│   ├── freq_module.py
│   └── prototype.py
├── utils/
│   ├── data.py
│   ├── train.py
│   ├── metrics.py
│   └── viz.py
├── scripts/
│   ├── train.py
│   ├── train.sh
│   ├── train_optimized.sh
│   ├── eval.py
│   ├── inference.py
│   ├── diagnose_threshold.py
│   └── visualize_results.py
├── data/
└── checkpoints/
```

## 文档导航

- [PROJECT_MEMORY.md](PROJECT_MEMORY.md)：项目路线、现状与持久化记忆
- [PROJECT_PLAN.md](PROJECT_PLAN.md)：项目总体规划
- [TECHNICAL_DESIGN.md](TECHNICAL_DESIGN.md)：第一阶段技术设计
- [EXPLAINABILITY_DESIGN.md](EXPLAINABILITY_DESIGN.md)：第二阶段可解释性方案
- [EXPLAINABILITY_ANALYSIS.md](EXPLAINABILITY_ANALYSIS.md)：可解释性扩展分析
- [TODO.md](TODO.md)：原始实现清单

## 核心参考

- GAPL：原型学习与交叉注意力思路
- 图像取证中的 SRM 高频滤波思路
- CLIP 视觉语义特征作为冻结空域 backbone

## 当前注意事项

- `README` 描述的是项目的正式目标与当前真实进展，不代表第二阶段已经完成
- 当前第一阶段结果的主要问题集中在阈值和评估判读，而不是训练链路缺失
- 若要了解最新状态，优先参考 `PROJECT_MEMORY.md`

## License

MIT License
