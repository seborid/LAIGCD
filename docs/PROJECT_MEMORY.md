# LAIGCD 项目记忆

更新日期：2026-06-10

## 1. 这是什么项目

项目正式题目是：

- **基于空域语义分析和频域分析的轻量化、可解释性 AIGC 人脸检测系统**

项目应按两个部分理解：

- **第一部分：真伪判断** ✅ **已完成**
  - 目标：判断输入人脸图像是真实还是 AIGC / deepfake 伪造
  - 当前主实现：`CLIP ViT-B/32` 空域语义特征 + `SRM` 频域分支 + 原型学习分类头
  - **测试集准确率**: 99.13%
  - **AP**: 0.9996
- **第二部分：多模态可解释性** ⏳ **待实现**
  - 目标：在真假判断结果之外，结合原图、热力图、频域响应等信息生成自然语言解释
  - 当前状态：已有设计文档和思路，但尚未形成完整可运行主链路

LAIGCD 的实现已经明显收敛到一个更具体的方向：

- **当前真实定位**：轻量级 **AIGC / Deepfake 人脸检测系统**
- **当前主要数据源**：
  - `140k-real-and-fake-faces`
  - `130k-real-vs-fake-face`
- **当前核心方法**：
  - 冻结的 `CLIP ViT-B/32` 空域特征
  - `SRM` 频域分支
  - `PrototypeModule` 原型学习分类头

结论：后续讨论项目时，应把它视为"**AIGC 人脸检测 + 多模态可解释性**"的两阶段系统；其中第一阶段**已完成**，第二阶段仍主要停留在设计与扩展阶段。

## 2. 项目路线演进

### 阶段A：总体方案规划

先确定了项目的正式结构：

- 第一阶段做 **真假检测**
- 第二阶段做 **多模态可解释性**

其中整体技术主线是：

- 空域语义分析
- 频域分析
- 轻量化实现
- 可解释性增强

随后完成路线设计和技术文档，第一阶段核心设计是：

- 用冻结 CLIP 降低训练成本
- 用轻量频域模块补强伪造痕迹检测
- 用原型学习增强可解释性和泛化

对应文档：

- `PROJECT_PLAN.md`
- `TECHNICAL_DESIGN.md`
- `TODO.md`

### 阶段B：第一阶段核心代码落地

核心模型和训练骨架已经写完：

- `models/freq_module.py`
- `models/prototype.py`
- `models/detector.py`
- `utils/train.py`
- `utils/metrics.py`
- `utils/viz.py`

这一阶段已经完成第一部分从"设计"到"可训练代码"的转换。

### 阶段C：第一阶段数据集适配与工程化

项目从理想化的 `data/train|val/...` 目录，演进成直接兼容实际 Kaggle 数据集结构：

- `utils/data.py` 已支持 140k 与 130k 两套数据结构
- 当前仓库未包含自动下载脚本，数据需手动放置到 `data/` 目录
- `scripts/train.py` / `scripts/inference.py` / `scripts/eval.py` 已形成训练、推理、评估入口
- 增加了日志、checkpoint、结果可视化脚本

这一阶段是项目从"模型原型"走向"能跑完整实验"的关键阶段。

### 阶段D：第一阶段实验推进

已经完成多轮真实训练：

- `quick_run`
- `standard_run`
- `full_run` ✅ **最终模型**

训练规模从小样本验证代码，推进到全量训练。

### 阶段E：第一阶段评估与问题修复 ✅ **已完成**

- 修复了 `eval.py` 的数据解包问题（支持字典格式）
- 修复了 `torch.load` 的 `weights_only` 参数问题
- 确认模型在测试集上达到 **99.13% 准确率**
- 生成完整评估报告：`ASSETS_EVALUATION_REPORT.md`

## 3. 当前实际进度

### ✅ 已完成

**第一阶段（真假判断检测器）**：
- ✅ 核心模型实现完成
- ✅ 双数据集加载完成
- ✅ 训练循环、EMA、AMP、checkpoint 保存完成
- ✅ 推理脚本完成
- ✅ 可视化脚本完成
- ✅ 数据集已下载到本地（~17GB）
- ✅ 30轮全量训练完成
- ✅ **测试集评估完成**：Accuracy 99.13%, AP 0.9996
- ✅ **评估脚本修复完成**
- ✅ **生成评估报告**

**第一阶段文档**：
- ✅ `ASSETS_EVALUATION_REPORT.md` - 评估报告

**第二阶段设计文档**：
- ✅ `EXPLAINABILITY_DESIGN.md`
- ✅ `EXPLAINABILITY_ANALYSIS.md`

### ⏳ 进行中 / 未完成

**第二阶段（多模态可解释性）**：
- ⏳ 原型注意力可视化
- ⏳ 空域热力图生成
- ⏳ 频域热力图生成
- ⏳ FakeVLM 自然语言解释链路
- ⏳ Web 演示界面
- ⏳ API / 部署

## 4. 第一阶段实验快照

### 训练配置 (full_run)

| 参数 | 值 |
|------|-----|
| Epochs | 30 |
| Batch Size | 64 |
| Learning Rate | 0.0001 |
| Optimizer | AdamW |
| AMP | 启用 |
| EMA | 启用 (decay=0.9999) |

### 最终性能 (测试集)

| 指标 | 值 |
|------|-----|
| **Accuracy** | **99.13%** |
| **AP** | **0.9996** |
| **AUC** | **0.9996** |
| **Precision** | **99.01%** |
| **Recall** | **99.22%** |
| **F1 Score** | **0.9912** |

### 混淆矩阵 (测试集 33,355 样本)

| | 预测Real | 预测Fake |
|---|----------|----------|
| **真实Real** | 16,838 | 162 (FP) |
| **真实Fake** | 127 (FN) | 16,228 |

### 概率分布 (验证集)

- **真实图片预测概率**: 平均 0.0114 (接近0)
- **伪造图片预测概率**: 平均 0.9945 (接近1)
- **最优阈值**: 0.68
- **最优阈值 Accuracy**: 99.34%

### 数据集规模

- `data/140k-real-and-fake-faces`: ~4.1GB
- `data/130k-real-vs-fake-face`: ~13GB
- `checkpoints/`: ~21GB
- 训练时间: ~2小时 (30 epochs)

## 5. 关键判断更新

### 判断1：第一阶段已完成 ✅

模型在测试集上达到 99.13% 准确率，AP=0.9996，完全满足第一阶段目标。

### 判断2：模型质量极高

- 真实和伪造图片的概率分布分离度极高
- 对多种生成器（FLUX_DEV、FLUX_PRO、SDXL）均有良好表现
- 训练稳定，从第10轮开始 AP>0.999

### 判断3：第二阶段是下一步重点

现在应专注于实现可解释性模块，为答辩演示准备。

## 6. 后续优先级

### P0：答辩准备（立即）

1. 准备答辩PPT，展示第一阶段成果
2. 准备现场演示（单图检测）

### P1：第二阶段可解释性

1. 实现原型注意力可视化
2. 实现空域+频域热力图
3. 简易演示界面

### P2：完整第二阶段（可选）

1. FakeVLM 多模态解释
2. 完整 Web 演示界面

### P3：工程化（可选）

1. API 部署
2. ONNX / TensorRT 优化

## 7. 关键入口文件

- **模型主入口**: `models/detector.py`
- **数据主入口**: `utils/data.py`
- **训练主入口**: `scripts/train.py`
- **推理主入口**: `scripts/inference.py`
- **评估主入口**: `scripts/eval.py` ✅ 已修复
- **阈值诊断**: `scripts/diagnose_threshold.py`
- **最佳模型**: `checkpoints/full_run/best_model.pth`
- **评估报告**: `docs/ASSETS_EVALUATION_REPORT.md`

## 8. 快速使用指南

### 推理单张图片

```bash
python scripts/inference.py \
  /path/to/image.jpg \
  --checkpoint checkpoints/full_run/best_model.pth
```

### 评估测试集

```bash
python scripts/eval.py \
  --checkpoint checkpoints/full_run/best_model.pth \
  --data_path data \
  --split test \
  --output_dir checkpoints/full_run/eval_results
```

### 阈值诊断

```bash
python scripts/diagnose_threshold.py
```

## 9. 后续维护约定

以后只要发生以下变化，优先更新本文件：

- 第二阶段多模态解释主链路落地
- 项目定位从"课程项目原型"升级成"可演示/可部署版本"
- 答辩完成并获得反馈

这份文件的目标不是代替设计文档，而是作为 **项目当前真实状态的持久化记忆**。

---

**第一阶段状态总结**：模型训练完成，测试集准确率 99.13%，第一阶段任务圆满完成！
