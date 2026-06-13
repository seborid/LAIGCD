# LAIGCD 协作文档

更新日期：2026-06-10

## 项目定位

LAIGCD 是一个两阶段课程项目：

- 第一阶段：AIGC / Deepfake 人脸真伪检测
- 第二阶段：多模态可解释性输出

当前仓库的真实状态是：**第一阶段已完成，第二阶段待实现。**

## 当前状态

### 第一阶段

- 主模型：`CLIP ViT-B/32` 空域特征 + `SRM` 频域分支 + `PrototypeModule`
- 数据集：`140k-real-and-fake-faces`、`130k-real-vs-fake-face`
- 训练闭环：已完成
- 推理闭环：已完成
- 评估闭环：已完成
- 最佳模型：`checkpoints/full_run/best_model.pth`
- 测试集结果：`Accuracy 99.13%`，`AP 0.9996`，`AUC 0.9996`

### 第二阶段

- 已有设计文档
- 尚未打通空域热力图、频域热力图、自然语言解释和展示界面的完整链路

## 关键入口

- 模型：`models/detector.py`
- 数据：`utils/data.py`
- 训练：`scripts/train.py`
- 推理：`scripts/inference.py`
- 评估：`scripts/eval.py`
- 阈值诊断：`scripts/diagnose_threshold.py`
- 可视化：`scripts/visualize_results.py`

## 正确用法

### 单图推理

```bash
python3 scripts/inference.py /path/to/image.jpg \
  --checkpoint checkpoints/full_run/best_model.pth
```

### 目录推理

```bash
python3 scripts/inference.py \
  --checkpoint checkpoints/full_run/best_model.pth \
  --image_dir /path/to/images \
  --output results.json
```

### 测试集评估

```bash
python3 scripts/eval.py \
  --checkpoint checkpoints/full_run/best_model.pth \
  --data_path data \
  --split test
```

## 文档分工

- `README.md`：对外说明与当前项目概览
- `docs/PROJECT_MEMORY.md`：当前真实状态与阶段结论
- `docs/ASSETS_EVALUATION_REPORT.md`：第一阶段测试评估结果
- `docs/PROJECT_PLAN.md`：早期规划，不代表最新进度
- `docs/TODO.md`：当前任务清单

## 下一步重点

1. 打通第二阶段最小可演示链路
2. 完善原型注意力和热力图可视化
3. 准备答辩展示界面或命令行演示流程
