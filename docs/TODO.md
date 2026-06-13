# LAIGCD 当前任务清单

更新日期：2026-06-13

## 说明

本文件用于记录当前真实进度和下一步任务，已经替代早期"从零实现"式清单。

- 当前状态以 `README.md` 和 `docs/PROJECT_MEMORY.md` 为准
- 早期规划与设计细节见 `docs/PROJECT_PLAN.md`

## 当前结论

- 第一阶段真伪检测器已经完成训练、评估、推理闭环
- 当前最佳模型为 `checkpoints/full_run/best_model.pth`
- 第一阶段测试集结果：`Accuracy 99.13%`，`AP 0.9996`，`AUC 0.9996`
- 第二阶段多模态可解释性框架已完成，包括热力图生成和FakeVLM自然语言解释

## 已完成

### 第一阶段：检测器
- [x] 核心模型实现：`CLIP ViT-B/32` + `SRM` + `PrototypeModule`
- [x] 双数据集适配：`140k-real-and-fake-faces`、`130k-real-vs-fake-face`
- [x] 训练入口：`scripts/train.py`、`scripts/train_optimized.sh`
- [x] 推理入口：`scripts/inference.py`
- [x] 评估入口：`scripts/eval.py`
- [x] 阈值诊断：`scripts/diagnose_threshold.py`
- [x] 训练结果可视化：`scripts/visualize_results.py`
- [x] 多轮训练产物沉淀：`quick_run`、`standard_run`、`full_run`
- [x] 第一阶段测试集评估与评估报告

### 第二阶段：可解释性
- [x] 空域热力图生成功能
- [x] 频域热力图生成功能
- [x] 原型注意力可视化
- [x] 解释脚本：`scripts/explain.py`
- [x] **FakeVLM解释器模块**：`models/fakevlm_explainer.py`
- [x] **自然语言解释脚本**：`scripts/nl_explain.py`
- [x] **配置管理模块**：`configs/fakevlm_config.py`
- [x] explain.py集成FakeVLM自然语言解释功能

## 进行中

- [ ] FakeVLM模型下载和测试验证
- [ ] 准备答辩展示案例和演示流程

## 未完成

- [ ] Web演示界面
- [ ] API/部署
- [ ] 自动化测试
- [ ] 完整消融实验
- [ ] ONNX/TensorRT/量化优化

## 下一步优先级

### P0 - 答辩准备

- [ ] 验证FakeVLM集成功能（下载模型并测试）
- [ ] 准备3-5个典型案例的完整解释输出
- [ ] 整理答辩演示流程

### P1 - 功能完善

- [ ] 优化热力图可视化效果
- [ ] 调整自然语言解释prompt以获得更好的解释质量

### P2 - 扩展功能

- [ ] 构建简易Web界面
- [ ] 添加更多预设配置

## FakeVLM 集成说明

### 新增文件

- `models/fakevlm_explainer.py` - FakeVLM解释器核心模块
- `scripts/nl_explain.py` - 独立的自然语言解释脚本
- `configs/fakevlm_config.py` - 配置管理

### 使用方法

1. **单图解释（含自然语言）**：
```bash
python scripts/explain.py image.jpg --use_fakevlm
```

2. **仅生成自然语言解释**：
```bash
python scripts/nl_explain.py image.jpg
python scripts/nl_explain.py image.jpg --spatial spatial.png --frequency freq.png
```

3. **使用8bit量化（节省显存）**：
```bash
python scripts/nl_explain.py image.jpg --load_in_8bit
```

4. **批量处理**：
```bash
python scripts/nl_explain.py --image_dir images/
```

### 依赖说明

FakeVLM需要以下依赖（可选）：
- `transformers>=4.45.0`
- `flash-attn>=2.0.0` (可选，用于加速)
- `bitsandbytes>=0.41.0` (可选，用于8bit量化)

首次运行会自动从HuggingFace下载 `llava-hf/llava-1.5-7b-hf` 模型（约13GB）。

## 参考文档

- `docs/PROJECT_MEMORY.md`：项目最新状态
- `docs/ASSETS_EVALUATION_REPORT.md`：第一阶段评估结果
- `docs/EXPLAINABILITY_DESIGN.md`：第二阶段设计方案
- `docs/PROJECT_PLAN.md`：项目早期规划
