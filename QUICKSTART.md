# LAIGCD 快速开始指南

## 第一步：环境搭建

```bash
# 创建项目目录（如果还没有）
cd /home/seborid/deepfake/LAIGCD

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装PyTorch (CUDA 11.8版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install open-clip-torch transformers pytorch-wavelets scikit-learn matplotlib tqdm pillow
```

## 第二步：创建项目结构

```bash
# 创建子目录
mkdir -p models utils scripts data/train data/val checkpoints results

# 创建空的__init__.py文件
touch models/__init__.py
touch utils/__init__.py
```

## 第三步：准备数据

数据集结构应如下所示：

```
data/
├── train/
│   ├── real/          # 真实图像
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── fake/          # AI生成图像
│       ├── fake1.jpg
│       ├── fake2.jpg
│       └── ...
└── val/
    ├── real/          # 验证用真实图像
    └── fake/          # 验证用AI图像
```

**数据集推荐（总计约100GB）：**

1. **真实图像** (~30GB)
   - ImageNet子集
   - COCO子集

2. **AI生成图像** (~70GB)
   - GenImage (SDv1.4, SDXL, Midjourney)
   - CNNSpot (ProGAN, StyleGAN)

## 第四步：实现核心模块

按以下顺序实现：

1. **models/freq_module.py** - 频域特征模块
2. **models/prototype.py** - 原型学习模块
3. **models/detector.py** - 完整检测器
4. **utils/data.py** - 数据加载器
5. **utils/train.py** - 训练函数
6. **scripts/train.py** - 训练脚本

## 第五步：开始训练

```bash
# 修改scripts/train.sh中的数据路径
bash scripts/train.sh
```

## 第六步：评估模型

```bash
# 在测试集上评估
python scripts/eval.py --checkpoint checkpoints/best.pth --data_path data/test/
```

## 第七步：推理示例

```bash
# 单图推理
python scripts/inference.py --image_path test.jpg --checkpoint checkpoints/best.pth

# 批量推理
python scripts/inference.py --image_dir test_images/ --checkpoint checkpoints/best.pth --output results.json
```

---

## 常见问题

### Q1: 显存不足怎么办？

A: 减小batch_size或使用梯度累积：
```python
config = {
    'batch_size': 8,           # 从16降到8
    'accumulation_steps': 4,  # 从2增到4，等效batch=32
}
```

### Q2: 如何调整模型大小？

A: 调整原型数量：
```python
model = LightweightAIGCDetector(
    num_prototypes=8,  # 默认16，可改为8或32
)
```

### Q3: 训练不收敛怎么办？

A: 尝试以下调整：
1. 降低学习率：`1e-4 → 5e-5`
2. 增加warmup轮数：`warmup_epochs=5`
3. 增加数据增强

### Q4: 如何可视化原型注意力？

A: 使用可视化工具：
```python
from utils.viz import visualize_prototype_attention

visualize_prototype_attention(model, image, 'attention.png')
```

---

## 下一步

详细技术设计请参考：
- [PROJECT_PLAN.md](PROJECT_PLAN.md) - 项目计划
- [TECHNICAL_DESIGN.md](TECHNICAL_DESIGN.md) - 技术设计
- [TODO.md](TODO.md) - 实现清单

祝项目顺利！🚀
