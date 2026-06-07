# LAIGCD 数据准备指南

## 推荐数据源

### 1. GenImage 数据集（推荐）

GenImage 是目前最全面的AI生成图像检测数据集，包含多种生成器的图像。

**下载方式：**
```bash
# 克隆仓库
git clone https://github.com/GenImage-Dataset/GenImage.git

# GenImage 数据集包含：
# - SDv1.4 (Stable Diffusion v1.4)
# - ADM (Diffusion Model)
# - BigGAN
# - Glide
# - VQDM
# - 等
```

**目录结构（下载后）：**
```
GenImage/
└── GenImage/
    └── train/
        ├── SDv14/
        │   ├── 0_real/
        │   └── 1_fake/
        ├── ADM/
        │   ├── 0_real/
        │   └── 1_fake/
        └── ...
```

**转换为LAIGCD格式：**
```bash
# 创建目标目录
mkdir -p data/train/real data/train/fake
mkdir -p data/val/real data/val/fake

# 从GenImage复制数据
# 示例：从SDv14复制
cp -r GenImage/GenImage/train/SDv14/0_real/* data/train/real/
cp -r GenImage/GenImage/train/SDv14/1_fake/* data/train/fake/

# 从其他生成器复制fake图像
cp -r GenImage/GenImage/train/ADM/1_fake/* data/train/fake/
```

---

### 2. UniversalFakeDetect 数据集

传统的GAN生成图像检测数据集。

**下载方式：**
```bash
git clone https://github.com/WisconsinAIVision/UniversalFakeDetect.git
```

**包含的生成器：**
- ProGAN
- StyleGAN
- BigGAN
- CycleGAN
- StarGAN
- 等

---

### 3. 从HuggingFace下载

**Community Forensics 数据集：**
```bash
# 安装huggingface-cli
pip install huggingface-hub

# 下载数据集
huggingface-cli download OwensLab/CommunityForensics-Small --repo-type dataset --local-dir community_forensics
```

---

### 4. 真实图像数据

**ImageNet子集：**
```bash
# 使用ImageNet-1k的部分类别
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar

# 或使用更小的ImageNet子集
# 一些研究团队会发布ImageNet的10类/50类子集
```

**COCO数据集：**
```bash
# 下载COCO数据集
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip

# 复制部分图像作为真实图像
cp train2017/*.jpg data/train/real/
```

---

### 5. 在线生成（小规模测试）

如果你想快速测试，可以使用在线工具生成少量图像：

**使用在线API：**
- OpenAI DALL-E API
- Midjourney Discord
- Stable Diffusion WebUI

**或使用预生成的数据集：**
- [DiffusionDB](https://github.com/poloclub/DiffusionDB) - 2000+张SD生成图像
- [MJ-Bench](https://github.com/kohya-ss/MJ-Bench) - Midjourney生成图像

---

## 快速准备方案

### 方案A：使用GenImage（推荐）

```bash
# 1. 下载GenImage
cd /home/seborid/deepfake
git clone https://github.com/GenImage-Dataset/GenImage.git

# 2. 创建LAIGCD数据目录
mkdir -p LAIGCD/data/train/real LAIGCD/data/train/fake
mkdir -p LAIGCD/data/val/real LAIGCD/data/val/fake

# 3. 复制SDv14数据（约10GB）
cp GenImage/GenImage/train/SDv14/0_real/* LAIGCD/data/train/real/
cp GenImage/GenImage/train/SDv14/1_fake/* LAIGCD/data/train/fake/

# 4. 从其他生成器补充fake图像
cp GenImage/GenImage/train/ADM/1_fake/* LAIGCD/data/train/fake/
cp GenImage/GenImage/train/BigGAN/1_fake/* LAIGCD/data/train/fake/

# 5. 分割验证集（从训练集中取10%）
python scripts/split_val.py --data_path LAIGCD/data --val_ratio 0.1
```

### 方案B：最小化数据集（快速测试）

```bash
# 只需要约1000张图像即可开始测试

# 1. 创建目录
mkdir -p data/train/real data/train/fake

# 2. 从ImageNet下载100张真实图像
# (可以使用wget批量下载)

# 3. 从DiffusionDB下载100张AI图像
# https://github.com/poloclub/DiffusionDB

# 4. 或者自己生成一些图像用于测试
```

---

## 数据准备脚本

创建一个辅助脚本来组织数据：

```bash
# scripts/prepare_data.sh
#!/bin/bash

# 源数据路径
SOURCE_DIR="GenImage/GenImage/train"
TARGET_DIR="data"

# 创建目标目录
mkdir -p $TARGET_DIR/train/real $TARGET_DIR/train/fake
mkdir -p $TARGET_DIR/val/real $TARGET_DIR/val/fake

# 复制真实图像
for gen_dir in $SOURCE_DIR/*/; do
    real_dir="${gen_dir}0_real/"
    if [ -d "$real_dir" ]; then
        cp ${real_dir}*.jpg $TARGET_DIR/train/real/ 2>/dev/null
        echo "从 $(basename $gen_dir) 复制真实图像"
    fi
done

# 复制伪造图像（从几个主要生成器）
fake_generators=("SDv14" "ADM" "BigGAN" "Glide")
for gen in "${fake_generators[@]}"; do
    fake_dir="$SOURCE_DIR/$gen/1_fake/"
    if [ -d "$fake_dir" ]; then
        cp ${fake_dir}*.jpg $TARGET_DIR/train/fake/ 2>/dev/null
        echo "从 $gen 复制伪造图像"
    fi
done

echo "数据准备完成！"
echo "训练集真实图像: $(ls $TARGET_DIR/train/real | wc -l)"
echo "训练集伪造图像: $(ls $TARGET_DIR/train/fake | wc -l)"
```

---

## 数据集大小建议

| 用途 | 真实图像 | 伪造图像 | 总大小 |
|------|----------|----------|--------|
| 快速测试 | 500 | 500 | ~500MB |
| 基线训练 | 5,000 | 5,000 | ~5GB |
| 完整训练 | 20,000 | 20,000 | ~20GB |
| 大规模训练 | 50,000+ | 50,000+ | ~50GB+ |

**LAIGCD设计目标：100GB以内完整训练**

---

## 注意事项

1. **图像格式**：支持 .jpg, .jpeg, .png, .webp
2. **目录名称**：必须是 `real` 和 `fake`（区分大小写）
3. **嵌套结构**：fake目录下可以有子目录（按生成器分类）
4. **图像质量**：建议保持原始分辨率，代码会自动resize

---

## 下一步

准备好数据后，可以开始训练：

```bash
cd LAIGCD
bash scripts/train.sh --data_path data --output_dir checkpoints
```
