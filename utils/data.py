#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LAIGCD 数据加载模块
支持多种deepfake人脸检测数据集，无需移动或修改原始数据
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal
import random

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from PIL import Image


# ============= 数据集配置 =============

# 140k数据集结构: real_vs_fake/real-vs-fake/{split}/{label}/
DATASET_140K_STRUCTURE = {
    'base': '140k-real-and-fake-faces',
    'csv_file': '140k-real-and-fake-faces',
    'inner_path': 'real_vs_fake/real-vs-fake',
    'splits': ['train', 'valid', 'test'],
    'labels': ['real', 'fake']
}

# 130k数据集结构: images/{label}/ 或 images/{label}/{generator}/
DATASET_130K_STRUCTURE = {
    'base': '130k-real-vs-fake-face',
    'inner_path': 'images',
    'labels': ['real', 'fake'],
    'fake_generators': ['FLUX_DEV', 'FLUX_PRO', 'SDXL']  # fake子类
}


class AIGCSample:
    """单个样本的信息"""
    def __init__(self, image_path: str, label: int, dataset_source: str,
                 generator: Optional[str] = None, split: str = 'train'):
        self.image_path = image_path
        self.label = label  # 0=real, 1=fake
        self.dataset_source = dataset_source  # '140k' or '130k'
        self.generator = generator  # 对于fake，记录生成器类型
        self.split = split

    def __repr__(self):
        return f"AIGCSample(path={Path(self.image_path).name}, label={'fake' if self.label else 'real'}, source={self.dataset_source}, gen={self.generator})"


class AIGCDataset(Dataset):
    """
    多数据集融合的Deepfake检测数据集

    特性:
    - 自动识别不同数据集格式
    - 保持数据来源信息用于分别评估
    - 记录生成器类型用于细粒度分析
    - 不修改原始数据集结构

    数据集结构:
        data_path/
        ├── 140k-real-and-fake-faces/
        │   ├── real_vs_fake/real-vs-fake/
        │   │   ├── train/real/, train/fake/
        │   │   ├── valid/real/, valid/fake/
        │   │   └── test/real/, test/fake/
        │   └── *.csv (可选)
        └── 130k-real-vs-fake-face/
            └── images/
                ├── real/
                └── fake/
                    ├── FLUX_DEV/
                    ├── FLUX_PRO/
                    └── SDXL/
    """

    def __init__(
        self,
        data_path: str,
        split: Literal['train', 'valid', 'test'] = 'train',
        transform: Optional[transforms.Compose] = None,
        datasets: Optional[List[str]] = None,
        val_split: float = 0.2,
        test_split: float = 0.1,
        seed: int = 42,
        max_samples: Optional[int] = None,
        subset_mode: Literal['balanced', 'random'] = 'balanced'
    ):
        """
        Args:
            data_path: 数据集根目录
            split: 'train', 'valid', 'test'
            transform: 图像变换
            datasets: 使用哪些数据集 ['140k', '130k']，None表示全部
            val_split: 对于130k数据集的验证集比例
            test_split: 对于130k数据集的测试集比例
            seed: 随机种子
            max_samples: 小规模测试时限制样本数，None表示使用全部数据
                         建议值: train=1000, val=200, test=200
            subset_mode: 'balanced'-保持real/fake平衡采样
                        'random'-随机采样
        """
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.datasets = datasets or ['140k', '130k']
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.max_samples = max_samples
        self.subset_mode = subset_mode

        # 加载所有样本
        self.samples = self._load_samples()

        # 应用小规模采样
        if max_samples is not None and len(self.samples) > max_samples:
            self.samples = self._subset_samples(max_samples, subset_mode)
            print(f"🔬 小规模测试模式: 采样 {len(self.samples)} 个样本 (模式: {subset_mode})")

        print(f"加载完成: {split}集共 {len(self.samples)} 个样本")

        # 统计信息
        self._print_stats()

    def _load_samples(self) -> List[AIGCSample]:
        """加载所有样本"""
        samples = []

        # 加载140k数据集
        if '140k' in self.datasets:
            samples.extend(self._load_140k())

        # 加载130k数据集
        if '130k' in self.datasets:
            samples.extend(self._load_130k())

        return samples

    def _load_140k(self) -> List[AIGCSample]:
        """加载140k数据集（已有train/valid/test划分）"""
        samples = []
        base_path = self.data_path / DATASET_140K_STRUCTURE['base']
        inner = DATASET_140K_STRUCTURE['inner_path']

        if not base_path.exists():
            print(f"⚠️  140k数据集不存在: {base_path}")
            return samples

        print(f"加载140k数据集 from {base_path}")

        for label_name, label_idx in [('real', 0), ('fake', 1)]:
            label_path = base_path / inner / self.split / label_name
            if not label_path.exists():
                print(f"  ⚠️  {label_path} 不存在")
                continue

            # 递归查找所有图片
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for img_path in label_path.glob(ext):
                    samples.append(AIGCSample(
                        image_path=str(img_path),
                        label=label_idx,
                        dataset_source='140k',
                        generator=None,
                        split=self.split
                    ))

        print(f"  ✓ 140k {self.split}: {len(samples)} 张")
        return samples

    def _load_130k(self) -> List[AIGCSample]:
        """加载130k数据集（需要自动划分train/valid/test）"""
        samples = []
        base_path = self.data_path / DATASET_130K_STRUCTURE['base']
        inner = DATASET_130K_STRUCTURE['inner_path']

        if not base_path.exists():
            print(f"⚠️  130k数据集不存在: {base_path}")
            return samples

        print(f"加载130k数据集 from {base_path}")

        # 加载real图片
        real_path = base_path / inner / 'real'
        if real_path.exists():
            real_images = self._get_all_images(real_path)
            real_samples = self._split_dataset(real_images, label=0, generator=None)
            samples.extend([s for s in real_samples if s.split == self.split])
            print(f"  ✓ 130k real: {len([s for s in real_samples if s.split == self.split])} 张")

        # 加载fake图片（带生成器类型）
        fake_path = base_path / inner / 'fake'
        if fake_path.exists():
            for generator in DATASET_130K_STRUCTURE['fake_generators']:
                gen_path = fake_path / generator
                if gen_path.exists():
                    fake_images = self._get_all_images(gen_path)
                    fake_samples = self._split_dataset(fake_images, label=1, generator=generator)
                    samples.extend([s for s in fake_samples if s.split == self.split])
                    print(f"  ✓ 130k fake/{generator}: {len([s for s in fake_samples if s.split == self.split])} 张")

        return samples

    def _get_all_images(self, root_dir: Path) -> List[Path]:
        """递归获取目录下所有图片"""
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            images.extend(root_dir.rglob(ext))
        return images

    def _split_dataset(
        self,
        image_paths: List[Path],
        label: int,
        generator: Optional[str]
    ) -> List[AIGCSample]:
        """将数据集划分为train/valid/test"""
        random.seed(self.seed)
        random.shuffle(image_paths)

        n = len(image_paths)
        n_test = int(n * self.test_split)
        n_val = int(n * self.val_split)
        n_train = n - n_val - n_test

        splits = (['train'] * n_train) + (['valid'] * n_val) + (['test'] * n_test)

        samples = []
        for img_path, split in zip(image_paths, splits):
            samples.append(AIGCSample(
                image_path=str(img_path),
                label=label,
                dataset_source='130k',
                generator=generator,
                split=split
            ))

        return samples

    def _subset_samples(self, max_samples: int, mode: str) -> List[AIGCSample]:
        """
        对已加载的样本进行子集采样

        Args:
            max_samples: 目标样本数
            mode: 'balanced'-从real/fake各采样一半；'random'-完全随机

        Returns:
            采样后的样本列表
        """
        random.seed(self.seed)

        if mode == 'balanced':
            # 平衡采样：从real和fake各取一半
            real_samples = [s for s in self.samples if s.label == 0]
            fake_samples = [s for s in self.samples if s.label == 1]

            n_per_class = max_samples // 2

            # 如果某一类不足，就全部取，另一类补充
            if len(real_samples) < n_per_class:
                selected_real = real_samples
                selected_fake = random.sample(fake_samples, max_samples - len(real_samples))
            elif len(fake_samples) < n_per_class:
                selected_fake = fake_samples
                selected_real = random.sample(real_samples, max_samples - len(fake_samples))
            else:
                selected_real = random.sample(real_samples, n_per_class)
                selected_fake = random.sample(fake_samples, max_samples - n_per_class)

            return selected_real + selected_fake

        else:  # 'random'
            # 完全随机采样
            return random.sample(self.samples, max_samples)

    def _print_stats(self):
        """打印数据集统计信息"""
        print("\n" + "="*60)
        print(f"数据集统计 - {self.split}")
        print("="*60)

        # 按数据源统计
        for source in ['140k', '130k']:
            source_samples = [s for s in self.samples if s.dataset_source == source]
            real_count = sum(1 for s in source_samples if s.label == 0)
            fake_count = sum(1 for s in source_samples if s.label == 1)

            if source_samples:
                print(f"\n{source}:")
                print(f"  总数: {len(source_samples)}")
                print(f"  真实: {real_count}, 伪造: {fake_count}")

                # 对于130k，显示生成器分布
                if source == '130k':
                    fake_by_gen = {}
                    for s in source_samples:
                        if s.label == 1 and s.generator:
                            fake_by_gen[s.generator] = fake_by_gen.get(s.generator, 0) + 1
                    if fake_by_gen:
                        print(f"  伪造分布: {fake_by_gen}")

        print("\n" + "="*60 + "\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载图像
        image = Image.open(sample.image_path).convert('RGB')

        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 返回数据和元信息
        return {
            'image': image,
            'label': torch.tensor(sample.label, dtype=torch.long),
            'dataset_source': sample.dataset_source,
            'generator': sample.generator or 'unknown',
            'image_path': sample.image_path
        }


class SimpleDataset(Dataset):
    """简化的数据集，用于快速测试"""

    def __init__(self, data_path: str, split: str = 'train',
                 transform: Optional[transforms.Compose] = None):
        """
        支持简单目录结构:
        data_path/{split}/real/ 和 data_path/{split}/fake/
        """
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform

        self.samples = []
        for label_name, label_idx in [('real', 0), ('fake', 1)]:
            label_path = self.data_path / split / label_name
            if label_path.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    self.samples.extend([
                        (str(p), label_idx) for p in label_path.glob(ext)
                    ])

        print(f"SimpleDataset ({split}): {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# ============= 数据变换 =============

def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """训练集数据增强"""
    return transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    """验证/测试集变换"""
    return transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# ============= 数据加载器 =============

def get_train_dataloader(
    data_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    img_size: int = 224,
    datasets: Optional[List[str]] = None,
    seed: int = 42,
    max_samples: Optional[int] = None,
    subset_mode: Literal['balanced', 'random'] = 'balanced'
) -> DataLoader:
    """
    获取训练数据加载器

    Args:
        data_path: 数据集根目录
        batch_size: 批次大小
        num_workers: 数据加载线程数
        img_size: 图像大小
        datasets: 使用哪些数据集，None表示全部
        seed: 随机种子
        max_samples: 小规模测试时限制样本数，None表示全部
                    建议值: 1000 (快速测试), 10000 (中等测试)
        subset_mode: 'balanced'-保持平衡, 'random'-随机

    小规模测试示例:
        loader = get_train_dataloader(data_path, max_samples=1000)
    """
    dataset = AIGCDataset(
        data_path=data_path,
        split='train',
        transform=get_train_transforms(img_size),
        datasets=datasets,
        seed=seed,
        max_samples=max_samples,
        subset_mode=subset_mode
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )


def get_val_dataloader(
    data_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    img_size: int = 224,
    datasets: Optional[List[str]] = None,
    seed: int = 42,
    max_samples: Optional[int] = None,
    subset_mode: Literal['balanced', 'random'] = 'balanced'
) -> DataLoader:
    """
    获取验证数据加载器

    Args:
        max_samples: 小规模测试时限制样本数，建议值: 200
        subset_mode: 'balanced'或'random'
    """
    dataset = AIGCDataset(
        data_path=data_path,
        split='valid',
        transform=get_val_transforms(img_size),
        datasets=datasets,
        seed=seed,
        max_samples=max_samples,
        subset_mode=subset_mode
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )


def get_test_dataloader(
    data_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    img_size: int = 224,
    datasets: Optional[List[str]] = None,
    seed: int = 42,
    max_samples: Optional[int] = None,
    subset_mode: Literal['balanced', 'random'] = 'balanced'
) -> DataLoader:
    """
    获取测试数据加载器

    Args:
        max_samples: 小规模测试时限制样本数，建议值: 200
        subset_mode: 'balanced'或'random'
    """
    dataset = AIGCDataset(
        data_path=data_path,
        split='test',
        transform=get_val_transforms(img_size),
        datasets=datasets,
        seed=seed,
        max_samples=max_samples,
        subset_mode=subset_mode
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )


# ============= 分数据集评估加载器 =============

def get_subset_dataloader(
    data_path: str,
    split: Literal['valid', 'test'] = 'test',
    dataset_source: Literal['140k', '130k'] = '140k',
    batch_size: int = 16,
    num_workers: int = 4,
    img_size: int = 224,
    seed: int = 42
) -> DataLoader:
    """获取特定数据集的加载器（用于分别评估）"""
    dataset = AIGCDataset(
        data_path=data_path,
        split=split,
        transform=get_val_transforms(img_size),
        datasets=[dataset_source],  # 只加载指定数据集
        seed=seed
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )


def get_generator_dataloader(
    data_path: str,
    generator: str,
    split: Literal['valid', 'test'] = 'test',
    batch_size: int = 16,
    num_workers: int = 4,
    img_size: int = 224,
    seed: int = 42
) -> DataLoader:
    """
    获取特定生成器的加载器（用于130k数据集的细粒度评估）

    Args:
        generator: 'FLUX_DEV', 'FLUX_PRO', 'SDXL'
    """
    dataset = AIGCDataset(
        data_path=data_path,
        split=split,
        transform=get_val_transforms(img_size),
        datasets=['130k'],  # 130k才有生成器信息
        seed=seed
    )

    # 过滤出特定生成器的样本
    filtered_samples = [s for s in dataset.samples if s.generator == generator]
    dataset.samples = filtered_samples

    print(f"Generator {generator} ({split}): {len(filtered_samples)} samples")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )


# ============= 测试代码 =============

if __name__ == '__main__':
    """测试数据加载器"""
    data_path = '/home/seborid/deepfake/LAIGCD/data'

    print("="*60)
    print("测试 AIGCDataset")
    print("="*60 + "\n")

    # 测试训练集
    print("加载训练集...")
    train_dataset = AIGCDataset(
        data_path=data_path,
        split='train',
        datasets=['140k', '130k'],
        transform=get_train_transforms()
    )

    # 测试验证集
    print("\n加载验证集...")
    val_dataset = AIGCDataset(
        data_path=data_path,
        split='valid',
        datasets=['140k', '130k'],
        transform=get_val_transforms()
    )

    # 测试测试集
    print("\n加载测试集...")
    test_dataset = AIGCDataset(
        data_path=data_path,
        split='test',
        datasets=['140k', '130k'],
        transform=get_val_transforms()
    )

    # 测试单个样本
    print("\n测试样本加载...")
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"图像shape: {sample['image'].shape}")
        print(f"标签: {sample['label']}")
        print(f"来源: {sample['dataset_source']}")
        print(f"生成器: {sample['generator']}")
        print(f"路径: {sample['image_path']}")

    print("\n✅ 数据加载器测试完成!")
