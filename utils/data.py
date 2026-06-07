"""
数据加载和处理模块
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class AIGCDataset(Dataset):
    """
    AI生成内容检测数据集

    Args:
        data_path: 数据根目录
        split: 'train' 或 'val' 或 'test'
        transform: 数据变换
    """
    def __init__(self, data_path, split='train', transform=None):
        super().__init__()

        self.data_path = data_path
        self.split = split
        self.transform = transform

        # 收集所有图像路径和标签
        self.samples = []
        self._collect_samples()

    def _collect_samples(self):
        """收集所有图像路径和标签"""
        split_dir = os.path.join(self.data_path, self.split)

        if not os.path.exists(split_dir):
            raise ValueError(f"数据目录不存在: {split_dir}")

        # 真实图像目录
        real_dir = os.path.join(split_dir, 'real')
        if os.path.exists(real_dir):
            for filename in os.listdir(real_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    self.samples.append({
                        'path': os.path.join(real_dir, filename),
                        'label': 0  # 0 = real
                    })

        # AI生成图像目录
        fake_dir = os.path.join(split_dir, 'fake')
        if os.path.exists(fake_dir):
            # 支持子目录结构（不同生成器）
            for root, dirs, files in os.walk(fake_dir):
                for filename in files:
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        self.samples.append({
                            'path': os.path.join(root, filename),
                            'label': 1  # 1 = fake
                        })

        print(f"{self.split} 数据集: 找到 {len(self.samples)} 张图像")
        if len(self.samples) > 0:
            real_count = sum(1 for s in self.samples if s['label'] == 0)
            fake_count = len(self.samples) - real_count
            print(f"  真实: {real_count}, 伪造: {fake_count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载图像
        image = Image.open(sample['path']).convert('RGB')

        # 应用变换
        if self.transform is not None:
            image = self.transform(image)

        return image, sample['label']


def get_train_transforms(img_size=224):
    """获取训练数据增强"""
    return transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.RandomCrop(img_size),
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


def get_val_transforms(img_size=224):
    """获取验证/测试数据变换"""
    return transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_train_dataloader(data_path, batch_size=32, num_workers=4, img_size=224):
    """获取训练数据加载器"""
    dataset = AIGCDataset(
        data_path=data_path,
        split='train',
        transform=get_train_transforms(img_size)
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    return dataloader


def get_val_dataloader(data_path, batch_size=32, num_workers=4, img_size=224, split='val'):
    """获取验证/测试数据加载器"""
    dataset = AIGCDataset(
        data_path=data_path,
        split=split,
        transform=get_val_transforms(img_size)
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return dataloader


class SimpleDataset(Dataset):
    """
    简单数据集（用于测试）
    生成随机图像用于快速测试模型
    """
    def __init__(self, num_samples=100, img_size=224):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 生成随机图像
        image = torch.randn(3, self.img_size, self.img_size)
        # 随机标签
        label = idx % 2
        return image, label


def test_dataloader():
    """测试数据加载器"""
    print("测试数据加载器...")

    # 测试SimpleDataset
    dataset = SimpleDataset(num_samples=100)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    print(f"数据集大小: {len(dataset)}")

    for images, labels in dataloader:
        print(f"批次形状: images={images.shape}, labels={labels.shape}")
        break

    print("✓ 数据加载器测试通过")


if __name__ == "__main__":
    test_dataloader()
