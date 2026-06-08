"""
LAIGCD 数据集下载脚本
下载两个deepfake人脸检测数据集到 data/ 目录
"""

import kagglehub
import shutil
import os
from pathlib import Path


def download_and_move(dataset_name, target_dir):
    """
    下载Kaggle数据集并移动到目标目录

    Args:
        dataset_name: Kaggle数据集名称 (格式: "user/dataset-name")
        target_dir: 目标保存路径
    """
    print(f"\n{'='*60}")
    print(f"开始下载数据集: {dataset_name}")
    print(f"{'='*60}")

    try:
        # 1. 下载数据集到缓存
        print("正在从Kaggle下载...")
        cache_path = kagglehub.dataset_download(dataset_name)
        print(f"缓存路径: {cache_path}")

        # 2. 创建目标文件夹
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)

        # 3. 提取数据集名称作为子文件夹名
        dataset_folder = dataset_name.split('/')[-1]
        final_path = target_path / dataset_folder

        # 4. 如果目标路径已存在，先删除
        if final_path.exists():
            print(f"目标路径已存在，删除旧文件: {final_path}")
            shutil.rmtree(final_path)

        # 5. 移动文件从缓存到目标路径
        print(f"移动文件到: {final_path}")
        shutil.move(cache_path, final_path)

        print(f"✅ 数据集保存成功: {final_path}")
        print(f"大小: {sum(f.stat().st_size for f in final_path.rglob('*') if f.is_file()) / (1024**3):.2f} GB")

        return final_path

    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None


def main():
    # 定义目标路径
    TARGET_BASE_DIR = "/home/seborid/deepfake/LAIGCD/data"

    # 要下载的数据集列表
    datasets = [
        "xhlulu/140k-real-and-fake-faces",     # 140k real and fake faces
        "shreyanshpatel1/130k-real-vs-fake-face"  # 130k real vs fake face
    ]

    print("="*60)
    print("LAIGCD 数据集下载脚本")
    print("="*60)
    print(f"目标目录: {TARGET_BASE_DIR}")
    print(f"将下载 {len(datasets)} 个数据集")

    # 逐个下载
    downloaded_paths = []
    for dataset in datasets:
        result = download_and_move(dataset, TARGET_BASE_DIR)
        if result:
            downloaded_paths.append(result)

    # 汇总
    print("\n" + "="*60)
    print("下载完成!")
    print("="*60)
    print(f"成功下载 {len(downloaded_paths)}/{len(datasets)} 个数据集")

    for path in downloaded_paths:
        print(f"  - {path}")

    # 显示目录结构
    print("\n数据目录结构:")
    base_path = Path(TARGET_BASE_DIR)
    if base_path.exists():
        for item in base_path.iterdir():
            if item.is_dir():
                print(f"  📁 {item.name}/")
                for sub in item.iterdir():
                    if sub.is_dir():
                        print(f"    📁 {sub.name}/")


if __name__ == "__main__":
    main()
