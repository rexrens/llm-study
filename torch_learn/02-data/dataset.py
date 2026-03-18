"""
PyTorch Dataset 自定义教程
Dataset 是 PyTorch 中用于自定义数据加载的抽象类。
"""

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset


# 1. 简单的 Dataset 示例
print("=== 1. 简单 Dataset 示例 ===")

class SimpleDataset(Dataset):
    """最简单的 Dataset 实现"""

    def __init__(self, data_size: int = 100):
        # 生成一些随机数据
        self.data = torch.randn(data_size, 10)
        self.labels = torch.randint(0, 2, (data_size,))

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回第 idx 个样本"""
        return self.data[idx], self.labels[idx]


# 使用示例
dataset = SimpleDataset(data_size=100)
print(f"数据集大小: {len(dataset)}")

# 获取单个样本
sample_data, sample_label = dataset[0]
print(f"第一个样本形状: {sample_data.shape}, 标签: {sample_label}")

# 获取多个样本
batch_data, batch_labels = dataset[10:15]
print(f"第10-14个样本形状: {batch_data.shape}")


# 2. 从文件加载的 Dataset
print("\n=== 2. 从文件加载 Dataset ===")

class ImageFileDataset(Dataset):
    """从图像文件夹加载图片的 Dataset"""

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            root_dir: 数据根目录
            transform: 可选的数据变换
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        # 收集所有图片文件
        self.image_files = []
        self.labels = []

        # 假设目录结构为: root/class_0/img1.jpg, root/class_1/img2.jpg, ...
        if self.root_dir.exists():
            class_dirs = sorted(self.root_dir.iterdir())
            for class_idx, class_dir in enumerate(class_dirs):
                if class_dir.is_dir():
                    for img_file in class_dir.glob("*.jpg"):
                        self.image_files.append(str(img_file))
                        self.labels.append(class_idx)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_files[idx]
        label = self.labels[idx]

        # 实际使用时需要 PIL 或 opencv 读取图片
        # 这里使用随机张量模拟
        image = torch.rand(3, 32, 32)  # 模拟 RGB 图片

        if self.transform:
            image = self.transform(image)

        return image, label


# 3. 从 CSV 加载的 Dataset
print("\n=== 3. 从 CSV 加载 Dataset ===")

import pandas as pd

class CSVDataset(Dataset):
    """从 CSV 文件加载数据"""

    def __init__(
        self,
        csv_path: str,
        feature_cols: list[str],
        label_col: str,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            csv_path: CSV 文件路径
            feature_cols: 特征列名列表
            label_col: 标签列名
            transform: 可选的数据变换
        """
        # 创建虚拟 CSV 文件
        if not os.path.exists(csv_path):
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            df = pd.DataFrame({
                "feature1": range(100),
                "feature2": range(100, 200),
                "label": [i % 2 for i in range(100)],
            })
            df.to_csv(csv_path, index=False)

        self.df = pd.read_csv(csv_path)
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 获取特征
        features = self.df.loc[idx, self.feature_cols].values
        features = torch.tensor(features, dtype=torch.float32)

        # 获取标签
        label = self.df.loc[idx, self.label_col]
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            features = self.transform(features)

        return features, label


# 使用示例
csv_dataset = CSVDataset(
    "data/sample.csv",
    feature_cols=["feature1", "feature2"],
    label_col="label",
)
print(f"CSV 数据集大小: {len(csv_dataset)}")


# 4. 数据集划分
print("\n=== 4. 数据集划分 ===")

from torch.utils.data import random_split, Subset

# 方法 1: random_split
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.2 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42),
)

print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# 方法 2: 使用索引
indices = list(range(len(dataset)))
np.random.seed(42)
np.random.shuffle(indices)

train_idx = indices[:int(0.7*len(indices))]
val_idx = indices[int(0.7*len(indices)):int(0.9*len(indices))]
test_idx = indices[int(0.9*len(indices)):]

train_subset = Subset(dataset, train_idx)
val_subset = Subset(dataset, val_idx)
test_subset = Subset(dataset, test_idx)

print(f"\n使用 Subset 划分:")
print(f"训练集大小: {len(train_subset)}")
print(f"验证集大小: {len(val_subset)}")
print(f"测试集大小: {len(test_subset)}")


# 5. 支持多模态的 Dataset
print("\n=== 5. 多模态 Dataset ===")

class MultiModalDataset(Dataset):
    """支持图像和文本的多模态 Dataset"""

    def __init__(self, data_size: int = 100):
        self.data_size = data_size

    def __len__(self) -> int:
        return self.data_size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # 模拟图像数据
        image = torch.rand(3, 32, 32)

        # 模拟文本数据 (tokenized)
        text_tokens = torch.randint(0, 1000, (20,))

        # 标签
        label = torch.tensor(idx % 2)

        return {
            "image": image,
            "text": text_tokens,
            "label": label,
        }


multi_modal_dataset = MultiModalDataset()
sample = multi_modal_dataset[0]
print(f"多模态样本 keys: {sample.keys()}")
print(f"图像形状: {sample['image'].shape}")
print(f"文本形状: {sample['text'].shape}")
print(f"标签: {sample['label']}")


# 6. 预处理和数据增强
print("\n=== 6. 数据预处理 ===")

class PreprocessedDataset(Dataset):
    """带预处理的 Dataset"""

    def __init__(self, data_size: int = 100, normalize: bool = True):
        self.data = torch.randn(data_size, 10)
        self.labels = torch.randint(0, 2, (data_size,))
        self.normalize = normalize

        # 如果需要归一化，计算统计量
        if self.normalize:
            self.mean = self.data.mean(dim=0)
            self.std = self.data.std(dim=0)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.data[idx]
        label = self.labels[idx]

        if self.normalize:
            data = (data - self.mean) / (self.std + 1e-8)

        return data, label


preprocessed_dataset = PreprocessedDataset(normalize=True)
sample_data, sample_label = preprocessed_dataset[0]
print(f"归一化后数据: mean={sample_data.mean():.4f}, std={sample_data.std():.4f}")


# 7. 内存优化的 Dataset (适用于大数据集)
print("\n=== 7. 内存优化 Dataset ===")

class LazyLoadDataset(Dataset):
    """延迟加载的 Dataset，适用于大型数据集"""

    def __init__(self, file_paths: list[str]):
        self.file_paths = file_paths

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # 只在需要时加载文件
        file_path = self.file_paths[idx]

        # 模拟文件加载
        # 实际使用时: data = torch.load(file_path)
        data = torch.randn(10)

        return data


# 创建虚拟文件路径
file_paths = [f"data/file_{i}.pt" for i in range(10)]
lazy_dataset = LazyLoadDataset(file_paths)
print(f"延迟加载数据集大小: {len(lazy_dataset)}")


# 8. 数据缓存
print("\n=== 8. 数据缓存 ===")

class CachedDataset(Dataset):
    """带缓存的 Dataset"""

    def __init__(self, dataset: Dataset, cache_size: int = 100):
        self.dataset = dataset
        self.cache = {}
        self.cache_size = cache_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        if idx not in self.cache:
            self.cache[idx] = self.dataset[idx]

            # 如果缓存超过大小，删除最旧的
            if len(self.cache) > self.cache_size:
                oldest_idx = next(iter(self.cache))
                del self.cache[oldest_idx]

        return self.cache[idx]


cached_dataset = CachedDataset(dataset)
print(f"带缓存的数据集")


# 9. 实战示例: 模拟图像分类 Dataset
print("\n=== 9. 实战: 图像分类 Dataset ===")

class ImageClassificationDataset(Dataset):
    """图像分类 Dataset 示例"""

    def __init__(
        self,
        data_size: int = 1000,
        num_classes: int = 10,
        transform: Optional[Callable] = None,
    ):
        self.data_size = data_size
        self.num_classes = num_classes
        self.transform = transform

        # 模拟图像数据 (CIFAR-10 风格: 3x32x32)
        self.images = torch.rand(data_size, 3, 32, 32)
        self.labels = torch.randint(0, num_classes, (data_size,))

        # 类别名称
        self.class_names = [f"class_{i}" for i in range(num_classes)]

    def __len__(self) -> int:
        return self.data_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx].clone()
        label = self.labels[idx]

        # 数据增强
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_name(self, label: int) -> str:
        """获取类别名称"""
        return self.class_names[label]


# 创建数据集
image_dataset = ImageClassificationDataset(
    data_size=1000,
    num_classes=10,
)

sample_image, sample_label = image_dataset[0]
print(f"图像形状: {sample_image.shape}")
print(f"标签: {sample_label} ({image_dataset.get_class_name(sample_label.item())})")

print("\nDataset 教程完成!")
