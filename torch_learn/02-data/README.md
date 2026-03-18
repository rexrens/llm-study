# 02-data: 数据加载和处理

本模块介绍 PyTorch 中的数据加载和处理机制，包括 Dataset、DataLoader 和数据变换。

## 学习目标

- 理解 Dataset 的基本概念和自定义方法
- 掌握 DataLoader 的使用和关键参数
- 学会使用数据增强提高模型泛化能力
- 掌握常见的数据预处理技巧

## 文件说明

### dataset.py - 自定义数据集

Dataset 是 PyTorch 中用于自定义数据加载的抽象类。

```bash
# 运行示例
python 02-data/dataset.py
```

**主要内容包括：**
- 简单的 Dataset 实现
- 从文件加载的 Dataset（图像、CSV）
- 数据集划分（训练/验证/测试）
- 多模态 Dataset
- 内存优化技巧（延迟加载、缓存）

### dataloader.py - 数据加载器

DataLoader 提供了批量数据加载、数据打乱、多进程加载等功能。

```bash
# 运行示例
python 02-data/dataloader.py
```

**主要内容包括：**
- DataLoader 基本使用
- 关键参数详解（batch_size, shuffle, num_workers, pin_memory, drop_last）
- 自定义 collate_fn
- 处理变长序列
- 处理不平衡数据集
- 性能优化技巧

### transforms.py - 数据变换

Transforms 用于数据预处理和增强，是提高模型泛化能力的关键。

```bash
# 运行示例
python 02-data/transforms.py
```

**主要内容包括：**
- 基本变换概念
- 常用图像变换
- 自定义变换
- Lambda 变换
- 文本和音频数据变换
- 训练/验证的不同处理
- 最佳实践

## 核心概念

### Dataset

Dataset 是一个抽象类，需要实现 `__len__` 和 `__getitem__` 方法：

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
```

### DataLoader

DataLoader 负责批量加载数据：

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,      # 批量大小
    shuffle=True,      # 是否打乱
    num_workers=4,      # 多进程加载数量
    pin_memory=True,    # 固定内存，加速 GPU 传输
)
```

### Transforms

使用 Compose 组合多个变换：

```python
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip

transform = Compose([
    RandomHorizontalFlip(p=0.5),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
```

## 常见模式

### 数据集划分

```python
from torch.utils.data import random_split

# 方法 1: random_split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size]
)
```

### 处理变长序列

```python
def pad_collate_fn(batch):
    sequences, labels = zip(*batch)
    max_len = max(len(s) for s in sequences)
    padded = torch.zeros(len(sequences), max_len)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq
    return padded, torch.tensor(labels)

dataloader = DataLoader(dataset, collate_fn=pad_collate_fn)
```

### 训练/验证分别处理

```python
# 训练: 使用数据增强
train_transform = Compose([...])
train_dataset = MyDataset(..., transform=train_transform)

# 验证: 不使用增强
val_transform = Compose([...])
val_dataset = MyDataset(..., transform=val_transform)
```

## 最佳实践

1. **数据集大小**: 小数据集可以预加载到内存，大数据集使用延迟加载
2. **num_workers**: 根据可用 CPU 核心数设置，通常为 4-8
3. **pin_memory**: 训练时启用，验证和推理时可关闭
4. **数据增强**: 只在训练时使用，验证时不使用
5. **随机种子**: 为可重复性设置固定种子
6. **缓存**: 对计算密集型的预处理考虑缓存结果

## 常见误区

1. **忘记 shuffle**: 训练时应该 shuffle，验证时不应该
2. **num_workers 过多**: 过多的 workers 会增加开销
3. **不当的数据增强**: 过强的增强可能破坏数据语义
4. **数据泄漏**: 确保验证集数据不影响训练

## 下一步

完成本模块后，继续学习：
- [03-nn-basics](../03-nn-basics/README.md) - 神经网络基础
- [04-cnn](../04-cnn/README.md) - 卷积神经网络
