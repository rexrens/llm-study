"""
PyTorch DataLoader 教程
DataLoader 提供了批量数据加载、数据打乱、多进程加载等功能。
"""

import time
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

# 1. 基本 DataLoader 使用
print("=== 1. 基本 DataLoader 使用 ===")

class SimpleDataset(Dataset):
    """简单数据集"""
    def __init__(self, size: int = 100):
        self.data = torch.randn(size, 10)
        self.labels = torch.randint(0, 2, (size,))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]


dataset = SimpleDataset(size=100)

# 创建 DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=16,      # 批量大小
    shuffle=True,      # 是否打乱
)

print(f"数据集大小: {len(dataset)}")
print(f"批次数量: {len(dataloader)}")

# 遍历 DataLoader
for batch_idx, (data, labels) in enumerate(dataloader):
    if batch_idx >= 2:  # 只打印前两个批次
        break
    print(f"Batch {batch_idx}: data shape={data.shape}, labels={labels}")


# 2. 关键参数说明
print("\n=== 2. DataLoader 关键参数 ===")

# batch_size: 每个批次的样本数
dataloader = DataLoader(dataset, batch_size=32)

# shuffle: 每个 epoch 开始时是否打乱数据
dataloader_shuffle = DataLoader(dataset, shuffle=True)

# num_workers: 使用多少个子进程加载数据 (0 表示主进程加载)
dataloader_workers = DataLoader(dataset, num_workers=0)  # 在实际使用中可以设置为 4

# drop_last: 如果数据集大小不能被 batch_size 整除，
#           是否丢弃最后不完整的批次
dataloader_drop = DataLoader(dataset, batch_size=17, drop_last=True)

print(f"drop_last=False 时批次数量: {len(DataLoader(dataset, batch_size=17, drop_last=False))}")
print(f"drop_last=True 时批次数量: {len(dataloader_drop)}")

# pin_memory: 将数据加载到固定内存 (pinned memory) 中，
#             加速从 CPU 到 GPU 的数据传输
dataloader_pin = DataLoader(dataset, pin_memory=True)


# 3. 完整的训练循环示例
print("\n=== 3. 完整训练循环 ===")

def train_model(num_epochs: int = 3, batch_size: int = 32):
    """模拟训练循环"""
    dataset = SimpleDataset(size=1000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for batch_idx, (data, labels) in enumerate(dataloader):
            # 模拟前向传播和损失计算
            loss = torch.randn(1).item()
            total_loss += loss
            num_batches += 1

            # 每 10 个批次打印一次
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Avg Loss: {avg_loss:.4f}")

        print(f"Epoch {epoch+1} 完成, 总平均损失: {total_loss/num_batches:.4f}\n")

train_model(num_epochs=2, batch_size=32)


# 4. 自定义 Collate Function
print("\n=== 4. 自定义 Collate Function ===")

# 默认情况下，DataLoader 使用默认的 collate_fn
# 它会将样本堆叠成一个 batch

# 当样本具有不同形状时，需要自定义 collate_fn

class VariableSizeDataset(Dataset):
    """变长数据集"""
    def __init__(self, size: int = 10):
        self.sequences = [torch.randn(torch.randint(5, 20, (1,)).item()) for _ in range(size)]
        self.labels = torch.randint(0, 2, (size,))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.labels[idx]


# 方法 1: 填充到固定长度
def pad_collate_fn(batch):
    """将变长序列填充到相同长度"""
    sequences, labels = zip(*batch)

    # 找到最大长度
    max_len = max(seq.size(0) for seq in sequences)

    # 填充
    padded_sequences = torch.zeros(len(sequences), max_len)
    for i, seq in enumerate(sequences):
        padded_sequences[i, :seq.size(0)] = seq

    labels = torch.stack(labels)
    return padded_sequences, labels


var_dataset = VariableSizeDataset(size=10)
var_dataloader = DataLoader(
    var_dataset,
    batch_size=4,
    collate_fn=pad_collate_fn,
)

for i, (data, labels) in enumerate(var_dataloader):
    if i >= 1:
        break
    print(f"填充后数据形状: {data.shape}")
    print(f"标签: {labels}")


# 方法 2: 返回打包的序列 (用于 RNN)
def pack_collate_fn(batch):
    """返回打包的序列（不填充）"""
    from torch.nn.utils.rnn import pack_sequence

    sequences, labels = zip(*batch)
    # 按长度降序排序
    sorted_idx = sorted(range(len(sequences)), key=lambda i: sequences[i].size(0), reverse=True)
    sequences = [sequences[i] for i in sorted_idx]
    labels = torch.tensor([labels[i] for i in sorted_idx])

    packed_sequences = pack_sequence(sequences, enforce_sorted=True)
    return packed_sequences, labels


# 5. 数据增强在 DataLoader 中
print("\n=== 5. 数据增强 ===")

from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation, Normalize, ToTensor

class ImageDataset(Dataset):
    """模拟图像数据集"""
    def __init__(self, size: int = 100, transform=None):
        self.images = torch.rand(size, 3, 32, 32)
        self.labels = torch.randint(0, 10, (size,))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# 训练时的数据增强
train_transform = Compose([
    # ToTensor(),  # 如果是 PIL 图片
    # RandomHorizontalFlip(p=0.5),  # 需要 PIL 图片
    # RandomRotation(degrees=10),
])

train_dataset = ImageDataset(size=100, transform=train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)


# 6. 验证时的 DataLoader (不 shuffle, 不 augment)
print("\n=== 6. 训练/验证 DataLoader ===")

val_dataset = ImageDataset(size=50, transform=None)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,  # 验证时不打乱
)

print(f"训练 DataLoader: shuffle=True")
print(f"验证 DataLoader: shuffle=False")


# 7. 性能优化
print("\n=== 7. 性能优化 ===")

def measure_dataloader_speed(dataloader: DataLoader, name: str, num_batches: int = 10):
    """测量 DataLoader 加载速度"""
    start_time = time.time()

    for i, (data, labels) in enumerate(dataloader):
        if i >= num_batches:
            break
        # 模拟一些计算
        _ = data.sum()

    elapsed = time.time() - start_time
    print(f"{name}: {elapsed:.4f}s for {num_batches} batches")


dataset = SimpleDataset(size=10000)

# 不同配置的比较
dataloader_1 = DataLoader(dataset, batch_size=32, num_workers=0, pin_memory=False)
dataloader_2 = DataLoader(dataset, batch_size=32, num_workers=0, pin_memory=True)
# dataloader_3 = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)  # 实际使用时可开启

measure_dataloader_speed(dataloader_1, "num_workers=0, pin_memory=False")
measure_dataloader_speed(dataloader_2, "num_workers=0, pin_memory=True")


# 8. 处理不平衡数据集
print("\n=== 8. 不平衡数据集 ===")

class ImbalancedDataset(Dataset):
    """不平衡数据集"""
    def __init__(self, size: int = 1000):
        # 类别 0: 90%, 类别 1: 10%
        self.data = torch.randn(size, 10)
        self.labels = torch.cat([
            torch.zeros(int(size * 0.9), dtype=torch.long),
            torch.ones(int(size * 0.1), dtype=torch.long),
        ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]


from torch.utils.data import WeightedRandomSampler

imbalanced_dataset = ImbalancedDataset(size=1000)

# 计算每个样本的权重
class_counts = torch.bincount(imbalanced_dataset.labels)
class_weights = 1.0 / class_counts.float()
sample_weights = class_weights[imbalanced_dataset.labels]

# 创建采样器
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(imbalanced_dataset),
    replacement=True,
)

# 使用采样器创建 DataLoader
dataloader_balanced = DataLoader(
    imbalanced_dataset,
    batch_size=32,
    sampler=sampler,  # 注意: 使用 sampler 时不能设置 shuffle=True
)

print(f"原始数据集分布: {class_counts.tolist()}")

# 验证平衡效果
labels_in_batch = []
for data, labels in dataloader_balanced:
    labels_in_batch.extend(labels.tolist())
    if len(labels_in_batch) >= 100:
        break

balanced_counts = torch.bincount(torch.tensor(labels_in_batch))
print(f"采样后分布: {balanced_counts.tolist()}")


# 9. 数据集缓存和预加载
print("\n=== 9. 数据集预加载 ===")

class PreloadedDataset(Dataset):
    """预加载所有数据到内存"""

    def __init__(self, size: int = 1000):
        print("正在预加载数据...")
        self.data = torch.randn(size, 10)
        self.labels = torch.randint(0, 2, (size,))
        print("预加载完成!")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]


preloaded_dataset = PreloadedDataset(size=1000)


# 10. 实战: 完整的数据加载流程
print("\n=== 10. 实战: 完整流程 ===")

def prepare_dataloaders(
    train_size: int = 1000,
    val_size: int = 200,
    test_size: int = 200,
    batch_size: int = 32,
):
    """准备训练、验证、测试 DataLoader"""

    # 创建数据集
    train_dataset = SimpleDataset(size=train_size)
    val_dataset = SimpleDataset(size=val_size)
    test_dataset = SimpleDataset(size=test_size)

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,      # 训练时打乱
        num_workers=0,      # 实际使用时可设置为 4
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,      # 验证时不打乱
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"训练集: {len(train_dataset)} 样本, {len(train_loader)} 批次")
    print(f"验证集: {len(val_dataset)} 样本, {len(val_loader)} 批次")
    print(f"测试集: {len(test_dataset)} 样本, {len(test_loader)} 批次")

    return train_loader, val_loader, test_loader


train_loader, val_loader, test_loader = prepare_dataloaders()

print("\nDataLoader 教程完成!")
