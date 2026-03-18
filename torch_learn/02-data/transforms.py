"""
PyTorch Transforms 数据变换教程
Transforms 用于数据预处理和增强，是提高模型泛化能力的关键。
"""

import random
from typing import Optional

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, RandomRotation, ToTensor

# 1. 基本变换概念
print("=== 1. 基本变换概念 ===")

# torchvision 提供了丰富的预定义变换
# 适用于图像数据的常见变换

# 创建一些模拟图像数据
dummy_image = torch.rand(3, 32, 32)  # 模拟 RGB 图像

# ToTensor: 将 PIL Image 或 numpy array 转换为 tensor
# 注意: 这里 dummy_image 已经是 tensor，所以不需要转换

print(f"原始图像形状: {dummy_image.shape}")
print(f"数值范围: [{dummy_image.min():.3f}, {dummy_image.max():.3f}]")


# 2. 常用图像变换
print("\n=== 2. 常用图像变换 ===")

# Compose: 组合多个变换
transform = Compose([
    # ToTensor(),  # 将图像转换为 tensor 并归一化到 [0, 1]
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet 归一化
])

# Normalize: 减去均值，除以标准差
# 这有助于训练稳定性和收敛速度
normalized_image = transform(dummy_image)
print(f"归一化后数值范围: [{normalized_image.min():.3f}, {normalized_image.max():.3f}]")


# 3. 数据增强变换
print("\n=== 3. 数据增强变换 ===")

# RandomHorizontalFlip: 随机水平翻转 (p=0.5 表示 50% 概率)
hflip = RandomHorizontalFlip(p=0.5)

# RandomRotation: 随机旋转
rotate = RandomRotation(degrees=10)

# 这些变换会以一定概率应用于图像，用于增加训练数据多样性
# 注意: 实际使用时需要 PIL Image，这里只是演示概念


# 4. 自定义变换
print("\n=== 4. 自定义变换 ===")

class AddNoise:
    """添加高斯噪声"""

    def __init__(self, mean: float = 0.0, std: float = 0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise


class RandomCrop:
    """随机裁剪"""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        _, h, w = tensor.shape
        top = random.randint(0, max(0, h - self.size))
        left = random.randint(0, max(0, w - self.size))
        return tensor[:, top:top+self.size, left:left+self.size]


class ColorJitter:
    """颜色抖动"""

    def __init__(self, brightness: float = 0.0, contrast: float = 0.0):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # 简化版本，实际应该更复杂
        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            tensor = tensor * factor
        if self.contrast > 0:
            mean = tensor.mean()
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            tensor = (tensor - mean) * factor + mean
        return torch.clamp(tensor, 0, 1)


# 使用自定义变换
transform = Compose([
    AddNoise(std=0.05),
])

noisy_image = transform(dummy_image)
print(f"添加噪声后的差异: {(noisy_image - dummy_image).abs().mean():.4f}")


# 5. Lambda 变换
print("\n=== 5. Lambda 变换 ===")

from torchvision.transforms import Lambda

# Lambda: 使用 lambda 函数创建简单变换
to_one_hot = Lambda(
    lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)
)

# 示例: 将类别标签转换为 one-hot 编码
label = 3
one_hot_label = to_one_hot(label)
print(f"标签 {label} 的 one-hot 编码: {one_hot_label}")


# 6. 函数式变换
print("\n=== 6. 函数式变换 ===")

# torchvision.transforms.functional 提供函数式变换
from torchvision.transforms import functional as TF

# 这些变换不是类，而是函数，需要明确指定参数
# 适用于需要精确控制的情况


# 7. 训练和验证的不同变换
print("\n=== 7. 训练 vs 验证变换 ===")

# 训练时使用数据增强
train_transform = Compose([
    # ToTensor(),
    AddNoise(std=0.05),
    # RandomHorizontalFlip(p=0.5),
])

# 验证时不使用增强，只使用基本变换
val_transform = Compose([
    # ToTensor(),
])

print("训练: 使用数据增强")
print("验证: 不使用数据增强")


# 8. 文本数据变换
print("\n=== 8. 文本数据变换 ===")

class TextTransform:
    """文本数据变换"""

    def __init__(self, vocab_size: int = 1000, max_length: int = 128):
        self.vocab_size = vocab_size
        self.max_length = max_length

    def tokenize(self, text: str) -> list[int]:
        """简单的分词 (模拟)"""
        # 实际使用时应该使用专业 tokenizer
        return [hash(word) % self.vocab_size for word in text.split()]

    def pad_or_truncate(self, tokens: list[int]) -> torch.Tensor:
        """填充或截断到固定长度"""
        if len(tokens) >= self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

    def __call__(self, text: str) -> torch.Tensor:
        tokens = self.tokenize(text)
        return self.pad_or_truncate(tokens)


# 使用文本变换
text_transform = TextTransform(vocab_size=1000, max_length=50)
text = "this is a sample text for demonstration"
encoded = text_transform(text)
print(f"编码后形状: {encoded.shape}")
print(f"编码结果 (前10个): {encoded[:10].tolist()}")


# 9. 音频数据变换
print("\n=== 9. 音频数据变换 ===")

class AudioTransform:
    """音频数据变换"""

    def __init__(self, sample_rate: int = 16000, n_mels: int = 80):
        self.sample_rate = sample_rate
        self.n_mels = n_mels

    def waveform_to_melspectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """将波形转换为 Mel 频谱图 (简化版)"""
        # 实际使用时应该使用 torchaudio.transforms
        # 这里只是一个占位符
        return torch.randn(self.n_mels, 100)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.waveform_to_melspectrogram(waveform)


# 10. 实战: 标准的图像预处理流程
print("\n=== 10. 实战: 标准流程 ===")

def get_train_transform(image_size: int = 224):
    """获取训练时的数据变换"""
    return Compose([
        # Resize(image_size),  # 调整大小
        # ToTensor(),  # 转为 tensor
        # RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        # RandomRotation(degrees=15),  # 随机旋转
        # ColorJitter(brightness=0.2, contrast=0.2),  # 颜色抖动
        # Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 归一化
        #           std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform(image_size: int = 224):
    """获取验证时的数据变换"""
    return Compose([
        # Resize(image_size),
        # ToTensor(),
        # Normalize(mean=[0.485, 0.456, 0.406],
        #           std=[0.229, 0.224, 0.225]),
    ])


# 实际使用示例
"""
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# 加载数据集
train_dataset = CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=get_train_transform(),
)

val_dataset = CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=get_val_transform(),
)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
"""

# 11. 数据增强的最佳实践
print("\n=== 11. 最佳实践 ===")

"""
数据增强的最佳实践:

1. 只在训练时使用数据增强，验证和测试时不使用
2. 增强强度要适度，过强的增强可能破坏数据语义
3. 根据任务选择合适的增强方法
   - 图像分类: 几何变换、颜色变换
   - 目标检测: 注意边界框的同步变换
   - 分割: 保持像素级标签的一致性
4. 使用固定随机种子以实现可重复性
5. 在线增强 (实时) 比离线生成更好
"""

# 12. 可视化变换效果
print("\n=== 12. 可视化变换效果 ===")

def visualize_transforms(image: torch.Tensor, transform, num_samples: int = 4):
    """可视化变换效果"""
    print(f"原始图像形状: {image.shape}")
    print(f"生成 {num_samples} 个增强样本:")

    for i in range(num_samples):
        augmented = transform(image.clone())
        print(f"  样本 {i+1}: shape={augmented.shape}, "
              f"min={augmented.min():.3f}, max={augmented.max():.3f}")


# 演示
transform = Compose([
    AddNoise(std=0.1),
])
visualize_transforms(dummy_image, transform)

print("\nTransforms 教程完成!")
