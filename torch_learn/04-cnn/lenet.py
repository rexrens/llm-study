"""
LeNet-5 卷积神经网络实现
LeNet-5 是最早的卷积神经网络之一，由 Yann LeCun 于 1998 年提出。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """
    LeNet-5 架构

    原始架构:
        1. Input: 32x32 grayscale image
        2. Conv: 6 kernels of size 5x5, stride 1, padding 0 -> 28x28x6
        3. ReLU activation
        4. AvgPool: 2x2, stride 2 -> 14x14x6
        5. Conv: 16 kernels of size 5x5 -> 10x10x16
        6. ReLU activation
        7. AvgPool: 2x2, stride 2 -> 5x5x16
        8. Conv: 120 kernels of size 5x5 -> 1x1x120
        9. ReLU activation
        10. FC: 120 -> 84
        11. FC: 84 -> 10
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # 第一个卷积块
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 第二个卷积块
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 第三个卷积块 (原始是 Conv 5x5, 简化为 FC)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)

        # 全连接层
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状为 (batch_size, 1, 32, 32)

        Returns:
            输出 logits，形状为 (batch_size, num_classes)
        """
        # Conv1 -> ReLU -> Pool
        x = F.relu(self.conv1(x))   # (batch, 6, 28, 28)
        x = self.pool1(x)            # (batch, 6, 14, 14)

        # Conv2 -> ReLU -> Pool
        x = F.relu(self.conv2(x))   # (batch, 16, 10, 10)
        x = self.pool2(x)            # (batch, 16, 5, 5)

        # Conv3 -> ReLU
        x = F.relu(self.conv3(x))   # (batch, 120, 1, 1)

        # 展平
        x = x.view(x.size(0), -1)    # (batch, 120)

        # 全连接层
        x = F.relu(self.fc1(x))     # (batch, 84)
        x = self.fc2(x)             # (batch, num_classes)

        return x


# 现代化版本（使用 MaxPool 和 BatchNorm）
class LeNet5Modern(nn.Module):
    """
    现代化的 LeNet-5 架构

    改进:
        - 使用 MaxPool 代替 AvgPool
        - 添加 BatchNorm
        - 添加 Dropout 防止过拟合
    """

    def __init__(self, num_classes: int = 10, dropout_prob: float = 0.25):
        super().__init__()

        # 第一个卷积块
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二个卷积块
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第三个卷积块
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(120)

        # 全连接层
        self.fc1 = nn.Linear(120, 84)
        self.bn_fc1 = nn.BatchNorm1d(84)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv1 -> BN -> ReLU -> Pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Conv2 -> BN -> ReLU -> Pool
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Conv3 -> BN -> ReLU
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # 展平
        x = x.view(x.size(0), -1)

        # FC1 -> BN -> ReLU -> Dropout -> FC2
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# 测试
if __name__ == "__main__":
    # 创建模型
    model = LeNet5(num_classes=10)
    modern_model = LeNet5Modern(num_classes=10)

    print("原始 LeNet-5:")
    print(model)
    print(f"\n参数量: {sum(p.numel() for p in model.parameters()):,}")

    print("\n现代化 LeNet-5:")
    print(modern_model)
    print(f"\n参数量: {sum(p.numel() for p in modern_model.parameters()):,}")

    # 测试前向传播
    x = torch.randn(4, 1, 32, 32)  # batch_size=4, 1 channel, 32x32 image
    output = model(x)
    print(f"\n输出形状: {output.shape}")

    # 测试预测
    predictions = torch.argmax(output, dim=1)
    print(f"预测类别: {predictions.tolist()}")
