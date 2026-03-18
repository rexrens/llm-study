"""
ResNet (残差网络) 实现
ResNet 通过残差连接解决了深层网络难以训练的问题。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    基本残差块 (用于 ResNet-18 和 ResNet-34)

    结构:
        x -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+ x) -> ReLU
    """

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 残差连接
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    瓶颈残差块 (用于 ResNet-50, ResNet-101, ResNet-152)

    结构:
        x -> Conv1x1 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Conv1x1 -> BN -> (+ x) -> ReLU
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ):
        super().__init__()

        # 1x1 卷积降维
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 卷积
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 卷积升维
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion,
            kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 残差连接
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet 主干网络

    结构:
        Input -> Conv7x7 -> BN -> ReLU -> MaxPool
        -> Layer1 -> Layer2 -> Layer3 -> Layer4
        -> AvgPool -> FC -> Output
    """

    def __init__(
        self,
        block: type[BasicBlock | Bottleneck],
        layers: list[int],
        num_classes: int = 1000,
    ):
        super().__init__()

        self.in_channels = 64

        # 初始卷积层
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 输出层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: type[BasicBlock | Bottleneck],
        out_channels: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """创建残差层"""
        downsample = None

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride, downsample)
        )
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# 预定义的 ResNet 变体
def resnet18(num_classes: int = 10, in_channels: int = 3) -> ResNet:
    """ResNet-18"""
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

    # 如果输入通道不是 3，修改第一层
    if in_channels != 3:
        model.conv1 = nn.Conv2d(
            in_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )

    return model


def resnet34(num_classes: int = 10, in_channels: int = 3) -> ResNet:
    """ResNet-34"""
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

    if in_channels != 3:
        model.conv1 = nn.Conv2d(
            in_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )

    return model


def resnet50(num_classes: int = 10, in_channels: int = 3) -> ResNet:
    """ResNet-50"""
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

    if in_channels != 3:
        model.conv1 = nn.Conv2d(
            in_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )

    return model


def resnet101(num_classes: int = 10, in_channels: int = 3) -> ResNet:
    """ResNet-101"""
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)

    if in_channels != 3:
        model.conv1 = nn.Conv2d(
            in_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )

    return model


def resnet152(num_classes: int = 10, in_channels: int = 3) -> ResNet:
    """ResNet-152"""
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)

    if in_channels != 3:
        model.conv1 = nn.Conv2d(
            in_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )

    return model


# 简化版 ResNet (用于 MNIST 等小图像)
class SmallResNet(nn.Module):
    """
    简化版 ResNet，适用于小尺寸图像 (如 MNIST 28x28)

    结构:
        Input -> Conv3x3 -> BN -> ReLU
        -> Layer1 (64) -> Layer2 (128) -> Layer3 (256)
        -> AvgPool -> FC -> Output
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(BasicBlock, 64, 2)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(
        self,
        block: type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride, downsample)
        )
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# 测试
if __name__ == "__main__":
    print("=== ResNet 测试 ===\n")

    # ResNet-18
    print("ResNet-18:")
    model = resnet18(num_classes=10)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"输入: {x.shape} -> 输出: {output.shape}\n")

    # Small ResNet (MNIST)
    print("Small ResNet (MNIST):")
    small_model = SmallResNet(num_classes=10, in_channels=1)
    print(f"参数量: {sum(p.numel() for p in small_model.parameters()):,}")

    x = torch.randn(4, 1, 28, 28)
    output = small_model(x)
    print(f"输入: {x.shape} -> 输出: {output.shape}")

    # 预测
    predictions = torch.argmax(output, dim=1)
    print(f"预测类别: {predictions.tolist()}")
