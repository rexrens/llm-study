"""
语义分割基础教程
语义分割是对图像中每个像素进行分类的任务。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    卷积块: Conv3x3 -> BN -> ReLU
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class EncoderBlock(nn.Module):
    """
    编码器块: ConvBlock x2 -> MaxPool

    用于下采样和特征提取。
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入特征图

        Returns:
            (下采样后的特征, 跳跃连接的特征)
        """
        skip = self.conv2(self.conv1(x))
        down = self.pool(skip)
        return down, skip


class DecoderBlock(nn.Module):
    """
    解码器块: Upsample -> ConvBlock x2

    用于上采样和恢复空间分辨率。
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 来自上一层的特征
            skip: 来自编码器的跳跃连接

        Returns:
            上采样并融合后的特征
        """
        x = self.up(x)

        # 处理可能的尺寸不匹配
        if x.shape != skip.shape:
            diff_h = skip.shape[2] - x.shape[2]
            diff_w = skip.shape[3] - x.shape[3]
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                          diff_h // 2, diff_h - diff_h // 2])

        # 拼接跳跃连接
        x = torch.cat([x, skip], dim=1)

        return self.conv2(self.conv1(x))


class FCN(nn.Module):
    """
    全卷积网络 (Fully Convolutional Network)

    FCN 是最早的语义分割网络之一，将分类网络转换为分割网络。

    结构:
        Backbone -> 1x1 Conv -> Upsample
    """

    def __init__(
        self,
        num_classes: int = 21,
        backbone: Optional[nn.Module] = None,
        in_channels: int = 3,
    ):
        super().__init__()

        if backbone is None:
            # 简化的主干网络
            self.backbone = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),

                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),

                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )
            final_channels = 512
        else:
            self.backbone = backbone
            final_channels = backbone[-1].out_channels  # 假设最后一层有 out_channels

        # 1x1 卷积转换为类别预测
        self.classifier = nn.Conv2d(final_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 (batch, 3, H, W)

        Returns:
            分割图 (batch, num_classes, H/4, W/4) - 需要上采样到原始尺寸
        """
        features = self.backbone(x)
        logits = self.classifier(features)

        # 上采样到原始尺寸 (简化: 4x 上采样)
        logits = F.interpolate(
            logits,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        return logits


class UNet(nn.Module):
    """
    U-Net: 用于生物医学图像分割的经典架构

    U-Net 使用编码器-解码器结构，并通过跳跃连接保留空间信息。

    结构:
        Encoder (下采样) -> Bottleneck -> Decoder (上采样)
                    |____________________|
                         跳跃连接
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        base_channels: int = 64,
    ):
        super().__init__()

        # 编码器
        self.encoder1 = EncoderBlock(in_channels, base_channels)
        self.encoder2 = EncoderBlock(base_channels, base_channels * 2)
        self.encoder3 = EncoderBlock(base_channels * 2, base_channels * 4)
        self.encoder4 = EncoderBlock(base_channels * 4, base_channels * 8)

        # 瓶颈层
        self.bottleneck = nn.Sequential(
            ConvBlock(base_channels * 8, base_channels * 16),
            ConvBlock(base_channels * 16, base_channels * 16),
        )

        # 解码器
        self.decoder4 = DecoderBlock(base_channels * 16, base_channels * 8)
        self.decoder3 = DecoderBlock(base_channels * 8, base_channels * 4)
        self.decoder2 = DecoderBlock(base_channels * 4, base_channels * 2)
        self.decoder1 = DecoderBlock(base_channels * 2, base_channels)

        # 最终卷积
        self.final_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 (batch, 3, H, W)

        Returns:
            分割图 (batch, num_classes, H, W)
        """
        # 编码器路径
        x1, skip1 = self.encoder1(x)   # /2
        x2, skip2 = self.encoder2(x1)  # /4
        x3, skip3 = self.encoder3(x2)  # /8
        x4, skip4 = self.encoder4(x3)  # /16

        # 瓶颈层
        bottleneck = self.bottleneck(x4)  # /16

        # 解码器路径
        x = self.decoder4(bottleneck, skip4)  # /8
        x = self.decoder3(x, skip3)           # /4
        x = self.decoder2(x, skip2)           # /2
        x = self.decoder1(x, skip1)           # /1

        # 最终输出
        logits = self.final_conv(x)

        return logits


class DeepLabV3(nn.Module):
    """
    DeepLabV3 概念性实现

    DeepLabV3 使用空洞卷积 (Dilated Convolution) 来扩大感受野。
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 21,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # 简化的主干网络
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # ASPP (Atrous Spatial Pyramid Pooling)
        # 使用不同空洞率的卷积
        self.aspp = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, hidden_dim, kernel_size=1, padding=0),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, hidden_dim, kernel_size=3, padding=6, dilation=6),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, hidden_dim, kernel_size=3, padding=12, dilation=12),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, hidden_dim, kernel_size=3, padding=18, dilation=18),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ),
        ])

        # 图像池化分支
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 5, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        # 最终分类层
        self.classifier = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 (batch, 3, H, W)

        Returns:
            分割图 (batch, num_classes, H, W)
        """
        input_size = x.shape[2:]

        # 提取特征
        features = self.backbone(x)

        # ASPP 分支
        aspp_features = []
        for aspp_layer in self.aspp:
            aspp_features.append(aspp_layer(features))

        # 图像池化分支
        pooled = self.image_pool(features)
        pooled = F.interpolate(pooled, size=features.shape[2:], mode="bilinear", align_corners=False)
        aspp_features.append(pooled)

        # 拼接所有分支
        x = torch.cat(aspp_features, dim=1)

        # 融合和分类
        x = self.fusion(x)
        logits = self.classifier(x)

        # 上采样到原始尺寸
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

        return logits


def calculate_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> list[float]:
    """
    计算每个类别的 IoU

    Args:
        pred: 预测分割图 (H, W)
        target: 真实分割图 (H, W)
        num_classes: 类别数

    Returns:
        每个类别的 IoU
    """
    ious = []

    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)

        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()

        iou = intersection / (union + 1e-8)
        ious.append(iou.item())

    return ious


# 演示
if __name__ == "__main__":
    print("=== 语义分割基础教程 ===\n")

    # 1. FCN
    print("1. FCN (全卷积网络):")
    fcn = FCN(num_classes=5, in_channels=3)
    x = torch.randn(2, 3, 224, 224)
    output = fcn(x)

    print(f"   输入: {x.shape}")
    print(f"   输出: {output.shape}")
    print(f"   参数量: {sum(p.numel() for p in fcn.parameters()):,}\n")

    # 2. U-Net
    print("2. U-Net:")
    unet = UNet(in_channels=3, num_classes=5, base_channels=64)
    x = torch.randn(1, 3, 256, 256)
    output = unet(x)

    print(f"   输入: {x.shape}")
    print(f"   输出: {output.shape}")
    print(f"   参数量: {sum(p.numel() for p in unet.parameters()):,}\n")

    # 3. DeepLabV3
    print("3. DeepLabV3:")
    deeplab = DeepLabV3(in_channels=3, num_classes=5)
    x = torch.randn(1, 3, 256, 256)
    output = deeplab(x)

    print(f"   输入: {x.shape}")
    print(f"   输出: {output.shape}")
    print(f"   参数量: {sum(p.numel() for p in deeplab.parameters()):,}\n")

    # 4. IoU 计算
    print("4. IoU 计算:")
    pred = torch.tensor([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 2, 2],
        [2, 2, 2, 2],
    ])

    target = torch.tensor([
        [0, 0, 0, 1],
        [0, 1, 1, 1],
        [2, 2, 2, 2],
        [2, 2, 2, 2],
    ])

    ious = calculate_iou(pred, target, num_classes=3)
    print(f"   预测:\n{pred}")
    print(f"   目标:\n{target}")
    print(f"   IoU per class: {[f'{iou:.2f}' for iou in ious]}")
    print(f"   mIoU: {sum(ious)/len(ious):.2f}\n")

    # 5. 模型对比
    print("5. 模型对比:")
    models = {
        "FCN": FCN(num_classes=5, in_channels=3),
        "U-Net": UNet(in_channels=3, num_classes=5, base_channels=32),
        "DeepLabV3": DeepLabV3(in_channels=3, num_classes=5),
    }

    x = torch.randn(1, 3, 224, 224)

    for name, model in models.items():
        output = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f"   {name:12s}: params={params:>8,}, output={output.shape}")

    print("\n语义分割基础教程完成!")
