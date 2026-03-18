# 05-advanced-cv: 高级计算机视觉

本模块介绍高级计算机视觉任务，包括迁移学习、目标检测和语义分割。

## 学习目标

- 理解迁移学习的概念和应用
- 掌握使用预训练模型进行微调
- 了解目标检测的基本原理
- 了解语义分割的主要架构

## 文件说明

### transfer_learning.py - 迁移学习

使用预训练模型（ResNet、EfficientNet、MobileNet）进行迁移学习。

```bash
# 演示迁移学习
python 05-advanced-cv/transfer_learning.py

# 在 CIFAR-10 上微调 ResNet-18
python 05-advanced-cv/transfer_learning.py --model resnet18 --dataset cifar10

# 冻结主干网络，只训练分类头
python 05-advanced-cv/transfer_learning.py --model resnet50 --freeze_backbone
```

**主要参数：**
- `--model`: 预训练模型（resnet18, resnet50, efficientnet_b0, mobilenet_v3_small）
- `--dataset`: 数据集（cifar10, flowers102, custom）
- `--freeze_backbone`: 是否冻结主干网络
- `--learning_rate`: 学习率
- `--num_epochs`: 训练轮数

**关键概念：**
- **预训练模型**: 在大规模数据集（如 ImageNet）上训练好的模型
- **微调**: 在新数据集上继续训练预训练模型
- **冻结主干**: 固定特征提取部分，只训练分类层

### object_detection.py - 目标检测

介绍目标检测的基本概念和实现原理。

```bash
# 运行演示
python 05-advanced-cv/object_detection.py
```

**包含内容：**
- 锚框（Anchors）生成
- IoU (Intersection over Union) 计算
- 非极大值抑制 (NMS)
- 边界框回归
- YOLO 概念性实现

### segmentation.py - 语义分割

介绍语义分割的网络架构。

```bash
# 运行演示
python 05-advanced-cv/segmentation.py
```

**包含内容：**
- FCN (Fully Convolutional Network)
- U-Net（编码器-解码器结构）
- DeepLabV3（空洞卷积）
- IoU 计算方法

## 核心概念

### 迁移学习

迁移学习利用在大规模数据集上预训练的模型，加速新任务的训练：

```python
from torchvision import models

# 加载预训练模型
model = models.resnet18(pretrained=True)

# 修改输出层以适应新任务
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 微调
for param in model.parameters():
    param.requires_grad = True  # 可选: 冻结部分参数
```

### 目标检测关键概念

1. **锚框（Anchors）**: 预定义的边界框，用于定位目标
2. **IoU**: 衡量两个边界框重叠程度的指标
3. **NMS**: 移除重叠度高的冗余检测框
4. **两阶段检测器**: 如 Faster R-CNN，先生成区域建议再分类
5. **单阶段检测器**: 如 YOLO、SSD，直接预测类别和位置

### 语义分割关键概念

1. **FCN**: 将分类网络转换为全卷积网络，输出像素级预测
2. **U-Net**: 编码器-解码器结构，通过跳跃连接保留空间信息
3. **ASPP**: 空洞金字塔池化，扩大感受野

## 迁移学习策略

### 1. 微调整个网络

适用于新数据集较大、与预训练数据集相似的情况：

```bash
python 05-advanced-cv/transfer_learning.py --model resnet18 --dataset cifar10
```

### 2. 冻结主干网络

适用于新数据集较小、计算资源有限的情况：

```bash
python 05-advanced-cv/transfer_learning.py --model resnet50 --freeze_backbone
```

### 3. 差异化学习率

主干网络使用较小学习率，分类头使用较大学习率：

```python
optimizer = torch.optim.Adam([
    {"params": backbone_params, "lr": 0.0001},
    {"params": classifier_params, "lr": 0.001},
])
```

## 模型对比

### 迁移学习模型

| 模型 | 参数量 | 特点 | 适用场景 |
|------|--------|------|----------|
| ResNet-18 | ~11M | 平衡性能和效率 | 通用场景 |
| ResNet-50 | ~25M | 更强性能 | 需要更高精度 |
| EfficientNet-B0 | ~5M | 高效 | 移动设备 |
| MobileNet-V3-S | ~2.5M | 极致高效 | 边缘设备 |

### 语义分割模型

| 模型 | 参数量 | 特点 | 适用场景 |
|------|--------|------|----------|
| FCN | ~20M | 简单快速 | 基础分割 |
| U-Net | ~30M | 跳跃连接 | 医学图像 |
| DeepLabV3 | ~40M | 空洞卷积 | 复杂场景 |

## 数据准备

### 自定义数据集

对于自定义数据集，组织如下结构：

```
data/
├── train/
│   ├── class_0/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class_1/
│       └── ...
└── val/
    └── ...
```

使用时：

```bash
python 05-advanced-cv/transfer_learning.py \
    --dataset custom \
    --data_dir data \
    --num_classes 2
```

## 常见问题

### 迁移学习效果不佳

1. 确保预处理与预训练模型一致
2. 调整学习率（通常比从头训练更小）
3. 尝试不同预训练模型

### 检测框不准确

- 增加锚框的数量和多样性
- 调整 IoU 阈值
- 使用更好的特征提取器

### 分割边缘模糊

- 使用跳跃连接（U-Net）
- 增加模型容量
- 尝试空洞卷积（DeepLabV3）

## 下一步

完成本模块后，可以：

1. 尝试在真实数据集上应用迁移学习
2. 学习使用高级检测框架（如 Detectron2、MMDetection）
3. 探索分割框架（如 segmentation_models_pytorch）

## 参考资源

- **预训练模型**: torchvision.models
- **检测框架**: Detectron2, MMDetection
- **分割框架**: segmentation_models_pytorch
