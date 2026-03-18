# 04-cnn: 卷积神经网络

本模块介绍卷积神经网络（CNN）的实现和训练，包括经典的 LeNet-5 和现代的 ResNet。

## 学习目标

- 理解卷积神经网络的基本原理
- 掌握 LeNet-5 架构和实现
- 理解残差连接和 ResNet 架构
- 学会训练和评估 CNN 模型

## 文件说明

### lenet.py - LeNet-5 实现

LeNet-5 是最早的卷积神经网络之一，由 Yann LeCun 于 1998 年提出。

```bash
# 运行示例
python 04-cnn/lenet.py
```

**包含内容：**
- 原始 LeNet-5 架构
- 现代化 LeNet-5（使用 MaxPool、BatchNorm、Dropout）

### resnet.py - ResNet 实现

ResNet 通过残差连接解决了深层网络难以训练的问题。

```bash
# 运行示例
python 04-cnn/resnet.py
```

**包含内容：**
- BasicBlock（用于 ResNet-18/34）
- Bottleneck（用于 ResNet-50/101/152）
- SmallResNet（简化版，适用于小图像）
- 预定义的 ResNet 变体

### train.py - 训练脚本

在 MNIST 或 CIFAR-10 上训练 CNN 模型。

```bash
# 在 MNIST 上训练 LeNet-5
python 04-cnn/train.py --model lenet --dataset mnist

# 在 CIFAR-10 上训练 ResNet
python 04-cnn/train.py --model resnet --dataset cifar10 --num_epochs 20 --batch_size 32

# 查看所有参数
python 04-cnn/train.py --help
```

**主要参数：**
- `--model`: 模型选择（lenet, lenet_modern, resnet）
- `--dataset`: 数据集选择（mnist, cifar10）
- `--batch_size`: 批次大小（默认 64）
- `--num_epochs`: 训练轮数（默认 10）
- `--learning_rate`: 学习率（默认 0.001）
- `--checkpoint_dir`: 检查点保存目录

### predict.py - 推理脚本

使用训练好的模型进行预测和可视化。

```bash
# 使用训练好的模型进行预测
python 04-cnn/predict.py --model lenet --checkpoint checkpoints/best_model.pth --dataset mnist

# 保存混淆矩阵
python 04-cnn/predict.py --model lenet --checkpoint checkpoints/best_model.pth --save_cm
```

## 核心概念

### 卷积神经网络

CNN 主要包含以下组件：

1. **卷积层（Conv2d）**: 提取局部特征
2. **池化层（MaxPool2d/AvgPool2d）**: 降低空间维度
3. **激活函数（ReLU）**: 引入非线性
4. **全连接层**: 最终的分类或回归

### LeNet-5 架构

```
Input (32x32x1)
    ↓
Conv2d(1→6, 5x5) + ReLU + MaxPool(2x2)
    ↓
Conv2d(6→16, 5x5) + ReLU + MaxPool(2x2)
    ↓
Conv2d(16→120, 5x5) + ReLU
    ↓
FC(120→84) + ReLU
    ↓
FC(84→10)
```

### ResNet 核心思想

残差连接允许梯度更容易地流过深层网络：

```
x → Conv1 → BN → ReLU → Conv2 → BN → (+ x) → ReLU → output
```

## 快速开始

### 1. 训练模型

```bash
# 训练 LeNet-5（默认）
python 04-cnn/train.py

# 训练更多 epoch
python 04-cnn/train.py --num_epochs 20

# 使用更大的学习率
python 04-cnn/train.py --learning_rate 0.01
```

### 2. 查看训练结果

训练完成后，检查以下文件：

- `checkpoints/best_model.pth` - 最佳模型检查点
- `checkpoints/training_curves.png` - 训练曲线图

### 3. 使用模型预测

```bash
python 04-cnn/predict.py \
    --model lenet \
    --checkpoint checkpoints/best_model.pth \
    --dataset mnist
```

预测结果将保存为：
- `predictions.png` - 预测样本的可视化
- `confusion_matrix.png` - 混淆矩阵（使用 `--save_cm` 参数）

## 模型对比

| 模型 | 参数量 | 适用场景 | MNIST 准确率 | CIFAR-10 准确率 |
|------|--------|----------|-------------|---------------|
| LeNet-5 | ~60K | 简单任务，学习 CNN 基础 | ~98% | ~65% |
| LeNet-5 Modern | ~70K | 需要更好的泛化能力 | ~99% | ~70% |
| SmallResNet | ~600K | 中等复杂度任务 | ~99.5% | ~85% |
| ResNet-18 | ~11M | 复杂任务，工业级应用 | N/A | ~92% |

## 常见问题

### 训练时 GPU 内存不足

尝试减小 `batch_size`：

```bash
python 04-cnn/train.py --batch_size 32
```

### 训练曲线震荡

降低学习率或增加学习率衰减：

```bash
python 04-cnn/train.py --learning_rate 0.0001
```

### 模型过拟合

增加数据增强、Dropout 或权重衰减：

```bash
python 04-cnn/train.py --model lenet_modern --weight_decay 0.0001
```

## 下一步

完成本模块后，继续学习：
- [05-advanced-cv](../05-advanced-cv/README.md) - 高级计算机视觉
