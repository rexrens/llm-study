# PyTorch 教程

一个全面的 PyTorch 学习资源，涵盖从基础到高级计算机视觉应用。

## 目录

```
torch_learn/
├── README.md                  # 本文件
├── pyproject.toml             # 项目配置
├── 01-basics/                 # PyTorch 基础
│   ├── tensors.py             # 张量操作
│   ├── autograd.py            # 自动求导
│   ├── exercises.py           # 练习题
│   └── README.md
├── 02-data/                  # 数据加载和处理
│   ├── dataset.py             # 自定义 Dataset
│   ├── dataloader.py          # DataLoader 使用
│   ├── transforms.py          # 数据变换
│   └── README.md
├── 03-nn-basics/             # 神经网络基础
│   ├── model_building.py      # 模型构建
│   ├── loss_optim.py          # 损失函数和优化器
│   ├── training_loop.py       # 训练循环
│   └── README.md
├── 04-cnn/                   # 卷积神经网络
│   ├── lenet.py               # LeNet-5 实现
│   ├── resnet.py              # ResNet 实现
│   ├── train.py               # 训练脚本
│   ├── predict.py             # 推理脚本
│   └── README.md
├── 05-advanced-cv/           # 高级计算机视觉
│   ├── transfer_learning.py   # 迁移学习
│   ├── object_detection.py    # 目标检测基础
│   ├── segmentation.py        # 语义分割
│   └── README.md
├── common/                   # 公共模块
│   ├── config.py              # 配置管理
│   ├── logging.py             # 日志工具
│   └── metrics.py             # 评估指标
└── utils/                    # 工具函数
    ├── device.py              # 设备选择
    └── seed.py               # 随机种子
```

## 学习路径

### 第一步：PyTorch 基础

学习 PyTorch 的核心概念：张量和自动求导。

```bash
cd 01-basics
python tensors.py      # 张量操作
python autograd.py     # 自动求导
python exercises.py    # 练习题
```

**学习目标：**
- 理解张量的创建、操作和变换
- 掌握自动求导的原理和使用

### 第二步：数据加载

学习如何高效地加载和处理数据。

```bash
cd 02-data
python dataset.py      # 自定义数据集
python dataloader.py   # 数据加载器
python transforms.py   # 数据增强
```

**学习目标：**
- 掌握 Dataset 和 DataLoader 的使用
- 理解数据增强和预处理

### 第三步：神经网络基础

学习构建和训练神经网络。

```bash
cd 03-nn-basics
python model_building.py  # 模型构建
python loss_optim.py      # 损失和优化
python training_loop.py   # 完整训练流程
```

**学习目标：**
- 理解 nn.Module 和常用层
- 掌握损失函数和优化器
- 学会编写训练循环

### 第四步：卷积神经网络

学习 CNN 架构和在图像分类中的应用。

```bash
cd 04-cnn
python lenet.py      # LeNet-5
python resnet.py     # ResNet

# 训练模型
python train.py --model lenet --dataset mnist

# 使用模型预测
python predict.py --model lenet --checkpoint checkpoints/best_model.pth --dataset mnist
```

**学习目标：**
- 理解 CNN 的原理和架构
- 掌握 LeNet-5 和 ResNet
- 学会训练和评估 CNN 模型

### 第五步：高级计算机视觉

学习迁移学习、目标检测和语义分割。

```bash
cd 05-advanced-cv
python transfer_learning.py   # 迁移学习
python object_detection.py    # 目标检测
python segmentation.py        # 语义分割
```

**学习目标：**
- 理解迁移学习的概念
- 了解目标检测和分割的基本原理

## 快速开始

### 环境设置

本教程使用 [uv](https://github.com/astral-sh/uv) 作为包管理器：

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步依赖
uv sync

# 激活虚拟环境
source .venv/bin/activate  # Linux/macOS
# 或 .venv\Scripts\activate  # Windows
```

### 运行示例

```bash
# 基础示例
python 01-basics/tensors.py

# 训练一个简单的 CNN
python 04-cnn/train.py --model lenet --dataset mnist --num_epochs 5

# 迁移学习
python 05-advanced-cv/transfer_learning.py --model resnet18 --dataset cifar10
```

## 项目结构说明

### common/ - 公共模块

提供跨模块共享的工具：

- `config.py`: YAML 配置管理
- `logging.py`: 结构化日志
- `metrics.py`: 评估指标和可视化

### utils/ - 工具函数

- `device.py`: 智能设备选择（CPU/CUDA/MPS）
- `seed.py`: 随机种子设置

## 模型性能参考

### MNIST 手写数字识别

| 模型 | 参数量 | 准确率 | 训练时间 |
|------|--------|--------|----------|
| LeNet-5 | ~60K | ~98.5% | ~1 min |
| LeNet-5 Modern | ~70K | ~99.2% | ~1 min |
| SmallResNet | ~600K | ~99.5% | ~2 min |

### CIFAR-10 图像分类

| 模型 | 参数量 | 准确率 | 训练时间 |
|------|--------|--------|----------|
| LeNet-5 | ~60K | ~65% | ~2 min |
| LeNet-5 Modern | ~70K | ~70% | ~2 min |
| ResNet-18 (预训练) | ~11M | ~92% | ~10 min |
| ResNet-18 (从头训练) | ~11M | ~88% | ~15 min |

*注：训练时间基于单张 RTX 3090 GPU，仅供参考*

## 最佳实践

1. **学习顺序**: 按照 01 → 02 → 03 → 04 → 05 的顺序学习
2. **动手实践**: 每个模块都包含可运行的代码示例
3. **理解原理**: 不要只运行代码，要理解背后的原理
4. **尝试修改**: 修改参数和结构，观察结果变化
5. **记录笔记**: 记录重要的概念和技巧

## 常见问题

### GPU 不可用

如果没有 GPU，PyTorch 会自动使用 CPU：

```python
from utils.device import get_device
device = get_device()  # 自动选择最佳设备
```

### 内存不足

- 减小 `batch_size`
- 使用梯度累积
- 使用更小的模型

### 训练不收敛

- 检查学习率
- 确保数据加载正确
- 检查损失函数

## 推荐资源

### 官方文档
- [PyTorch 文档](https://pytorch.org/docs/)
- [PyTorch 教程](https://pytorch.org/tutorials/)

### 视频教程
- PyTorch 官方 YouTube 频道
- DeepLearning.AI PyTorch 课程

### 书籍
- 《深度学习与 PyTorch 实战》
- 《Python 深度学习实战》

## 许可证

本项目仅供学习使用。

## 贡献

欢迎提出建议和改进！
