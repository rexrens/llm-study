# 01-basics: PyTorch 基础

本模块介绍 PyTorch 的基础概念，包括张量操作和自动求导系统。

## 学习目标

- 理解 PyTorch 张量 (Tensor) 的基本概念和操作
- 掌握张量的创建、索引、切片和形状变换
- 理解自动求导 (Autograd) 的工作原理
- 学会使用梯度进行参数优化

## 文件说明

### tensors.py - 张量基础

张量是 PyTorch 中最核心的数据结构，类似于 NumPy 数组但支持 GPU 加速。

```bash
# 运行示例
python 01-basics/tensors.py
```

**主要内容包括：**
- 张量的创建方法
- 索引和切片
- 张量运算（逐元素、矩阵运算）
- 广播机制
- 形状操作
- GPU/CPU 互转
- 聚合操作

### autograd.py - 自动求导

Autograd 是 PyTorch 的自动求导系统，用于计算神经网络中的梯度。

```bash
# 运行示例
python 01-basics/autograd.py
```

**主要内容包括：**
- 基本的梯度计算
- 梯度累积与清零
- 多变量求导
- 禁用梯度计算（用于推理）
- 梯度裁剪（防止梯度爆炸）
- 自定义反向传播
- 线性回归的梯度下降实例

### exercises.py - 练习题

包含 10 个练习题，涵盖张量和自动求导的核心概念。

```bash
# 运行练习
python 01-basics/exercises.py
```

**练习内容包括：**
1. 张量创建
2. 索引和切片
3. 形状操作
4. 张量运算
5. 广播机制
6. 自动求导
7. 神经网络前向传播
8. 梯度下降优化
9. GPU 操作
10. 综合应用

## 核心概念

### Tensor (张量)

张量是 PyTorch 中的核心数据结构，类似于 NumPy 的多维数组：

```python
import torch

# 创建张量
x = torch.tensor([1, 2, 3])
x = torch.zeros(2, 3)      # 全零
x = torch.rand(2, 3)       # 随机
x = torch.randn(2, 3)      # 正态分布
```

### Autograd (自动求导)

通过 `requires_grad=True` 标记需要计算梯度的张量：

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()  # 反向传播
print(x.grad)  # 4.0
```

### 梯度清零

重要：每次反向传播后需要清零梯度：

```python
optimizer.zero_grad()  # 或手动
x.grad.zero_()
```

### 禁用梯度（推理时）

使用 `torch.no_grad()` 或 `detach()` 禁用梯度计算以提高性能：

```python
with torch.no_grad():
    output = model(input)

# 或
output = model(input).detach()
```

## 常见误区

1. **忘记清零梯度**：每次反向传播后必须清零梯度，否则梯度会累积
2. **in-place 操作**：使用 `_` 后缀的操作会直接修改张量，可能影响计算图
3. **GPU 内存泄漏**：长时间运行时，不用的张量可能占用 GPU 内存
4. **数据类型不匹配**：确保所有张量的数据类型一致

## 下一步

完成本模块后，继续学习：
- [02-data](../02-data/README.md) - 数据加载和处理
- [03-nn-basics](../03-nn-basics/README.md) - 神经网络基础
