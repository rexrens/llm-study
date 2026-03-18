"""
PyTorch Tensors 基础教程
张量 (Tensor) 是 PyTorch 中最基本的数据结构，类似于 NumPy 的数组但支持 GPU 加速。
"""

import torch

# 1. 创建张量
print("=== 1. 创建张量 ===")

# 从 Python 数据结构创建
x = torch.tensor([1, 2, 3])
print(f"从列表创建: {x}")

# 从 NumPy 创建
import numpy as np
np_array = np.array([1, 2, 3])
x_np = torch.from_numpy(np_array)
print(f"从 NumPy 创建: {x_np}")

# 创建特定形状和值的张量
zeros = torch.zeros(2, 3)  # 全零
ones = torch.ones(2, 3)  # 全一
rand = torch.rand(2, 3)  # 0-1 随机
randn = torch.randn(2, 3)  # 标准正态分布
eye = torch.eye(3)  # 单位矩阵

print(f"零矩阵:\n{zeros}")
print(f"随机矩阵:\n{rand}")

# 2. 张量属性
print("\n=== 2. 张量属性 ===")
x = torch.randn(2, 3)
print(f"形状: {x.shape}")
print(f"数据类型: {x.dtype}")
print(f"设备: {x.device}")

# 3. 索引和切片
print("\n=== 3. 索引和切片 ===")
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"原张量:\n{x}")
print(f"第一行: {x[0]}")
print(f"第一列: {x[:, 0]}")
print(f"子张量: {x[0:1, 1:3]}")

# 4. 运算
print("\n=== 4. 张量运算 ===")
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# 逐元素运算
print(f"加法: {a + b}")
print(f"乘法: {a * b}")
print(f"幂运算: {a ** 2}")

# 矩阵运算
A = torch.randn(2, 3)
B = torch.randn(3, 2)
C = torch.matmul(A, B)  # 矩阵乘法
print(f"矩阵乘法形状: {C.shape}")

# 5. 广播机制
print("\n=== 5. 广播机制 ===")
x = torch.ones(3, 1)
y = torch.ones(1, 3)
z = x + y
print(f"x (3,1): {x.squeeze().tolist()}")
print(f"y (1,3): {y.squeeze().tolist()}")
print(f"x + y (3,3):\n{z}")

# 6. 形状操作
print("\n=== 6. 形状操作 ===")
x = torch.randn(4, 6)
print(f"原形状: {x.shape}")

# reshape - 元素总数不变
y = x.reshape(3, 8)
print(f"reshape (3,8): {y.shape}")

# view - 更高效，要求内存连续
z = x.view(2, 12)
print(f"view (2,12): {z.shape}")

# transpose
t = x.transpose(0, 1)
print(f"transpose: {t.shape}")

# unsqueeze / squeeze
x = torch.tensor([1, 2, 3])
y = x.unsqueeze(0)  # 增加维度
z = y.squeeze()  # 移除维度
print(f"unsqueeze(0) 后形状: {y.shape}")
print(f"squeeze() 后形状: {z.shape}")

# 7. GPU/CPU 互转
print("\n=== 7. GPU/CPU 互转 ===")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

x = torch.randn(2, 3)
x_gpu = x.to(device)  # 移到 GPU
x_cpu = x_gpu.cpu()  # 移回 CPU
print(f"GPU 张量设备: {x_gpu.device}")

# 8. 聚合操作
print("\n=== 8. 聚合操作 ===")
x = torch.randn(2, 3)
print(f"张量:\n{x}")
print(f"求和: {x.sum()}")
print(f"平均值: {x.mean()}")
print(f"最大值: {x.max()}")
print(f"沿 dim=0 求和: {x.sum(dim=0)}")
print(f"沿 dim=1 求和: {x.sum(dim=1)}")

# 9. PyTorch vs NumPy 互转
print("\n=== 9. PyTorch vs NumPy ===")
x_np = np.array([1, 2, 3])
x_torch = torch.from_numpy(x_np)
x_back = x_torch.numpy()

print(f"NumPy -> PyTorch: {x_torch}")
print(f"PyTorch -> NumPy: {x_back}")

# 10. 常见模式
print("\n=== 10. 常见模式 ===")

# 创建可训练参数
w = torch.randn(10, 10, requires_grad=True)
print(f"可训练参数: requires_grad={w.requires_grad}")

# 创建不可训练参数
w = torch.randn(10, 10, requires_grad=False)
print(f"不可训练参数: requires_grad={w.requires_grad}")

# detach - 分离计算图
x = torch.randn(2, 3, requires_grad=True)
y = x.detach()  # y 不需要梯度
print(f"detach 后 requires_grad={y.requires_grad}")

# 示例：简单的神经网络前向传播
print("\n=== 示例: 简单前向传播 ===")
batch_size = 4
input_features = 10
output_features = 5

# 模拟输入
x = torch.randn(batch_size, input_features)

# 权重和偏置
W = torch.randn(input_features, output_features, requires_grad=True)
b = torch.randn(output_features, requires_grad=True)

# 前向传播
output = torch.matmul(x, W) + b
print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
print(f"输出:\n{output}")
