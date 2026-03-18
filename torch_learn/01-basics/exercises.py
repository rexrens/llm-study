"""
PyTorch 练习题
完成以下练习以巩固张量和自动求导的知识。
"""

import torch
import numpy as np

# 练习 1: 张量基础
print("=== 练习 1: 张量基础 ===")

# 1.1 创建一个 3x4 的随机张量，值在 [0, 1) 之间
random_tensor = torch.rand(3, 4)
print("1.1 随机张量:", random_tensor.shape)

# 1.2 创建一个 5x5 的单位矩阵
identity = torch.eye(5)
print("1.2 单位矩阵:", identity.shape)

# 1.3 创建一个形状为 (2, 3, 4) 的全零张量
zeros_3d = torch.zeros(2, 3, 4)
print("1.3 零张量:", zeros_3d.shape)

# 练习 2: 索引和切片
print("\n=== 练习 2: 索引和切片 ===")
x = torch.arange(24).reshape(4, 6)
print("原始张量:\n", x)

# 2.1 提取第二行
row_2 = x[1]
print("2.1 第二行:", row_2)

# 2.2 提取第 3 到 5 列
cols_3_5 = x[:, 2:5]
print("2.2 第 3-5 列:\n", cols_3_5)

# 2.3 提取 (1, 2) 到 (3, 4) 的子矩阵
sub_matrix = x[1:3, 2:4]
print("2.3 子矩阵:\n", sub_matrix)

# 练习 3: 形状操作
print("\n=== 练习 3: 形状操作 ===")
x = torch.randn(4, 6)
print(f"原始形状: {x.shape}")

# 3.1 展平为一维
flattened = x.flatten()
print(f"3.1 展平后: {flattened.shape}")

# 3.2 reshape 为 (3, 8)
reshaped = x.reshape(3, 8)
print(f"3.2 reshape: {reshaped.shape}")

# 3.3 增加一个维度变成 (1, 4, 6)
unsqueezed = x.unsqueeze(0)
print(f"3.3 unsqueeze: {unsqueezed.shape}")

# 练习 4: 张量运算
print("\n=== 练习 4: 张量运算 ===")
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# 4.1 逐元素相加
elementwise_sum = a + b
print("4.1 逐元素相加:\n", elementwise_sum)

# 4.2 矩阵乘法
matmul = torch.matmul(a, b)
print("4.2 矩阵乘法:\n", matmul)

# 4.3 计算 a 的转置
transposed = a.T
print("4.3 转置:\n", transposed)

# 练习 5: 广播
print("\n=== 练习 5: 广播 ===")
# 5.1 将 (3, 1) 和 (1, 3) 的张量相加，观察结果
x = torch.ones(3, 1)
y = torch.ones(1, 3)
broadcast_sum = x + y
print("5.1 广播结果:\n", broadcast_sum)

# 练习 6: 自动求导
print("\n=== 练习 6: 自动求导 ===")

# 6.1 计算 f(x) = x^3 在 x=2 处的导数
x = torch.tensor([2.0], requires_grad=True)
f = x ** 3
f.backward()
print("6.1 f'(2) =", x.grad.item())  # 应该是 12

# 6.2 计算 z = x^2 + y^2 在 (x=1, y=2) 处的梯度
x = torch.tensor([1.0], requires_grad=True)
y = torch.tensor([2.0], requires_grad=True)
z = x ** 2 + y ** 2
z.backward()
print("6.2 dz/dx =", x.grad.item(), ", dz/dy =", y.grad.item())

# 练习 7: 简单的神经网络前向传播
print("\n=== 练习 7: 神经网络前向传播 ===")

# 7.1 实现一个简单的前向传播: output = ReLU(x @ W + b)
x = torch.randn(4, 10)  # 4 个样本, 10 个特征
W = torch.randn(10, 5, requires_grad=True)
b = torch.randn(5, requires_grad=True)

# 前向传播
output = torch.relu(torch.matmul(x, W) + b)
print("7.1 输出形状:", output.shape)

# 7.2 计算输出均值并反向传播
loss = output.mean()
loss.backward()
print("7.2 W 的梯度形状:", W.grad.shape)

# 练习 8: 梯度下降优化
print("\n=== 练习 8: 梯度下降优化 ===")

# 8.1 使用梯度下降找到 f(x) = (x - 3)^2 的最小值
x = torch.tensor([10.0], requires_grad=True)
learning_rate = 0.1

for i in range(20):
    f = (x - 3) ** 2
    f.backward()

    with torch.no_grad():
        x -= learning_rate * x.grad
        x.grad.zero_()

    if (i + 1) % 5 == 0:
        print(f"Iter {i+1}: x = {x.item():.4f}, f(x) = {f.item():.4f}")

print(f"最终结果: x = {x.item():.4f} (最优解是 3)")

# 练习 9: GPU 操作
print("\n=== 练习 9: GPU 操作 ===")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 创建张量并移到 GPU
x_cpu = torch.randn(100, 100)
x_gpu = x_cpu.to(device)

# 在 GPU 上进行运算
y_gpu = x_gpu @ x_gpu

# 移回 CPU
y_cpu = y_gpu.cpu()
print(f"GPU 运算结果形状: {y_cpu.shape}")

# 练习 10: 综合应用
print("\n=== 练习 10: 综合应用 ===")

# 实现一个简单的二元分类模型的前向传播和损失计算
batch_size = 8
input_dim = 10
output_dim = 2

# 输入数据
x = torch.randn(batch_size, input_dim, requires_grad=False)
y = torch.randint(0, 2, (batch_size,))

# 模型参数
W1 = torch.randn(input_dim, 16, requires_grad=True)
b1 = torch.randn(16, requires_grad=True)
W2 = torch.randn(16, output_dim, requires_grad=True)
b2 = torch.randn(output_dim, requires_grad=True)

# 前向传播
hidden = torch.relu(torch.matmul(x, W1) + b1)
logits = torch.matmul(hidden, W2) + b2

# 计算交叉熵损失
loss = torch.nn.functional.cross_entropy(logits, y)

print(f"Loss: {loss.item():.4f}")

# 反向传播
loss.backward()

print(f"各参数梯度形状:")
print(f"W1: {W1.grad.shape}")
print(f"W2: {W2.grad.shape}")

print("\n所有练习完成!")
