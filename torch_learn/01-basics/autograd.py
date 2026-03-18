"""
PyTorch Autograd 自动求导教程
Autograd 是 PyTorch 的自动求导系统，用于计算神经网络中的梯度。
"""

import torch

# 1. 基本概念
print("=== 1. 基本概念 ===")

# requires_grad=True 表示需要计算梯度
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x ** 2 + 2 * x + 1

# 计算梯度
y_sum = y.sum()
y_sum.backward()  # 反向传播

print(f"x: {x}")
print(f"y: {y}")
print(f"dy/dx: {x.grad}")  # 应该是 [6, 8] 因为 dy/dx = 2x + 2

# 2. 梯度累积
print("\n=== 2. 梯度累积 ===")
x = torch.tensor([2.0], requires_grad=True)

for i in range(3):
    y = x ** 2
    y.backward()
    print(f"第 {i+1} 次梯度: {x.grad.item()}")

    # 重要：使用前需要清零梯度
    x.grad.zero_()

# 3. 多变量求导
print("\n=== 3. 多变量求导 ===")
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = torch.tensor([3.0, 4.0], requires_grad=True)

z = x ** 2 + y ** 2
z_sum = z.sum()
z_sum.backward()

print(f"x 的梯度: {x.grad}")  # [2, 4]
print(f"y 的梯度: {y.grad}")  # [6, 8]

# 4. 禁用梯度计算 (推理时使用)
print("\n=== 4. 禁用梯度计算 ===")
x = torch.randn(2, 3, requires_grad=True)

# 方法 1: no_grad 上下文管理器
with torch.no_grad():
    y = x + 1
    print(f"no_grad 后 requires_grad={y.requires_grad}")

# 方法 2: detach
y_detached = x.detach()
print(f"detach 后 requires_grad={y_detached.requires_grad}")

# 性能对比
import time

x = torch.randn(1000, 1000, requires_grad=True)

start = time.time()
for _ in range(100):
    with torch.no_grad():
        y = x @ x
print(f"no_grad 耗时: {time.time() - start:.4f}s")

start = time.time()
for _ in range(100):
    y = x @ x
print(f"with_grad 耗时: {time.time() - start:.4f}s")

# 5. 梯度裁剪 (防止梯度爆炸)
print("\n=== 5. 梯度裁剪 ===")
model = torch.nn.Linear(10, 5)
x = torch.randn(4, 10, requires_grad=True)

# 前向传播
output = model(x)
loss = output.sum()

# 反向传播
loss.backward()

# 查看梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: max grad = {param.grad.abs().max().item():.4f}")

# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

print("\n裁剪后:")
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: max grad = {param.grad.abs().max().item():.4f}")

# 6. 自定义反向传播
print("\n=== 6. 自定义反向传播 ===")

class CustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)  # 保存前向传播的输入
        return x ** 3

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # dy/dx = 3 * x^2 * grad_output
        return 3 * x ** 2 * grad_output

x = torch.tensor([2.0], requires_grad=True)
y = CustomFunction.apply(x)
y.backward()

print(f"自定义梯度: {x.grad.item()}")  # 应该是 12 (3 * 2^2)

# 7. 雅可比矩阵和海森矩阵 (高级)
print("\n=== 7. 雅可比矩阵 (高级) ===")
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = torch.stack([x[0] ** 2, x[1] ** 3])

# 计算雅可比矩阵
jacobian = torch.autograd.functional.jacobian(
    lambda x: torch.stack([x[0] ** 2, x[1] ** 3]),
    x
)
print(f"雅可比矩阵:\n{jacobian}")

# 8. 实例: 线性回归的梯度下降
print("\n=== 8. 实例: 线性回归 ===")

# 生成数据
torch.manual_seed(42)
x = torch.randn(100, 1)
true_w, true_b = 3.0, 2.0
y = true_w * x + true_b + 0.1 * torch.randn(100, 1)

# 初始化参数
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

learning_rate = 0.01
num_epochs = 100

print(f"真实参数: w={true_w}, b={true_b}")
print(f"初始参数: w={w.item():.2f}, b={b.item():.2f}")

for epoch in range(num_epochs):
    # 前向传播
    y_pred = w * x + b

    # 计算损失 (MSE)
    loss = torch.mean((y_pred - y) ** 2)

    # 反向传播
    loss.backward()

    # 更新参数 (梯度下降)
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

        # 清零梯度
        w.grad.zero_()
        b.grad.zero_()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}, w={w.item():.2f}, b={b.item():.2f}")

print(f"学习参数: w={w.item():.2f}, b={b.item():.2f}")
