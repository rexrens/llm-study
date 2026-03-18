# 03-nn-basics: 神经网络基础

本模块介绍 PyTorch 中的神经网络构建、损失函数、优化器和训练循环。

## 学习目标

- 掌握使用 nn.Module 构建神经网络
- 理解常用的损失函数和优化器
- 学会编写完整的训练循环
- 掌握训练技巧（梯度裁剪、学习率调度、早停等）

## 文件说明

### model_building.py - 模型构建

nn.Module 是 PyTorch 中构建神经网络的核心基类。

```bash
# 运行示例
python 03-nn-basics/model_building.py
```

**主要内容包括：**
- 基本的 nn.Module 使用
- 使用 nn.Sequential 构建模型
- 复杂模型结构（MLP、残差网络）
- 参数初始化
- 共享权重
- 模型保存和加载
- 训练/评估模式切换

### loss_optim.py - 损失函数和优化器

损失函数用于衡量模型预测与真实值的差异，优化器用于更新模型参数。

```bash
# 运行示例
python 03-nn-basics/loss_optim.py
```

**主要内容包括：**
- 常用损失函数（MSE, CrossEntropy, BCE）
- 自定义损失函数
- 常用优化器（SGD, Adam, AdamW）
- 学习率调度器（StepLR, CosineAnnealingLR 等）
- 梯度裁剪
- 学习率预热
- 完整的训练步骤

### training_loop.py - 训练循环

完整的训练循环包括前向传播、损失计算、反向传播、参数更新等步骤。

```bash
# 运行示例
python 03-nn-basics/training_loop.py
```

**主要内容包括：**
- 最简单的训练循环
- 带验证的训练循环
- 带学习率调度的训练循环
- 带梯度裁剪的训练循环
- 检查点保存和恢复
- 早停机制
- 混合精度训练
- 完整的 Trainer 类

## 核心概念

### nn.Module

nn.Module 是所有神经网络模块的基类：

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### nn.Sequential

Sequential 按顺序执行层，适合简单的线性结构：

```python
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 2),
)
```

### 训练循环步骤

1. **前向传播**: `outputs = model(x)`
2. **计算损失**: `loss = criterion(outputs, y)`
3. **清零梯度**: `optimizer.zero_grad()`
4. **反向传播**: `loss.backward()`
5. **更新参数**: `optimizer.step()`

### 训练/评估模式

```python
model.train()   # 训练模式：启用 dropout, batch norm 使用 batch 统计
model.eval()    # 评估模式：禁用 dropout, batch norm 使用运行统计
```

## 常见模式

### 带验证的训练循环

```python
for epoch in range(num_epochs):
    # 训练
    model.train()
    for x, y in train_loader:
        outputs = model(x)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 验证
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            outputs = model(x)
            # 计算验证指标
```

### 保存和加载模型

```python
# 保存模型参数
torch.save(model.state_dict(), "model.pth")

# 加载模型参数
model = MyModel(...)
model.load_state_dict(torch.load("model.pth"))
```

### 学习率调度

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(num_epochs):
    train(...)
    scheduler.step()  # 更新学习率
```

## 最佳实践

1. **模型构建**: 优先使用 nn.Sequential，复杂结构使用 nn.Module
2. **参数初始化**: Xavier/He 初始化通常效果更好
3. **优化器选择**: Adam 是默认选择，SGD + momentum 适合精细调优
4. **梯度处理**: 训练稳定时不需要裁剪，RNN/LSTM 建议裁剪
5. **检查点**: 定期保存检查点，防止训练中断
6. **早停**: 根据验证损失决定是否提前停止训练
7. **混合精度**: GPU 训练时使用可加速并节省内存

## 常见误区

1. **忘记 eval()**: 推理时忘记切换到 eval 模式
2. **忘记 no_grad()**: 推理时忘记禁用梯度计算
3. **忘记清零梯度**: 每次反向传播前需要清零
4. **不恰当的学习率**: 太大不收敛，太小收敛慢
5. **保存整个模型**: 应该只保存 state_dict

## 下一步

完成本模块后，继续学习：
- [04-cnn](../04-cnn/README.md) - 卷积神经网络
- [05-advanced-cv](../05-advanced-cv/README.md) - 高级计算机视觉
