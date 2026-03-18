"""
PyTorch 损失函数和优化器教程
损失函数用于衡量模型预测与真实值的差异，优化器用于更新模型参数。
"""

import torch
import torch.nn as nn
import torch.optim as optim

# 1. 损失函数基础
print("=== 1. 损失函数基础 ===")

# 模拟预测和真实值
logits = torch.randn(4, 10)  # batch_size=4, num_classes=10
labels = torch.randint(0, 10, (4,))  # 真实类别

# 交叉熵损失 (分类任务最常用)
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, labels)
print(f"CrossEntropyLoss: {loss.item():.4f}")

# 注意: CrossEntropyLoss 内部包含了 LogSoftmax


# 2. 常用损失函数
print("\n=== 2. 常用损失函数 ===")

# 回归任务
# MSELoss - 均方误差
predictions = torch.randn(4, 1)
targets = torch.randn(4, 1)
mse_loss = nn.MSELoss()(predictions, targets)
print(f"MSELoss: {mse_loss.item():.4f}")

# L1Loss - 平均绝对误差
l1_loss = nn.L1Loss()(predictions, targets)
print(f"L1Loss: {l1_loss.item():.4f}")

# SmoothL1Loss - 结合了 L1 和 L2 的优点
smooth_l1 = nn.SmoothL1Loss()(predictions, targets)
print(f"SmoothL1Loss: {smooth_l1.item():.4f}")

# 分类任务
# NLLLoss - 负对数似然 (需要先 LogSoftmax)
log_probs = torch.log_softmax(logits, dim=-1)
nll_loss = nn.NLLLoss()(log_probs, labels)
print(f"NLLLoss: {nll_loss.item():.4f}")

# 二分类
binary_logits = torch.randn(4, 1)
binary_labels = torch.randint(0, 2, (4, 1), dtype=torch.float)
bce_loss = nn.BCEWithLogitsLoss()(binary_logits, binary_labels)
print(f"BCEWithLogitsLoss: {bce_loss.item():.4f}")

# 多标签分类 (每个样本可以有多个标签)
multi_label_targets = torch.randint(0, 2, (4, 10), dtype=torch.float)
multi_label_loss = nn.BCEWithLogitsLoss()(logits, multi_label_targets)
print(f"多标签 BCE Loss: {multi_label_loss.item():.4f}")


# 3. 损失函数的选择
print("\n=== 3. 损失函数选择指南 ===")

"""
回归任务:
- MSELoss: 最常用，对异常值敏感
- L1Loss: 对异常值鲁棒
- SmoothL1Loss: 平滑版本，综合两者优点

分类任务:
- 二分类: BCEWithLogitsLoss
- 多分类: CrossEntropyLoss (最常用)
- 多标签: BCEWithLogitsLoss

其他:
- HingeEmbeddingLoss: 用于度量学习
- TripletMarginLoss: 用于人脸识别等
- CTCLoss: 用于序列标注 (OCR, ASR)
"""


# 4. 自定义损失函数
print("\n=== 4. 自定义损失函数 ===")

class FocalLoss(nn.Module):
    """Focal Loss 用于处理类别不平衡"""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, num_classes)
            targets: (batch_size,)
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()


# 测试 Focal Loss
focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
loss = focal_loss(logits, labels)
print(f"Focal Loss: {loss.item():.4f}")


# 5. 常用优化器
print("\n=== 5. 常用优化器 ===")

# 创建一个简单模型
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 2),
)

# SGD (随机梯度下降)
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
print("SGD with momentum: 适合大多数情况")

# Adam (自适应矩估计)
optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
print("Adam: 收敛快，默认选择")

# AdamW (带权重衰减的 Adam)
optimizer_adamw = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
print("AdamW: 带权重衰减，防止过拟合")

# RMSprop
optimizer_rmsprop = optim.RMSprop(model.parameters(), lr=0.01)
print("RMSprop: 适合非平稳目标")


# 6. 学习率调度器
print("\n=== 6. 学习率调度器 ===")

# 创建模型和优化器
model = nn.Linear(10, 2)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# StepLR: 每隔 step_size 个 epoch，学习率乘以 gamma
scheduler_step = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# MultiStepLR: 在指定的 epoch 降低学习率
scheduler_multistep = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[30, 60, 90], gamma=0.1
)

# ExponentialLR: 每个 epoch 学习率乘以 gamma
scheduler_exp = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# CosineAnnealingLR: 余弦退火
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# ReduceLROnPlateau: 当指标不再改善时降低学习率
scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=5
)

# OneCycleLR: 单周期学习率策略
scheduler_onecycle = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, total_steps=100
)

print("StepLR: 定期衰减")
print("MultiStepLR: 在指定 epoch 衰减")
print("ExponentialLR: 指数衰减")
print("CosineAnnealingLR: 余弦退火")
print("ReduceLROnPlateau: 根据指标自适应")
print("OneCycleLR: 单周期策略")


# 7. 完整的训练步骤
print("\n=== 7. 完整训练步骤 ===")

# 创建模型和数据
model = nn.Linear(10, 2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 模拟数据
x = torch.randn(4, 10)
y = torch.randint(0, 2, (4,))

# 训练步骤
model.train()  # 设置为训练模式

# 1. 前向传播
outputs = model(x)
loss = criterion(outputs, y)
print(f"Step 1 - 前向传播, Loss: {loss.item():.4f}")

# 2. 反向传播
loss.backward()
print("Step 2 - 反向传播")

# 3. 更新参数
optimizer.step()
print("Step 3 - 更新参数")

# 4. 清零梯度
optimizer.zero_grad()
print("Step 4 - 清零梯度")

# 5. 更新学习率
scheduler.step()
print("Step 5 - 更新学习率")


# 8. 参数分组
print("\n=== 8. 参数分组 ===")

# 不同层使用不同的学习率
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 2),
)

optimizer = optim.Adam([
    {"params": model[0].parameters(), "lr": 0.001},  # 第一层用较小学习率
    {"params": model[2].parameters(), "lr": 0.01},   # 输出层用较大学习率
])

print("参数分组: 不同层可以有不同的学习率")


# 9. 梯度裁剪
print("\n=== 9. 梯度裁剪 ===")

# 创建模型并运行一次前向传播
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 2),
)

x = torch.randn(4, 10)
y = torch.randint(0, 2, (4,))

outputs = model(x)
loss = F.cross_entropy(outputs, y)
loss.backward()

# 查看梯度大小
max_grad = max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)
print(f"裁剪前最大梯度: {max_grad:.4f}")

# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 查看裁剪后的梯度
max_grad = max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)
print(f"裁剪后最大梯度: {max_grad:.4f}")


# 10. Warmup (学习率预热)
print("\n=== 10. 学习率预热 ===")

class WarmupScheduler:
    """学习率预热调度器"""

    def __init__(self, optimizer, warmup_steps: int, target_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # 线性预热
            lr = self.target_lr * self.current_step / self.warmup_steps
        else:
            lr = self.target_lr

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


# 使用预热
model = nn.Linear(10, 2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
warmup_scheduler = WarmupScheduler(optimizer, warmup_steps=5, target_lr=0.01)

print("预热期间学习率:")
for step in range(10):
    warmup_scheduler.step()
    print(f"  Step {step+1}: {warmup_scheduler.get_lr():.6f}")


# 11. 实战: 完整的训练循环
print("\n=== 11. 实战: 训练循环 ===")

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        # 前向传播
        outputs = model(x)
        loss = criterion(outputs, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# 模拟使用
from torch.utils.data import TensorDataset, DataLoader

# 创建虚拟数据
x_train = torch.randn(100, 10)
y_train = torch.randint(0, 2, (100,))
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

x_val = torch.randn(20, 10)
y_val = torch.randint(0, 2, (20,))
val_dataset = TensorDataset(x_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 创建模型
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 2),
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
device = torch.device("cpu")

# 训练几个 epoch
print("模拟训练:")
for epoch in range(3):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    scheduler.step()

    print(f"Epoch {epoch+1}: "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")


# 12. 最佳实践
print("\n=== 12. 最佳实践 ===")

"""
优化器选择:
- Adam: 默认选择，收敛快
- SGD + momentum: 适合需要精细调优的场景
- AdamW: 带 L2 正则化，防止过拟合

学习率调度:
- 分类任务: StepLR 或 CosineAnnealingLR
- 迁移学习: 从小学习率开始
- 长时间训练: OneCycleLR

梯度处理:
- 正常训练: 不需要梯度裁剪
- RNN/LSTM: 建议梯度裁剪
- 训练不稳定: 检查梯度爆炸

损失函数:
- 多分类: CrossEntropyLoss
- 回归: MSELoss 或 SmoothL1Loss
- 二分类: BCEWithLogitsLoss
"""

print("\n损失函数和优化器教程完成!")
