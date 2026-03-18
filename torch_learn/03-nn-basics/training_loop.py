"""
PyTorch 训练循环教程
完整的训练循环包括前向传播、损失计算、反向传播、参数更新等步骤。
"""

import os
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# 1. 最简单的训练循环
print("=== 1. 最简单的训练循环 ===")

# 创建虚拟数据
x_train = torch.randn(100, 10)
y_train = torch.randint(0, 2, (100,))
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 创建模型
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 2),
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练
num_epochs = 3
model.train()

for epoch in range(num_epochs):
    total_loss = 0

    for x, y in train_loader:
        # 前向传播
        outputs = model(x)
        loss = criterion(outputs, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


# 2. 带验证的训练循环
print("\n=== 2. 带验证的训练循环 ===")

# 创建验证数据
x_val = torch.randn(20, 10)
y_val = torch.randint(0, 2, (20,))
val_dataset = TensorDataset(x_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

def train_epoch(model, dataloader, criterion, optimizer):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in dataloader:
        outputs = model(x)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion):
    """验证"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# 创建新模型
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 2),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练
best_val_acc = 0.0
for epoch in range(5):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = validate(model, val_loader, criterion)

    print(f"Epoch {epoch+1}: "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")


# 3. 带学习率调度的训练循环
print("\n=== 3. 带学习率调度的训练循环 ===")

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 2),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

for epoch in range(6):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = validate(model, val_loader, criterion)
    scheduler.step()

    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch+1} (lr={current_lr:.6f}): "
          f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")


# 4. 带梯度裁剪的训练循环
print("\n=== 4. 带梯度裁剪的训练循环 ===")

def train_epoch_with_clip(model, dataloader, criterion, optimizer, max_norm=1.0):
    """带梯度裁剪的训练"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in dataloader:
        outputs = model(x)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 2),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("使用梯度裁剪训练:")
for epoch in range(3):
    train_loss, train_acc = train_epoch_with_clip(
        model, train_loader, criterion, optimizer, max_norm=1.0
    )
    print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%")


# 5. 带检查点的训练循环
print("\n=== 5. 带检查点的训练循环 ===")

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 2),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 尝试加载检查点
start_epoch = 0
checkpoint_path = checkpoint_dir / "checkpoint.pth"
if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"从 epoch {start_epoch} 恢复训练")

# 训练
num_epochs = 5
for epoch in range(start_epoch, num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%")

    # 保存检查点
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_acc": train_acc,
    }
    torch.save(checkpoint, checkpoint_path)


# 6. 带早停的训练循环
print("\n=== 6. 带早停的训练循环 ===")

class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        """
        Args:
            patience: 容忍的 epoch 数
            min_delta: 最小改善幅度
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """返回是否应该早停"""
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 2),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

early_stopping = EarlyStopping(patience=3, min_delta=0.001)

for epoch in range(10):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = validate(model, val_loader, criterion)

    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    if early_stopping(val_loss):
        print(f"早停触发于 epoch {epoch+1}")
        break


# 7. 混合精度训练 (需要 GPU)
print("\n=== 7. 混合精度训练 ===")

if torch.cuda.is_available():
    from torch.cuda.amp import autocast, GradScaler

    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2),
    ).cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()  # 梯度缩放器

    def train_amp(model, dataloader, criterion, optimizer, scaler):
        """混合精度训练"""
        model.train()
        total_loss = 0

        for x, y in dataloader:
            x, y = x.cuda(), y.cuda()

            # 使用 autocast 进行自动混合精度
            with autocast():
                outputs = model(x)
                loss = criterion(outputs, y)

            # 反向传播时使用 scaler
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    # 创建 CUDA 数据加载器
    train_dataset_cuda = TensorDataset(x_train, y_train)
    train_loader_cuda = DataLoader(train_dataset_cuda, batch_size=16, shuffle=True)

    print("使用混合精度训练:")
    for epoch in range(3):
        loss = train_amp(model, train_loader_cuda, criterion, optimizer, scaler)
        print(f"Epoch {epoch+1}: Loss: {loss:.4f}")
else:
    print("混合精度训练需要 CUDA GPU")


# 8. 完整的训练类
print("\n=== 8. 完整的训练类 ===")

class Trainer:
    """训练器类，封装所有训练逻辑"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = torch.device("cpu"),
        checkpoint_dir: str = "checkpoints",
        max_grad_norm: float = 0.0,  # 0 表示不裁剪
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(device)
        self.optimizer = optimizer or torch.optim.Adam(model.parameters())
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_grad_norm = max_grad_norm

        # 训练历史
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        # 早停
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(self):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)

            outputs = self.model(x)
            loss = self.criterion(outputs, y)

            self.optimizer.zero_grad()
            loss.backward()

            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        self.history["train_loss"].append(avg_loss)
        self.history["train_acc"].append(accuracy)

        return avg_loss, accuracy

    def validate(self):
        """验证"""
        if self.val_loader is None:
            return None, None

        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)

                outputs = self.model(x)
                loss = self.criterion(outputs, y)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total

        self.history["val_loss"].append(avg_loss)
        self.history["val_acc"].append(accuracy)

        return avg_loss, accuracy

    def save_checkpoint(self, epoch: int, filename: str = "checkpoint.pth"):
        """保存检查点"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        return filepath

    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint

    def train(self, num_epochs: int, save_every: int = 1, early_stop_patience: int = 5):
        """
        训练模型

        Args:
            num_epochs: 训练 epoch 数
            save_every: 每隔多少 epoch 保存检查点
            early_stop_patience: 早停容忍的 epoch 数，0 表示不早停
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(num_epochs):
            # 训练
            train_loss, train_acc = self.train_epoch()

            # 验证
            if self.val_loader is not None:
                val_loss, val_acc = self.validate()
                val_str = f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"

                # 早停检查
                if early_stop_patience > 0:
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                        self.save_checkpoint(epoch, "best_model.pth")
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= early_stop_patience:
                            print(f"早停触发于 epoch {epoch+1}")
                            break
            else:
                val_str = ""

            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()

            # 打印
            lr = self.optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1}/{num_epochs} (lr={lr:.6f}): "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | {val_str}")

            # 保存检查点
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch, f"checkpoint_epoch_{epoch+1}.pth")

        return self.history


# 使用 Trainer
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 2),
)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    scheduler=torch.optim.lr_scheduler.StepLR(
        torch.optim.Adam(model.parameters(), lr=0.001),
        step_size=3,
        gamma=0.1
    ),
    checkpoint_dir="checkpoints",
)

print("使用 Trainer 类训练:")
history = trainer.train(num_epochs=5, save_every=2, early_stop_patience=0)

print(f"\n训练历史:")
print(f"训练准确率: {[f'{acc:.2f}%' for acc in history['train_acc']]}")
print(f"验证准确率: {[f'{acc:.2f}%' for acc in history['val_acc']]}")

print("\n训练循环教程完成!")
