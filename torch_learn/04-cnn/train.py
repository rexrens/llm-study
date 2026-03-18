"""
CNN 训练脚本 - MNIST 手写数字识别
使用 LeNet-5 和 ResNet 训练 MNIST 数据集
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 导入模型
from lenet import LeNet5, LeNet5Modern
from resnet import SmallResNet

# 导入工具
import sys
sys.path.append(str(Path(__file__).parent.parent))
from common.metrics import MetricsTracker, plot_training_curves
from utils.device import get_device
from utils.seed import set_seed


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Train CNN on MNIST")

    # 数据参数
    parser.add_argument("--data_dir", type=str, default="./data", help="数据目录")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"],
                        help="数据集选择")

    # 模型参数
    parser.add_argument("--model", type=str, default="lenet", choices=["lenet", "lenet_modern", "resnet"],
                        help="模型选择")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="权重衰减")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # 其他参数
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载进程数")
    parser.add_argument("--pin_memory", action="store_true", help="使用固定内存")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="检查点目录")
    parser.add_argument("--save_every", type=int, default=1, help="每隔多少 epoch 保存")

    return parser.parse_args()


def get_data_transforms(dataset: str):
    """获取数据变换"""
    if dataset == "mnist":
        # MNIST: 1x28x28 灰度图
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 均值和标准差
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        return train_transform, test_transform

    elif dataset == "cifar10":
        # CIFAR-10: 3x32x32 RGB 图像
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        return train_transform, test_transform


def get_model(model_name: str, num_classes: int = 10, in_channels: int = 1):
    """获取模型"""
    if model_name == "lenet":
        return LeNet5(num_classes=num_classes)
    elif model_name == "lenet_modern":
        return LeNet5Modern(num_classes=num_classes, dropout_prob=0.25)
    elif model_name == "resnet":
        return SmallResNet(num_classes=num_classes, in_channels=in_channels)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, device: torch.device) -> tuple[float, float]:
    """训练一个 epoch"""
    model.train()
    tracker = MetricsTracker()

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        _, predicted = torch.max(outputs, 1)
        tracker.update(loss.item(), (predicted == labels).sum().item(), labels.size(0))

    return tracker.avg_loss, tracker.accuracy


def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
            device: torch.device) -> tuple[float, float]:
    """验证模型"""
    model.eval()
    tracker = MetricsTracker()

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            tracker.update(loss.item(), (predicted == labels).sum().item(), labels.size(0))

    return tracker.avg_loss, tracker.accuracy


def train(args):
    """训练模型"""
    # 设置随机种子
    set_seed(args.seed)

    # 设置设备
    device = get_device()
    print(f"使用设备: {device}")

    # 数据准备
    train_transform, test_transform = get_data_transforms(args.dataset)

    if args.dataset == "mnist":
        train_dataset = datasets.MNIST(
            root=args.data_dir,
            train=True,
            download=True,
            transform=train_transform
        )
        test_dataset = datasets.MNIST(
            root=args.data_dir,
            train=False,
            download=True,
            transform=test_transform
        )
        in_channels = 1
    else:  # cifar10
        train_dataset = datasets.CIFAR10(
            root=args.data_dir,
            train=True,
            download=True,
            transform=train_transform
        )
        test_dataset = datasets.CIFAR10(
            root=args.data_dir,
            train=False,
            download=True,
            transform=test_transform
        )
        in_channels = 3

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    print(f"训练集大小: {len(train_dataset)}, 批次: {len(train_loader)}")
    print(f"测试集大小: {len(test_dataset)}, 批次: {len(test_loader)}")

    # 创建模型
    model = get_model(args.model, num_classes=10, in_channels=in_channels).to(device)
    print(f"\n模型: {args.model}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 训练历史
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # 检查点目录
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 训练循环
    best_val_acc = 0.0

    print(f"\n开始训练 ({args.num_epochs} epochs)...")
    print("-" * 60)

    for epoch in range(args.num_epochs):
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # 验证
        val_loss, val_acc = validate(model, test_loader, criterion, device)

        # 更新学习率
        scheduler.step()

        # 记录历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 打印结果
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{args.num_epochs} (lr={lr:.6f}): "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, checkpoint_dir / "best_model.pth")
            print(f"  -> 保存最佳模型 (val_acc={val_acc:.2f}%)")

        # 定期保存检查点
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth")

    print("-" * 60)
    print(f"训练完成! 最佳验证准确率: {best_val_acc:.2f}%")

    # 绘制训练曲线
    fig = plot_training_curves(
        history["train_loss"],
        history["val_loss"],
        save_path=checkpoint_dir / "training_curves.png"
    )
    plt.close(fig)
    print(f"训练曲线已保存到: {checkpoint_dir / 'training_curves.png'}")

    return model, history


if __name__ == "__main__":
    args = parse_args()
    print("训练配置:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    model, history = train(args)
