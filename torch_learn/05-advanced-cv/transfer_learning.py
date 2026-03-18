"""
迁移学习教程
使用预训练模型（如 ResNet, EfficientNet）进行微调，加速训练并提高性能。
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# 导入工具
import sys
sys.path.append(str(Path(__file__).parent.parent))
from common.metrics import MetricsTracker
from utils.device import get_device
from utils.seed import set_seed


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Transfer Learning")

    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "resnet50", "efficientnet_b0", "mobilenet_v3_small"],
                        help="预训练模型")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "flowers102", "custom"],
                        help="数据集")
    parser.add_argument("--data_dir", type=str, default="./data", help="数据目录")

    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")

    parser.add_argument("--freeze_backbone", action="store_true",
                        help="冻结主干网络，只训练分类头")
    parser.add_argument("--num_classes", type=int, default=10, help="类别数")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="检查点目录")

    return parser.parse_args()


def get_pretrained_model(model_name: str, num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    """
    获取预训练模型

    Args:
        model_name: 模型名称
        num_classes: 输出类别数
        freeze_backbone: 是否冻结主干网络

    Returns:
        预训练模型
    """
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(pretrained=True)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # 冻结主干网络
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "fc" not in name and "classifier" not in name:
                param.requires_grad = False
                print(f"冻结: {name}")

    return model


def get_data_transforms(model_name: str, dataset: str = "cifar10"):
    """
    获取数据变换

    预训练模型通常期望输入为 224x224 的 RGB 图像，使用 ImageNet 的均值和标准差
    """
    # ImageNet 均值和标准差
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if dataset == "cifar10":
        cifar_mean = [0.4914, 0.4822, 0.4465]
        cifar_std = [0.2470, 0.2435, 0.2616]

        train_transform = transforms.Compose([
            transforms.Resize(224),  # 调整到预训练模型期望的大小
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ])

        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ])

    elif dataset == "flowers102":
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])

    else:  # custom
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])

    return train_transform, test_transform


def get_dataset(dataset: str, data_dir: str, train_transform, test_transform):
    """获取数据集"""
    if dataset == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=test_transform
        )
        num_classes = 10

    elif dataset == "flowers102":
        train_dataset = datasets.Flowers102(
            root=data_dir, split="train", download=True, transform=train_transform
        )
        test_dataset = datasets.Flowers102(
            root=data_dir, split="test", download=True, transform=test_transform
        )
        num_classes = 102

    elif dataset == "custom":
        # 假设数据目录结构为: data_dir/train/{class}/img.jpg
        train_dataset = datasets.ImageFolder(
            root=os.path.join(data_dir, "train"),
            transform=train_transform
        )
        test_dataset = datasets.ImageFolder(
            root=os.path.join(data_dir, "val"),
            transform=test_transform
        )
        num_classes = len(train_dataset.classes)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return train_dataset, test_dataset, num_classes


def train_epoch(model, dataloader, criterion, optimizer, device):
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


def validate(model, dataloader, criterion, device):
    """验证"""
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
    """训练迁移学习模型"""
    # 设置随机种子
    set_seed(42)

    # 设置设备
    device = get_device()
    print(f"使用设备: {device}")

    # 数据准备
    train_transform, test_transform = get_data_transforms(args.model, args.dataset)
    train_dataset, test_dataset, dataset_num_classes = get_dataset(
        args.dataset, args.data_dir, train_transform, test_transform
    )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"类别数: {dataset_num_classes}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 创建模型
    num_classes = args.num_classes if args.num_classes != 10 else dataset_num_classes
    model = get_pretrained_model(args.model, num_classes, args.freeze_backbone)
    model = model.to(device)

    # 统计可训练参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    # 对不同层使用不同的学习率
    if args.freeze_backbone:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=args.learning_rate)
    else:
        # 主干网络使用较小学习率，分类头使用较大学习率
        backbone_params = []
        classifier_params = []

        for name, param in model.named_parameters():
            if "fc" in name or "classifier" in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)

        optimizer = optim.Adam([
            {"params": backbone_params, "lr": args.learning_rate * 0.1},
            {"params": classifier_params, "lr": args.learning_rate},
        ], weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 检查点目录
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 训练
    print(f"\n开始训练 ({args.num_epochs} epochs)...")
    print("-" * 60)

    best_val_acc = 0.0

    for epoch in range(args.num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{args.num_epochs} (lr={lr:.6f}): "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, checkpoint_dir / f"best_model_{args.model}.pth")
            print(f"  -> 保存最佳模型 (val_acc={val_acc:.2f}%)")

    print("-" * 60)
    print(f"训练完成! 最佳验证准确率: {best_val_acc:.2f}%")

    return model


def demo():
    """演示迁移学习的使用"""
    print("=== 迁移学习演示 ===\n")

    # 创建模型
    model = get_pretrained_model("resnet18", num_classes=10, freeze_backbone=False)

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"模型参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")

    # 测试前向传播
    x = torch.randn(2, 3, 224, 224)  # 预训练模型期望 224x224 输入
    output = model(x)
    print(f"\n输入: {x.shape} -> 输出: {output.shape}")

    # 冻结主干网络
    print("\n冻结主干网络:")
    model_frozen = get_pretrained_model("resnet18", num_classes=10, freeze_backbone=True)

    frozen_params = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
    print(f"  冻结后可训练参数: {frozen_params:,}")


if __name__ == "__main__":
    args = parse_args()

    if len(args.dataset) > 0 and args.dataset != "cifar10":
        # 运行完整训练
        train(args)
    else:
        # 运行演示
        demo()
