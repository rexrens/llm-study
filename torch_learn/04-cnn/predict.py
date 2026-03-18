"""
CNN 推理脚本 - 使用训练好的模型进行预测
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 导入模型
from lenet import LeNet5, LeNet5Modern
from resnet import SmallResNet

# 导入工具
import sys
sys.path.append(str(Path(__file__).parent.parent))
from common.metrics import plot_confusion_matrix
from utils.device import get_device


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Predict with trained CNN")

    # 模型参数
    parser.add_argument("--model", type=str, default="lenet", choices=["lenet", "lenet_modern", "resnet"],
                        help="模型类型")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")

    # 数据参数
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"],
                        help="数据集选择")
    parser.add_argument("--data_dir", type=str, default="./data", help="数据目录")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")

    # 输出参数
    parser.add_argument("--num_samples", type=int, default=10, help="展示预测结果的样本数")
    parser.add_argument("--save_cm", action="store_true", help="保存混淆矩阵")

    return parser.parse_args()


def get_model(model_name: str, num_classes: int = 10, in_channels: int = 1):
    """获取模型"""
    if model_name == "lenet":
        return LeNet5(num_classes=num_classes)
    elif model_name == "lenet_modern":
        return LeNet5Modern(num_classes=num_classes)
    elif model_name == "resnet":
        return SmallResNet(num_classes=num_classes, in_channels=in_channels)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device):
    """加载模型检查点"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if "epoch" in checkpoint:
        print(f"加载检查点: epoch={checkpoint['epoch']}")
    if "val_acc" in checkpoint:
        print(f"验证准确率: {checkpoint['val_acc']:.2f}%")

    return model


def denormalize(tensor, mean, std):
    """反归一化图像"""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


def predict(args):
    """使用训练好的模型进行预测"""
    # 设置设备
    device = get_device()
    print(f"使用设备: {device}")

    # 数据准备
    if args.dataset == "mnist":
        mean, std = (0.1307,), (0.3081,)
        in_channels = 1
        class_names = [str(i) for i in range(10)]

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        test_dataset = datasets.MNIST(
            root=args.data_dir,
            train=False,
            download=True,
            transform=test_transform
        )
    else:  # cifar10
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        in_channels = 3
        class_names = ["plane", "car", "bird", "cat", "deer",
                      "dog", "frog", "horse", "ship", "truck"]

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        test_dataset = datasets.CIFAR10(
            root=args.data_dir,
            train=False,
            download=True,
            transform=test_transform
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print(f"测试集大小: {len(test_dataset)}")

    # 加载模型
    model = get_model(args.model, num_classes=10, in_channels=in_channels).to(device)
    model = load_checkpoint(model, args.checkpoint, device)
    model.eval()

    # 预测
    print("\n开始预测...")
    all_predictions = []
    all_labels = []
    all_images = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_images.append(images.cpu())

    # 计算准确率
    correct = sum(p == l for p, l in zip(all_predictions, all_labels))
    accuracy = 100.0 * correct / len(all_predictions)
    print(f"测试准确率: {accuracy:.2f}%")

    # 展示一些预测结果
    print(f"\n展示前 {args.num_samples} 个预测结果:")
    print("-" * 40)

    all_images = torch.cat(all_images, dim=0)

    fig, axes = plt.subplots(1, args.num_samples, figsize=(15, 2))

    for i in range(min(args.num_samples, len(all_images))):
        image = all_images[i]

        # 反归一化用于显示
        if in_channels == 1:
            image = image * std[0] + mean[0]
            image = image.squeeze()
            axes[i].imshow(image, cmap="gray")
        else:
            image = denormalize(image, mean, std)
            image = image.permute(1, 2, 0)
            axes[i].imshow(image.clamp(0, 1))

        label = all_labels[i]
        pred = all_predictions[i]

        color = "green" if pred == label else "red"
        axes[i].set_title(f"True: {class_names[label]}\nPred: {class_names[pred]}",
                          color=color, fontsize=8)
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("predictions.png", dpi=150, bbox_inches="tight")
    print(f"预测结果已保存到: predictions.png")
    plt.close()

    # 绘制混淆矩阵
    if args.save_cm:
        print("\n生成混淆矩阵...")
        fig = plot_confusion_matrix(
            torch.tensor(all_labels),
            torch.tensor(all_predictions),
            class_names=class_names,
            save_path="confusion_matrix.png"
        )
        print(f"混淆矩阵已保存到: confusion_matrix.png")
        plt.close()

    return all_predictions, all_labels


if __name__ == "__main__":
    args = parse_args()
    print("预测配置:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    predictions, labels = predict(args)
