"""
PyTorch 神经网络构建教程
nn.Module 是 PyTorch 中构建神经网络的核心基类。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 基本的 nn.Module
print("=== 1. 基本的 nn.Module ===")

class SimpleNet(nn.Module):
    """最简单的神经网络"""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        # 定义层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 创建模型
model = SimpleNet(input_size=10, hidden_size=20, output_size=2)
print(f"模型:\n{model}")

# 前向传播
x = torch.randn(4, 10)  # batch_size=4, input_dim=10
output = model(x)
print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")


# 2. 使用 nn.Sequential 构建模型
print("\n=== 2. 使用 nn.Sequential ===")

# Sequential 按顺序执行层
model_seq = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 2),
)

output = model_seq(x)
print(f"Sequential 输出形状: {output.shape}")


# 3. 更复杂的模型结构
print("\n=== 3. 复杂模型结构 ===")

class MLP(nn.Module):
    """多层感知机"""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        dropout_prob: float = 0.1,
    ):
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


model = MLP(input_size=10, hidden_sizes=[32, 16], output_size=2)
print(f"MLP:\n{model}")


# 4. 参数初始化
print("\n=== 4. 参数初始化 ===")

# 查看模型参数
for name, param in model.named_parameters():
    print(f"{name}: shape={param.shape}, requires_grad={param.requires_grad}")

# 自定义初始化
def init_weights(m: nn.Module):
    """自定义初始化函数"""
    if isinstance(m, nn.Linear):
        # Xavier/Glorot 初始化
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# 应用初始化
model.apply(init_weights)
print("\n应用自定义初始化后:")
print(f"第一层权重均值: {model.layers[0].weight.mean().item():.4f}")
print(f"第一层偏置均值: {model.layers[0].bias.mean().item():.4f}")


# 5. 常用层类型
print("\n=== 5. 常用层类型 ===")

# 全连接层
fc = nn.Linear(10, 5)
print(f"Linear: input=10, output=5")

# 卷积层
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
print(f"Conv2d: 3->16 channels, kernel=3x3")

# Batch Normalization
bn = nn.BatchNorm1d(num_features=10)
print(f"BatchNorm1d: 10 features")

# Dropout
dropout = nn.Dropout(p=0.5)
print(f"Dropout: p=0.5")

# RNN/LSTM/GRU
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
print(f"LSTM: input=10, hidden=20, layers=2")


# 6. 残差连接
print("\n=== 6. 残差连接 ===")

class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """残差连接: y = f(x) + x"""
        residual = x

        out = self.bn1(x)
        out = F.relu(out)
        out = self.fc1(out)

        out = self.bn2(out)
        out = F.relu(out)
        out = self.fc2(out)

        out += residual  # 残差连接
        return out


class ResNet(nn.Module):
    """简单的残差网络"""

    def __init__(self, input_size: int, hidden_size: int, num_blocks: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(num_blocks)
        ])
        self.output_proj = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_proj(x)
        return x


model = ResNet(input_size=10, hidden_size=20)
print(f"残差网络:\n{model}")


# 7. 共享权重
print("\n=== 7. 共享权重 ===")

class WeightSharing(nn.Module):
    """共享权重的层"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        # 创建一个共享的层
        self.shared_layer = nn.Linear(input_size, hidden_size)
        # 多次使用这个层
        self.layers = nn.ModuleList([self.shared_layer for _ in range(3)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for layer in self.layers:
            outputs.append(layer(x))
        return torch.stack(outputs, dim=0)  # (num_layers, batch, hidden)


model = WeightSharing(input_size=10, hidden_size=5)
# 所有 layers 指向同一个共享层
print(f"层 0 和层 1 是同一个对象: {model.layers[0] is model.layers[1]}")


# 8. 条件分支
print("\n=== 8. 条件分支 ===")

class ConditionalNet(nn.Module):
    """带条件分支的网络"""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 两个输出层，根据条件选择
        self.fc2a = nn.Linear(hidden_size, output_size)
        self.fc2b = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, use_branch_a: bool = True) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        if use_branch_a:
            return self.fc2a(x)
        else:
            return self.fc2b(x)


model = ConditionalNet(input_size=10, hidden_size=20, output_size=2)
x = torch.randn(4, 10)
output_a = model(x, use_branch_a=True)
output_b = model(x, use_branch_a=False)
print(f"分支 A 输出: {output_a.shape}")
print(f"分支 B 输出: {output_b.shape}")


# 9. 模型参数统计
print("\n=== 9. 参数统计 ===")

def count_parameters(model: nn.Module) -> dict:
    """统计模型参数"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
    }


params = count_parameters(model)
print(f"总参数量: {params['total']:,}")
print(f"可训练参数: {params['trainable']:,}")
print(f"不可训练参数: {params['non_trainable']:,}")


# 10. 冻结部分参数
print("\n=== 10. 冻结参数 ===")

# 冻结模型的前几层
for name, param in model.named_parameters():
    if "fc1" in name:  # 冻结 fc1 层
        param.requires_grad = False

params = count_parameters(model)
print(f"冻结后可训练参数: {params['trainable']:,}")


# 11. 模型的保存和加载
print("\n=== 11. 模型保存和加载 ===")

# 创建模型
model = MLP(input_size=10, hidden_sizes=[32, 16], output_size=2)

# 只保存模型参数
torch.save(model.state_dict(), "model_weights.pth")
print("保存模型参数到 model_weights.pth")

# 加载模型参数
loaded_model = MLP(input_size=10, hidden_sizes=[32, 16], output_size=2)
loaded_model.load_state_dict(torch.load("model_weights.pth"))
print("加载模型参数")

# 保存整个模型 (不推荐，有版本兼容性问题)
torch.save(model, "full_model.pth")
print("保存完整模型到 full_model.pth")


# 12. 模型推理模式
print("\n=== 12. 训练/评估模式 ===")

model = MLP(input_size=10, hidden_sizes=[32, 16], output_size=2, dropout_prob=0.5)

# 训练模式 (启用 dropout 等)
model.train()
print(f"训练模式: Dropout 层处于激活状态")

# 评估模式 (禁用 dropout, 启用 batch norm 的运行统计)
model.eval()
print(f"评估模式: Dropout 层被禁用")


# 13. 实战: 分类模型
print("\n=== 13. 实战: 分类模型 ===")

class Classifier(nn.Module):
    """标准的分类模型"""

    def __init__(
        self,
        num_classes: int = 10,
        hidden_size: int = 128,
        dropout_prob: float = 0.3,
    ):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(784, hidden_size),  # 假设输入是 28x28 图像展平
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前向传播
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """预测类别"""
        self.eval()
        with torch.no_grad():
            logits = self(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions


# 创建模型
model = Classifier(num_classes=10)

# 测试
x = torch.randn(4, 784)  # batch_size=4
output = model(x)
predictions = model.predict(x)

print(f"模型输出 (logits) 形状: {output.shape}")
print(f"预测类别: {predictions.tolist()}")


# 14. 打印模型结构
print("\n=== 14. 模型结构 ===")

# 使用 print
print("使用 print:")
print(model)

# 使用 summary (需要 torchsummary, 这里用简单的替代)
print("\n逐层信息:")
for name, module in model.named_children():
    print(f"  {name}: {module}")

print("\n模型构建教程完成!")
