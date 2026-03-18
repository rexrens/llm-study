"""
训练脚本
"""
import os
import argparse
from datetime import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from model import LLaMA, LLaMAConfig, count_parameters
from tokenizer import get_tokenizer


class TextDataset(Dataset):
    """简单的文本数据集"""

    def __init__(self, file_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 读取并 tokenize 文本
        print(f"加载数据: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Tokenize
        tokens = tokenizer.encode(text)
        print(f"Token数量: {len(tokens)}")

        # 切分成序列，支持小于最大长度的序列
        self.sequences = []
        for i in range(0, len(tokens), max_length):
            seq = tokens[i:i + max_length]
            # 只要序列长度 >= 2 就可以用于训练（预测下一个token）
            if len(seq) >= 2:
                self.sequences.append(seq)

        print(f"生成序列数: {len(self.sequences)}")
        if len(self.sequences) == 0:
            print("警告: 数据量太小，无法生成训练序列！")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens = torch.tensor(self.sequences[idx], dtype=torch.long)
        # 输入是 tokens[:-1]，目标是 tokens[1:] (预测下一个token)
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }


def get_attention_mask(input_ids):
    """生成因果注意力掩码"""
    batch_size, seq_len = input_ids.shape
    mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
    mask = mask.masked_fill(mask == 0, float('-inf'))
    return mask


def compute_loss(logits, labels, attention_mask=None):
    """
    计算交叉熵损失

    logits: (batch, seq_len, vocab_size)
    labels: (batch, seq_len)
    """
    # reshape: (batch * seq_len, vocab_size)
    batch_size, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)
    labels = labels.view(-1)

    # 计算 loss
    loss = nn.functional.cross_entropy(logits, labels, reduction='mean')
    return loss


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, args):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    total_tokens = 0

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # 前向传播
        outputs = model(input_ids)
        logits = outputs["logits"]

        # 计算损失
        loss = compute_loss(logits, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # 优化器步进
        optimizer.step()
        scheduler.step()

        # 统计
        total_loss += loss.item()
        total_tokens += labels.numel()

        # 打印进度
        if (step + 1) % args.log_interval == 0:
            avg_loss = total_loss / (step + 1)
            ppl = torch.exp(torch.tensor(avg_loss))
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch}, Step {step + 1}/{len(dataloader)}, "
                  f"Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}, "
                  f"PPL: {ppl:.2f}, LR: {lr:.2e}")

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, args):
    """评估模型"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids)
            logits = outputs["logits"]

            loss = compute_loss(logits, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    ppl = torch.exp(torch.tensor(avg_loss))
    return avg_loss, ppl


def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_dir):
    """保存检查点"""
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    path = Path(save_dir) / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, path)
    print(f"保存检查点: {path}")


def main():
    parser = argparse.ArgumentParser(description="训练 LLaMA 模型")
    parser.add_argument("--data", type=str, required=True, help="训练数据文件")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="输出目录")
    parser.add_argument("--tokenizer", type=str, default="cl100k_base", help="Tokenizer 编码名称 (cl100k_base, p50k_base, r50k_base)")

    # 模型配置
    parser.add_argument("--vocab_size", type=int, default=None, help="词表大小 (默认从 tokenizer 获取)")
    parser.add_argument("--hidden_size", type=int, default=512, help="隐藏层维度")
    parser.add_argument("--intermediate_size", type=int, default=1408, help="MLP 中间层维度")
    parser.add_argument("--num_layers", type=int, default=8, help="Transformer 层数")
    parser.add_argument("--num_heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--kv_heads", type=int, default=4, help="KV 头数 (GQA)")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="最大序列长度")

    # 训练配置
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--warmup_steps", type=int, default=100, help="预热步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")

    # 其他
    parser.add_argument("--log_interval", type=int, default=10, help="日志间隔")
    parser.add_argument("--eval_interval", type=int, default=1, help="评估间隔")
    parser.add_argument("--save_interval", type=int, default=1, help="保存间隔")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # 设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 加载 tokenizer
    print(f"加载 tokenizer: {args.tokenizer}")
    tokenizer = get_tokenizer(args.tokenizer)

    # 创建数据集
    dataset = TextDataset(args.data, tokenizer, max_length=args.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 使用 tokenizer 的词表大小
    vocab_size = args.vocab_size if args.vocab_size else tokenizer.vocab_size
    print(f"词表大小: {vocab_size}")

    # 创建模型
    config = LLaMAConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.kv_heads,
        max_position_embeddings=args.max_seq_len,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )
    model = LLaMA(config).to(device)

    # 打印参数量
    total_params = count_parameters(model)
    print(f"模型参数量: {total_params / 1e6:.2f}M")

    # 优化器
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 学习率调度器 (线性预热 + 余弦衰减)
    total_steps = len(dataloader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - args.warmup_steps,
        eta_min=args.lr * 0.1
    )
    # 预热
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(step / args.warmup_steps, 1.0)
    )

    # 合并调度器
    class CombinedScheduler:
        def __init__(self, warmup, main):
            self.warmup = warmup
            self.main = main
            self.warmup_steps = args.warmup_steps

        def step(self):
            if self.warmup.get_last_lr()[0] < 1.0:
                self.warmup.step()
            else:
                self.main.step()

        def get_last_lr(self):
            if self.warmup.get_last_lr()[0] < 1.0:
                return [self.warmup.get_last_lr()[0] * args.lr]
            else:
                return self.main.get_last_lr()

        def state_dict(self):
            return {
                'warmup': self.warmup.state_dict(),
                'main': self.main.state_dict(),
            }

        def load_state_dict(self, state_dict):
            self.warmup.load_state_dict(state_dict['warmup'])
            self.main.load_state_dict(state_dict['main'])

    scheduler = CombinedScheduler(warmup_scheduler, scheduler)

    # 保存配置
    config_dict = vars(args)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # 训练循环
    print(f"\n开始训练...")
    print(f"总步数: {total_steps}")

    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # 训练
        train_loss = train_epoch(model, dataloader, optimizer, scheduler, device, epoch, args)

        # 评估
        if epoch % args.eval_interval == 0:
            eval_loss, eval_ppl = evaluate(model, dataloader, device, args)
            print(f"Eval Loss: {eval_loss:.4f}, Eval PPL: {eval_ppl:.2f}")

            # 保存最佳模型
            if eval_loss < best_loss:
                best_loss = eval_loss
                save_checkpoint(model, optimizer, scheduler, epoch, eval_loss, args.output_dir)
                print(f"保存最佳模型 (Loss: {best_loss:.4f})")

        # 定期保存
        if epoch % args.save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss, args.output_dir)

    print("\n训练完成!")


if __name__ == "__main__":
    main()
