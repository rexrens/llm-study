"""
推理脚本：加载训练好的模型进行文本生成
"""
import argparse
import torch

from model import LLaMA, LLaMAConfig
from tokenizer import get_tokenizer


def load_model(checkpoint_path, config, device):
    """加载模型"""
    model = LLaMA(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"加载检查点: {checkpoint_path}")
    print(f"训练 Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="LLaMA 推理")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--prompt", type=str, default="你好", help="输入提示词")
    parser.add_argument("--tokenizer", type=str, default="cl100k_base", help="Tokenizer 编码名称")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k 采样")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 加载 tokenizer
    tokenizer = get_tokenizer(args.tokenizer)
    vocab_size = tokenizer.vocab_size
    print(f"词表大小: {vocab_size}")

    # 模型配置 (需要与训练时一致)
    config = LLaMAConfig(
        vocab_size=vocab_size,
        hidden_size=512,
        intermediate_size=1408,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=2048,
    )

    # 加载模型
    model = load_model(args.checkpoint, config, device)

    # Tokenize 输入
    input_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long).to(device)

    print(f"\n输入: {args.prompt}")
    print("生成中...")

    # 生成
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    # Decode 输出
    generated_text = tokenizer.decode(output_ids[0].tolist())
    print(f"\n输出: {generated_text}")


if __name__ == "__main__":
    main()
