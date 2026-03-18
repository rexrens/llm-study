"""
简单的 Tokenizer，使用 tiktoken (OpenAI 的分词器)
不依赖 Hugging Face 下载
"""
import tiktoken


class SimpleTokenizer:
    """使用 tiktoken 的简单分词器"""

    def __init__(self, encoding_name="cl100k_base"):
        """
        encoding_name:
        - "cl100k_base": GPT-4/ChatGPT (100k tokens)
        - "p50k_base": GPT-3 (50k tokens)
        - "r50k_base": GPT-2 (50k tokens)
        """
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.encoding.n_vocab
        self.pad_token_id = 0
        self.eos_token_id = 0  # tiktoken 使用 0 作为特殊 token

    def encode(self, text):
        """编码文本为 token IDs"""
        return self.encoding.encode(text)

    def decode(self, token_ids):
        """解码 token IDs 为文本"""
        return self.encoding.decode(token_ids)

    def __len__(self):
        return self.vocab_size


def get_tokenizer(encoding_name="cl100k_base"):
    """获取 tokenizer 实例"""
    return SimpleTokenizer(encoding_name)


if __name__ == "__main__":
    # 测试 tokenizer
    tokenizer = get_tokenizer()
    print(f"词表大小: {tokenizer.vocab_size}")

    text = "你好，这是一个测试！"
    tokens = tokenizer.encode(text)
    print(f"原文: {text}")
    print(f"Token IDs: {tokens}")
    print(f"解码: {tokenizer.decode(tokens)}")
