# 小型 LLM 训练项目

基于 PyTorch 实现的小型 LLM，参考 LLaMA 2 架构，参数量约 75M（使用 cl100k_base tokenizer 时），适合学习和研究。

## 模型架构

本项目实现了 LLaMA 2 的核心组件：

1. **RMS Normalization** - 替代 Layer Normalization
2. **RoPE (Rotary Position Embeddings)** - 旋转位置编码
3. **SwiGLU** - MLP 的激活函数
4. **GQA (Grouped Query Attention)** - 分组查询注意力
5. **KV Cache** - 推理时加速生成

### 默认配置

| 参数 | 值 |
|------|-----|
| 词汇表大小 | 100,277 (cl100k_base) |
| 隐藏层维度 | 512 |
| MLP 中间层 | 1,408 |
| Transformer 层数 | 8 |
| 注意力头数 | 8 |
| KV 头数 | 4 |
| 最大序列长度 | 2,048 |

预估参数量: ~75M

## 安装

使用 [uv](https://github.com/astral-sh/uv) 进行项目管理：

```bash
# 安装 uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步依赖（首次运行会创建虚拟环境）
uv sync

# 激活虚拟环境
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate  # Windows
```

## 快速开始

### 1. 测试模型

```bash
uv run model.py
```

### 2. 准备训练数据

创建一个文本文件，包含你想训练的文本数据（建议至少几万 tokens）：

```bash
# 使用示例数据
echo "这里是你的训练文本..." > data.txt
```

### 3. 训练模型

```bash
uv run train.py --data example_data.txt --batch_size 2 --epochs 5
```

### 4. 推理生成

```bash
uv run inference.py --checkpoint checkpoints/checkpoint_epoch_5.pt --prompt "深度学习是"
```

### 主要训练参数

```
--data              训练数据文件路径
--tokenizer         Tokenizer 编码名称 (cl100k_base, p50k_base, r50k_base)
--batch_size        批次大小 (默认: 4)
--epochs            训练轮数 (默认: 10)
--lr                学习率 (默认: 1e-3)
--max_seq_len       最大序列长度 (默认: 2048)
--device            设备 (默认: cuda/cpu)
```

### Tokenizer 选项

| 编码名称 | 说明 | 词表大小 |
|---------|------|---------|
| cl100k_base | GPT-4/ChatGPT (默认) | 100,277 |
| p50k_base | GPT-3 | 50,257 |
| r50k_base | GPT-2 | 50,257 |

## 代码结构

```
ai-llm/
├── model.py          # 模型定义 (LLaMA 2 架构)
├── tokenizer.py      # 分词器 (基于 tiktoken)
├── train.py          # 训练脚本
├── inference.py      # 推理/生成脚本
├── example_data.txt  # 示例训练数据
├── checkpoints/     # 模型检查点目录
├── pyproject.toml    # uv 项目配置
├── CLAUDE.md         # Claude Code 指南
└── README.md         # 说明文档
```

## 模型组件说明

### RMSNorm
```python
# Root Mean Square Normalization
output = input / sqrt(mean(input^2) + eps) * weight
```
无偏置、无均值减法，计算效率更高。

### RoPE
通过旋转编码将相对位置信息注入到 Query 和 Key 中，无需单独的位置嵌入。

### SwiGLU
```python
output = down(siLU(gate(x)) * up(x))
```
包含门控机制，性能优于普通 ReLU。

### GQA
减少 KV cache 的内存占用，多个 Query 头共享 Key-Value 头。

### KV Cache
推理时缓存历史 Key 和 Value，避免重复计算，大幅提升生成速度。

## 训练技巧

- 使用梯度裁剪防止梯度爆炸
- 学习率预热 + 余弦衰减
- 权重共享 (Embedding 和 LM Head)
- 因果注意力掩码 (Causal Masking)
- 支持 CPU 版本 PyTorch

## 注意事项

- 示例数据集（267 tokens）仅用于演示训练流程，无法训练出有意义的模型
- 实际训练需要准备足够的数据（建议至少几万 tokens）
- 词表大小影响模型参数量，cl100k_base 约 75M，p50k_base 约 40M

## 下一步

- 添加更多数据预处理选项
- 实现 FSDP/DDP 分布式训练
- 添加混合精度训练 (AMP)
- 实现 Flash Attention 优化
