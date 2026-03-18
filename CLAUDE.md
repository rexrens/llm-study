# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Chinese educational repository focused on Large Language Model (LLM) studies. It consists of two main parts:

1. **`/course/`** - A curated collection of LLM learning materials from various sources (Chinese and English)
2. **`/hand-on-llm/`** - An active PyTorch implementation of a LLaMA-2-style model (~95M parameters)

The repository serves as both a learning resource hub and a practical implementation playground for understanding LLM architecture.

## Codebase Structure

```
llm-study/
├── course/                # LLM learning materials collection
│   ├── 李宏毅大模型教程/     # Prof. Hung-yi Lee's course (Chinese)
│   ├── Datawhale-LLM-Cookbook/  # Practical LLM guide (Chinese)
│   ├── Datawhale-Happy-LLM/     # Theoretical course (Chinese)
│   ├── Coursera-GenAI-LLMs/     # Professional certification (English)
│   ├── ActiveLoop-LLM-Training/ # Production deployment (English)
│   ├── Google-LLM-Introduction/  # Quick intro (English)
│   └── Salesforce-LLM-Course/    # Business applications (English)
├── hand-on-llm/          # Active LLaMA-2 implementation
│   ├── model.py           # LLaMA model architecture
│   ├── train.py           # Training pipeline
│   ├── inference.py       # Text generation
│   ├── tokenizer.py       # Tiktoken wrapper
│   └── main.py            # Model initialization tests
└── torch_learn/           # PyTorch learning examples
```

## LLaMA Model Architecture (`hand-on-llm/`)

The `hand-on-llm/` directory contains a complete LLaMA-2 implementation for educational purposes. Key components:

### Core Components (`model.py`)
- `LLaMAConfig`: Hyperparameters (default config yields ~95M parameters)
- `RMSNorm`: Root Mean Square Normalization (no bias, no mean subtraction)
- `RotaryEmbedding` + `apply_rotary_pos_emb`: Rotational position encoding applied to Q/K
- `Attention`: Multi-head attention with GQA (Grouped Query Attention) support
- `MLP`: SwiGLU activation (gate_proj * siLU * up_proj -> down_proj)
- `LLaMABlock`: Pre-norm transformer block (RMSNorm -> Attention -> Add -> RMSNorm -> MLP -> Add)
- `LLaMA`: Full model with tied embedding/lm_head weights

### Key Design Choices
- Weight sharing between `embed_tokens` and `lm_head`
- No bias in all Linear layers
- Pre-normalization (RMSNorm before attention/MLP)
- KV cache for efficient inference in `generate()` method
- Causal attention masking

## Common Commands

### Package Management (uv)
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies from hand-on-llm/
cd hand-on-llm
uv sync

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
```

### Development Commands (`hand-on-llm/`)
```bash
# Test model initialization and forward pass
uv run python main.py

# Train model (requires data file)
uv run python train.py --data example_data.txt --batch_size 2 --epochs 5

# Run inference with trained checkpoint
uv run python inference.py --checkpoint checkpoints/checkpoint_epoch_1.pt --prompt "深度学习"
```

## Training Details (`train.py`)

- `TextDataset`: Loads text, tokenizes, creates fixed-length sequences with shifted labels (next token prediction)
- Learning rate schedule: Linear warmup + Cosine decay (via `CombinedScheduler`)
- Gradient clipping (default 1.0)
- Checkpoint saving at each epoch and best model based on eval loss

The default model config (hidden_size=512, intermediate_size=1408, num_layers=8) yields ~95M parameters.

## Important Notes

- When loading checkpoints in `inference.py`, the `LLaMAConfig` must match the training configuration exactly
- The causal attention mask is automatically generated in the model forward if not provided
- For GQA, `num_key_value_heads` divides `num_attention_heads` (default: 8/4 = 2 groups)
- All linear layers are bias-free per LLaMA design
- The repository uses Chinese comments and documentation with English code
