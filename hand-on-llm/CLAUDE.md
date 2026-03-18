# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch implementation of a small LLaMA-2-style model (~100M parameters) for educational purposes. The codebase implements core LLaMA 2 components: RMSNorm, RoPE (rotary position embeddings), SwiGLU activation, and GQA (grouped query attention).

This project uses [uv](https://github.com/astral-sh/uv) for package management.

## Architecture Summary

The model consists of these key components in `model.py`:

- `LLaMAConfig`: Dataclass defining model hyperparameters
- `RMSNorm`: Root Mean Square Normalization (no bias, no mean subtraction)
- `RotaryEmbedding` + `apply_rotary_pos_emb`: Rotational position encoding applied to Q/K
- `Attention`: Multi-head attention with GQA support (multiple query heads share key/value heads)
- `MLP`: SwiGLU activation (gate_proj and up_proj -> siLU * up -> down_proj)
- `LLaMABlock`: Pre-norm transformer block (RMSNorm -> Attention -> Add -> RMSNorm -> MLP -> Add)
- `LLaMA`: Full model with embedding, transformer layers, final norm, and tied lm_head/embed_tokens

Key architectural choices:
- Weight sharing between `embed_tokens` and `lm_head`
- No bias in all Linear layers
- Pre-normalization (RMSNorm before attention/MLP, not after)
- KV cache for efficient inference in `generate()` method

## Common Commands

```bash
# Install dependencies using uv (creates .venv if needed)
uv sync

# Test model initialization and forward pass
uv run python model.py

# Train model (requires data file)
uv run python train.py --data example_data.txt --batch_size 2 --epochs 5

# Run inference with trained checkpoint
uv run python inference.py --checkpoint checkpoints/checkpoint_epoch_1.pt --prompt "深度学习"
```

## Training Details

`train.py` implements:
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
