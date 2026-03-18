"""
简化版 LLaMA 2 模型实现
参数量约为 100M
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LLaMAConfig:
    """模型配置"""
    vocab_size: int = 32000 # 词汇表大小
    hidden_size: int = 512 # 模型维度
    intermediate_size: int = 1408  # 通常为 hidden_size * 2.67
    num_hidden_layers: int = 8
    num_attention_heads: int = 8 # 注意力机制的头数
    num_key_value_heads: int = 4  # GQA (Grouped Query Attention)
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    def __post_init__(self):
        # 计算参数量
        # Embedding: vocab_size * hidden_size
        # Layer (per): 4 * hidden_size^2 + 3 * hidden_size * intermediate_size
        # Output: vocab_size * hidden_size (share with embedding)
        embedding_params = self.vocab_size * self.hidden_size
        layer_params = self.num_hidden_layers * (
            4 * self.hidden_size ** 2 +
            3 * self.hidden_size * self.intermediate_size
        )
        total_params = embedding_params + layer_params
        print(f"预估参数量: {total_params / 1e6:.1f}M")


class RMSNorm(nn.Module):
    """RMS Normalization"""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        # hidden_states: (batch, seq_len, hidden_size)
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(hidden_states.dtype)


def rotate_half(x):
    """将 x 旋转一半维度，用于 RoPE"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """应用旋转位置编码"""
    # q, k: (batch, num_heads, seq_len, head_dim)
    # cos, sin 可能是 (max_len, head_dim) 或 (batch, seq_len, head_dim)
    if cos.dim() == 2:
        # 完整的 cos/sin，需要用 position_ids 索引
        cos = cos[position_ids].unsqueeze(1)  # (batch, 1, seq_len, head_dim)
        sin = sin[position_ids].unsqueeze(1)
    else:
        # 已经索引过的 cos/sin，形状为 (batch, seq_len, head_dim)
        cos = cos.unsqueeze(1)  # (batch, 1, seq_len, head_dim)
        sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """旋转位置编码"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # 预计算 cos 和 sin
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # 预计算缓存
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=inv_freq.device,
            dtype=torch.float32,
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device).type_as(self.inv_freq)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, seq_len, device, dtype, position_ids=None):
        if position_ids is not None:
            # 支持 KV cache 的任意位置
            max_pos = position_ids.max().item() + 1
            if max_pos > self.max_seq_len_cached:
                self._set_cos_sin_cache(max_pos, device, dtype)
            return (
                self.cos_cached[position_ids].to(dtype),
                self.sin_cached[position_ids].to(dtype),
            )
        else:
            # 常规训练模式
            if seq_len > self.max_seq_len_cached:
                self._set_cos_sin_cache(seq_len, device, dtype)
            return (
                self.cos_cached[:seq_len].to(dtype),
                self.sin_cached[:seq_len].to(dtype),
            )


class Attention(nn.Module):
    """多头注意力机制，支持 GQA"""

    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        # 根据是否指定n_kv_heads，确定用于键（key）和值（value）的头的数量。
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # Q, K, V 投影
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
    ):
        batch_size, seq_len, _ = hidden_states.shape
        query_seq_len = seq_len  # 记录原始输入长度

        # 投影到 Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # reshape: (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 应用旋转位置编码
        cos, sin = self.rotary_emb(seq_len, key_states.device, query_states.dtype, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # KV Cache (推理时使用)
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
            seq_len = key_states.shape[2]

        past_key_value = (key_states, value_states) if use_cache else None

        # 处理 GQA: 复制 K 和 V
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # 计算注意力分数
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # 应用 attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # 加权求和
        attn_output = torch.matmul(attn_weights, value_states)

        # reshape 回原始维度
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, query_seq_len, self.hidden_size)

        # 输出投影
        output = self.o_proj(attn_output)

        return output, attn_weights, past_key_value


class MLP(nn.Module):
    """SwiGLU 激活函数的 MLP"""

    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class LLaMABlock(nn.Module):
    """LLaMA 的 Transformer Block"""

    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.attention = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
    ):
        # Pre-attention RMSNorm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention
        hidden_states, attn_weights, past_key_value = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Pre-MLP RMSNorm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states, attn_weights, past_key_value)


class LLaMA(nn.Module):
    """完整的 LLaMA 模型"""

    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config

        # Token embedding 和 positional embedding (LLaMA 使用 RoPE，所以不需要单独的 pos embedding)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList([
            LLaMABlock(config) for _ in range(config.num_hidden_layers)
        ])

        # Final RMSNorm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Lm head (与 embedding 权重共享)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 标准差 = sqrt(2 / (hidden_size + 输出维度)) - Glorot/Xavier
                std = 0.02
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                if hasattr(module, "bias") and module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # lm_head 与 embed_tokens 共享权重
        self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
    ):
        batch_size, seq_len = input_ids.shape

        # Position IDs
        if position_ids is None:
            past_len = past_key_values[0][0].shape[2] if (past_key_values and past_key_values[0][0] is not None) else 0
            position_ids = torch.arange(past_len, past_len + seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Attention mask (causal mask)
        if attention_mask is None:
            # 当使用 past_key_values 时，需要考虑整个历史序列长度
            past_len = past_key_values[0][0].shape[2] if (past_key_values and past_key_values[0][0] is not None) else 0
            total_seq_len = past_len + seq_len

            # 创建 causal mask，形状为 (seq_len, total_seq_len)
            attention_mask = torch.ones((seq_len, total_seq_len), device=input_ids.device)
            attention_mask = torch.tril(attention_mask, diagonal=past_len)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))

        # Token embedding
        hidden_states = self.embed_tokens(input_ids)

        # 处理 past_key_values (用于推理加速)
        if past_key_values is not None:
            past_key_values = [
                list(past_kv) if past_kv is not None else [None, None]
                for past_kv in past_key_values
            ]
        else:
            past_key_values = [None] * self.config.num_hidden_layers

        # 通过所有 Transformer blocks
        all_hidden_states = []
        all_self_attns = []
        next_decoder_cache = []

        for i, layer in enumerate(self.layers):
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[i],
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]
            next_decoder_cache.append(layer_outputs[2])

        # Final norm
        hidden_states = self.norm(hidden_states)

        # LM head
        logits = self.lm_head(hidden_states)

        loss = None

        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "past_key_values": next_decoder_cache if use_cache else None,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=100,
        temperature=1.0,
        top_k=None,
        eos_token_id=None,
    ):
        """生成文本"""
        self.eval()
        batch_size = input_ids.shape[0]

        # 初始化 past_key_values
        past_key_values = None

        for _ in range(max_new_tokens):
            # 前向传播
            outputs = self(input_ids, use_cache=True, past_key_values=past_key_values)
            logits = outputs["logits"]
            past_key_values = outputs["past_key_values"]

            # 只取最后一个 token 的 logits
            logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 检查 EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            # 拼接到输入
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # 测试模型
    config = LLaMAConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1408,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )

    model = LLaMA(config)
    total_params = count_parameters(model)
    print(f"实际参数量: {total_params / 1e6:.2f}M")

    # 测试前向传播
    input_ids = torch.randint(0, config.vocab_size, (2, 32))
    outputs = model(input_ids)
    print(f"Logits shape: {outputs['logits'].shape}")

    # 测试生成
    generated = model.generate(input_ids, max_new_tokens=10)
    print(f"Generated shape: {generated.shape}")


if __name__ == "__main__":
    config = LLaMAConfig()
    norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
    x = torch.randn(1, 50, config.hidden_size)
    output = norm(x)
    print(output.shape)