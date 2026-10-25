"""Tiny decoder-only GPT implementation built with PyTorch.

Same overall shape as GPT-2, with the small upgrades modern small LLMs
(Llama-style) use because they help most at tiny scale:

- RMSNorm instead of LayerNorm
- rotary position embeddings (RoPE) instead of a learned position table
- SwiGLU feed-forward
- tied input/output embeddings (saves ~vocab*dim params)
- `scaled_dot_product_attention` (uses fused/flash kernels when available)
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from tinygpt.config import ModelConfig
from tinygpt.dataset import IGNORE_INDEX


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def rotary_tables(head_dim: int, length: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(length).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (batch, heads, seq, head_dim); rotate pairs of channels by position-dependent angles.
    left, right = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((left * cos - right * sin, left * sin + right * cos), dim=-1)
    return rotated.flatten(-2).type_as(x)


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention for autoregressive decoding."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.embedding_dim % config.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head dimension must be even for rotary embeddings")
        self.qkv = nn.Linear(config.embedding_dim, 3 * config.embedding_dim, bias=False)
        self.projection = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels = x.shape
        qkv = self.qkv(x).view(batch_size, sequence_length, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)

        query = apply_rotary(query, cos, sin)
        key = apply_rotary(key, cos, sin)

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, channels)
        return self.projection(attended)


class FeedForward(nn.Module):
    """SwiGLU MLP block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        # 8/3 * dim keeps the parameter count equal to a 4x GELU MLP.
        hidden = int(8 * config.embedding_dim / 3)
        hidden = 64 * math.ceil(hidden / 64)
        self.gate = nn.Linear(config.embedding_dim, hidden, bias=False)
        self.up = nn.Linear(config.embedding_dim, hidden, bias=False)
        self.down = nn.Linear(hidden, config.embedding_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = RMSNorm(config.embedding_dim)
        self.attention = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.embedding_dim)
        self.feed_forward = FeedForward(config)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.residual_dropout(self.attention(self.ln_1(x), cos, sin))
        x = x + self.feed_forward(self.ln_2(x))
        return x


class TinyGPT(nn.Module):
    """A compact GPT-style language model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.embedding_dim)
        self.language_model_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.language_model_head.weight = self.token_embedding.weight
        self.dropout = nn.Dropout(config.dropout)

        cos, sin = rotary_tables(config.embedding_dim // config.num_heads, config.context_length, config.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2 style: scale residual projections down by depth for stable training.
        for name, parameter in self.named_parameters():
            if name.endswith("projection.weight") or name.endswith("down.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.config.context_length:
            raise ValueError("Input sequence is longer than model context length")

        cos = self.rope_cos[:sequence_length]
        sin = self.rope_sin[:sequence_length]
        x = self.dropout(self.token_embedding(input_ids))

        for block in self.blocks:
            x = block(x, cos, sin)

        x = self.final_norm(x)
        logits = self.language_model_head(x)

        loss: torch.Tensor | None = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().view(batch_size * sequence_length, -1),
                targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float | None = 0.9,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        stop_token_ids: set[int] | None = None,
        banned_token_ids: set[int] | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample new tokens from the model.

        Stops early (batch size 1) when a token in `stop_token_ids` is produced,
        e.g. `<|end|>` at the end of a chat reply.
        """

        if temperature <= 0:
            raise ValueError("temperature must be greater than 0")

        for _ in range(max_new_tokens):
            cropped = input_ids[:, -self.config.context_length :]
            logits, _ = self(cropped)
            next_token_logits = logits[:, -1, :].float()

            if banned_token_ids:
                next_token_logits[:, list(banned_token_ids)] = float("-inf")

            if repetition_penalty > 1.0:
                for batch_index in range(input_ids.size(0)):
                    seen = torch.unique(cropped[batch_index])
                    scores = next_token_logits[batch_index, seen]
                    next_token_logits[batch_index, seen] = torch.where(
                        scores > 0, scores / repetition_penalty, scores * repetition_penalty
                    )

            next_token_logits = next_token_logits / temperature

            if top_k is not None and top_k > 0:
                k = min(top_k, next_token_logits.size(-1))
                values, _ = torch.topk(next_token_logits, k)
                next_token_logits = next_token_logits.masked_fill(next_token_logits < values[:, [-1]], float("-inf"))

            if top_p is not None and 0 < top_p < 1:
                sorted_logits, sorted_index = torch.sort(next_token_logits, descending=True)
                cumulative = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove = cumulative - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                next_token_logits = torch.full_like(next_token_logits, float("-inf")).scatter(1, sorted_index, sorted_logits)

            if do_sample:
                probabilities = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            input_ids = torch.cat((input_ids, next_token), dim=1)
            if stop_token_ids and input_ids.size(0) == 1 and int(next_token) in stop_token_ids:
                break
        return input_ids
