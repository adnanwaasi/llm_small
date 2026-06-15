"""Tiny decoder-only GPT implementation built with PyTorch."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from tinygpt.config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention for autoregressive decoding."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.embedding_dim % config.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.query = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.key = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.value = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.projection = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.context_length, config.context_length)).view(
                1, 1, config.context_length, config.context_length
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels = x.shape

        query = self.query(x).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(x).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(x).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_scores = attention_scores.masked_fill(self.mask[:, :, :sequence_length, :sequence_length] == 0, float("-inf"))
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attended = attention_weights @ value
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, channels)
        return self.projection(attended)


class FeedForward(nn.Module):
    """Transformer MLP block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.embedding_dim, 4 * config.embedding_dim),
            nn.GELU(),
            nn.Linear(4 * config.embedding_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embedding_dim)
        self.attention = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.embedding_dim)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.feed_forward(self.ln_2(x))
        return x


class TinyGPT(nn.Module):
    """A compact GPT-style language model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.language_model_head = nn.Linear(config.embedding_dim, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.config.context_length:
            raise ValueError("Input sequence is longer than model context length")

        positions = torch.arange(sequence_length, device=input_ids.device)
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(positions)
        x = self.dropout(token_embeddings + position_embeddings)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.language_model_head(x)

        loss: torch.Tensor | None = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(batch_size * sequence_length, -1), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 0.8,
        top_k: int | None = 20,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """Autoregressively sample new tokens from the model.

        The extra controls help tiny models trained on small corpora stay more
        readable by avoiding low-probability garbage tokens and endless repeats.
        """

        for _ in range(max_new_tokens):
            cropped = input_ids[:, -self.config.context_length :]
            logits, _ = self(cropped)
            next_token_logits = logits[:, -1, :]

            if repetition_penalty > 1.0:
                for batch_index in range(input_ids.size(0)):
                    unique_tokens = torch.unique(input_ids[batch_index])
                    next_token_logits[batch_index, unique_tokens] /= repetition_penalty

            if temperature <= 0:
                raise ValueError("temperature must be greater than 0")
            next_token_logits = next_token_logits / temperature

            if top_k is not None and top_k > 0:
                top_k = min(top_k, next_token_logits.size(-1))
                values, _ = torch.topk(next_token_logits, top_k)
                cutoff = values[:, [-1]]
                next_token_logits = next_token_logits.masked_fill(next_token_logits < cutoff, float("-inf"))

            if do_sample:
                probabilities = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            input_ids = torch.cat((input_ids, next_token), dim=1)
        return input_ids
