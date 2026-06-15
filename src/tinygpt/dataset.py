"""Dataset helpers for contiguous next-token prediction training."""

from __future__ import annotations

import torch


def split_token_stream(token_ids: list[int], train_split: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a token stream into train and validation tensors."""

    cutoff = int(len(token_ids) * train_split)
    train_ids = torch.tensor(token_ids[:cutoff], dtype=torch.long)
    val_ids = torch.tensor(token_ids[cutoff:], dtype=torch.long)
    return train_ids, val_ids


def get_batch(data: torch.Tensor, batch_size: int, context_length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of contiguous token windows."""

    if len(data) <= context_length:
        raise ValueError(
            "The tokenized corpus is too small for the chosen context length. "
            "Use more text or reduce context_length."
        )

    max_start = len(data) - context_length - 1
    positions = torch.randint(0, max_start + 1, (batch_size,))

    inputs = torch.stack([data[position : position + context_length] for position in positions])
    targets = torch.stack([data[position + 1 : position + context_length + 1] for position in positions])
    return inputs.to(device), targets.to(device)
