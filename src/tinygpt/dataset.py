"""Dataset helpers for contiguous next-token prediction training."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

import torch

from tinygpt.bpe import ByteBPETokenizer
from tinygpt.chat_format import encode_conversation
from tinygpt.conversation import Conversation

IGNORE_INDEX = -100


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


@dataclass(slots=True)
class ChatStream:
    """All conversations concatenated into one token stream with a loss mask."""

    ids: torch.Tensor
    mask: torch.Tensor
    learn_positions: torch.Tensor

    @classmethod
    def from_conversations(
        cls,
        tokenizer: ByteBPETokenizer,
        conversations: Iterable[Conversation],
        *,
        train_on: str,
    ) -> "ChatStream":
        all_ids: list[int] = []
        all_mask: list[int] = []
        for conversation in conversations:
            ids, mask = encode_conversation(tokenizer, conversation, train_on=train_on)
            all_ids.extend(ids)
            all_mask.extend(mask)
        ids_tensor = torch.tensor(all_ids, dtype=torch.long)
        mask_tensor = torch.tensor(all_mask, dtype=torch.bool)
        return cls(ids=ids_tensor, mask=mask_tensor, learn_positions=mask_tensor.nonzero().squeeze(1))

    @property
    def num_tokens(self) -> int:
        return int(self.ids.numel())

    @property
    def num_learn_tokens(self) -> int:
        return int(self.learn_positions.numel())

    def get_batch(
        self,
        batch_size: int,
        context_length: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample windows that are guaranteed to contain trainable targets.

        A trainable position `p` is picked first and the window is placed so
        that `p` falls inside it. With sparse `me` tokens this keeps every batch
        useful instead of mostly grading nothing.
        """

        if self.num_tokens <= context_length + 1:
            raise ValueError("Not enough tokens for the chosen context length. Add data or reduce context_length.")
        if self.num_learn_tokens == 0:
            raise ValueError("No trainable tokens. Fine-tuning data needs at least one `me` turn.")

        picks = self.learn_positions[
            torch.randint(0, self.num_learn_tokens, (batch_size,), generator=generator)
        ]
        offsets = torch.randint(1, context_length + 1, (batch_size,), generator=generator)
        max_start = self.num_tokens - context_length - 1
        starts = (picks - offsets).clamp(0, max_start)

        window = torch.arange(context_length + 1)
        index = starts.unsqueeze(1) + window
        chunk = self.ids[index]
        chunk_mask = self.mask[index]
        inputs = chunk[:, :-1]
        targets = chunk[:, 1:].masked_fill(~chunk_mask[:, 1:], IGNORE_INDEX)
        return inputs.to(device), targets.to(device)


def split_conversations(
    conversations: list[Conversation],
    val_fraction: float,
    seed: int,
) -> tuple[list[Conversation], list[Conversation]]:
    """Split by whole conversation so validation measures unseen chats."""

    shuffled = list(conversations)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_fraction)) if len(shuffled) > 1 else 0
    return shuffled[val_count:], shuffled[:val_count]
