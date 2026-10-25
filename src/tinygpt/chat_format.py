"""Turn conversations into token ids plus a loss mask.

Layout of one conversation:

    <|bos|> <|them|> hey wyd <|end|> <|me|> nothing much u? <|end|> ...

The loss mask decides which tokens the model is graded on:

- `train_on="all"`: every token (pretraining on public dialogues)
- `train_on="me"`: only the text + `<|end|>` of `me` turns (fine-tuning on
  you). The model still *reads* the other person's messages as context, but it
  is only ever taught to produce your replies.
"""

from __future__ import annotations

from tinygpt.bpe import ByteBPETokenizer
from tinygpt.conversation import Conversation, Turn


def encode_conversation(
    tokenizer: ByteBPETokenizer,
    conversation: Conversation,
    *,
    train_on: str = "me",
) -> tuple[list[int], list[int]]:
    if train_on not in {"me", "all"}:
        raise ValueError("train_on must be 'me' or 'all'")

    ids = [tokenizer.bos_id]
    mask = [0]
    for turn in conversation.turns:
        role_id = tokenizer.me_id if turn.role == "me" else tokenizer.them_id
        body = tokenizer.encode(turn.text) + [tokenizer.end_id]
        learn = train_on == "all" or turn.role == "me"
        ids.append(role_id)
        mask.append(1 if train_on == "all" else 0)
        ids.extend(body)
        mask.extend([int(learn)] * len(body))
    return ids, mask


def build_reply_prompt(tokenizer: ByteBPETokenizer, history: list[Turn]) -> list[int]:
    """Encode chat history and open a `me` turn for the model to complete."""

    ids, _ = encode_conversation(tokenizer, Conversation(turns=history), train_on="all")
    return ids + [tokenizer.me_id]
