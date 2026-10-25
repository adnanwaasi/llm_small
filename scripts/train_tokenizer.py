"""Train the byte-level BPE tokenizer on public + personal conversations."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from tinygpt.bpe import ByteBPETokenizer
from tinygpt.conversation import read_conversations
from tinygpt.tokenizer import save_tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", default="data/public/dialogues.jsonl")
    parser.add_argument("--personal", default="data/personal/conversations.jsonl")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--max-public-conversations", type=int, default=100_000, help="Sample size; BPE does not need everything")
    parser.add_argument("--personal-weight", type=int, default=5, help="Repeat your text so your slang gets its own tokens")
    parser.add_argument("--output", default="artifacts/tokenizer.json")
    return parser


def _texts(path: str, limit: int | None):
    if not Path(path).exists():
        return
    for conversation in itertools.islice(read_conversations(path), limit):
        for turn in conversation.turns:
            yield turn.text + "\n"


def main() -> None:
    args = build_parser().parse_args()
    personal = list(_texts(args.personal, None))
    texts = itertools.chain(_texts(args.public, args.max_public_conversations), personal * args.personal_weight)
    tokenizer = ByteBPETokenizer.train(texts, args.vocab_size, verbose=True)
    save_tokenizer(tokenizer, args.output)
    sample = "lol idk man, wanna grab food later? 😂"
    print(f"vocab_size={tokenizer.vocab_size} saved to {args.output}")
    print(f"sample: {sample!r} -> {len(tokenizer.encode(sample))} tokens")


if __name__ == "__main__":
    main()
