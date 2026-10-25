"""Byte-level BPE tokenizer written from scratch.

Byte-level means every possible input (emoji, Hinglish, Devanagari, typos)
can be encoded without an `<unk>` token. Merges are learned from the corpus so
frequent words and your own slang become single tokens.

Special tokens mark chat structure:

- `<|bos|>`  start of a conversation
- `<|them|>` start of a turn written by someone else
- `<|me|>`   start of a turn written by the person being imitated
- `<|end|>`  end of a turn
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import heapq
from typing import Iterable

import regex

# GPT-4 style pre-tokenization: merges never cross word/number/punctuation boundaries.
SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)
SPLIT_RE = regex.compile(SPLIT_PATTERN)
SPECIAL_TOKENS = ("<|bos|>", "<|them|>", "<|me|>", "<|end|>")


def _merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    merged: list[int] = []
    index = 0
    while index < len(ids):
        if index + 1 < len(ids) and ids[index] == pair[0] and ids[index + 1] == pair[1]:
            merged.append(new_id)
            index += 2
        else:
            merged.append(ids[index])
            index += 1
    return merged


@dataclass(slots=True)
class ByteBPETokenizer:
    """Byte-level BPE with chat special tokens."""

    merges: list[tuple[int, int]]
    special_tokens: tuple[str, ...] = SPECIAL_TOKENS
    tokenizer_type: str = "bpe"
    _ranks: dict[tuple[int, int], int] = field(default_factory=dict, repr=False)
    _vocab: dict[int, bytes] = field(default_factory=dict, repr=False)
    _cache: dict[str, list[int]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._vocab = {index: bytes([index]) for index in range(256)}
        for rank, (left, right) in enumerate(self.merges):
            self._vocab[256 + rank] = self._vocab[left] + self._vocab[right]

    @property
    def special_ids(self) -> dict[str, int]:
        offset = 256 + len(self.merges)
        return {token: offset + index for index, token in enumerate(self.special_tokens)}

    @property
    def bos_id(self) -> int:
        return self.special_ids["<|bos|>"]

    @property
    def me_id(self) -> int:
        return self.special_ids["<|me|>"]

    @property
    def them_id(self) -> int:
        return self.special_ids["<|them|>"]

    @property
    def end_id(self) -> int:
        return self.special_ids["<|end|>"]

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special_tokens)

    @property
    def default_start_token_id(self) -> int:
        return self.bos_id

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int, *, verbose: bool = False) -> "ByteBPETokenizer":
        """Learn merges until the vocabulary reaches `vocab_size`.

        Works on unique pre-tokenized chunks weighted by frequency and updates
        pair counts incrementally, so it stays fast enough in pure Python.
        """

        num_merges = vocab_size - 256 - len(SPECIAL_TOKENS)
        if num_merges < 0:
            raise ValueError("vocab_size is too small for byte-level BPE")

        chunk_counts: Counter[str] = Counter()
        for text in texts:
            chunk_counts.update(SPLIT_RE.findall(text))

        words = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
        freqs = list(chunk_counts.values())
        pair_counts: Counter[tuple[int, int]] = Counter()
        where: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
        for word_index, word in enumerate(words):
            for pair in zip(word, word[1:]):
                pair_counts[pair] += freqs[word_index]
                where[pair].add(word_index)

        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)
        merges: list[tuple[int, int]] = []

        while len(merges) < num_merges and heap:
            negative_count, pair = heapq.heappop(heap)
            # Lazy deletion: skip heap entries whose count has gone stale.
            if pair_counts.get(pair, 0) != -negative_count:
                continue
            if -negative_count < 2:
                break

            new_id = 256 + len(merges)
            merges.append(pair)
            changed: set[tuple[int, int]] = set()
            for word_index in list(where.pop(pair, ())):
                word = words[word_index]
                freq = freqs[word_index]
                for old_pair in zip(word, word[1:]):
                    pair_counts[old_pair] -= freq
                    changed.add(old_pair)
                word = _merge_pair(word, pair, new_id)
                words[word_index] = word
                for new_pair in zip(word, word[1:]):
                    pair_counts[new_pair] += freq
                    where[new_pair].add(word_index)
                    changed.add(new_pair)

            pair_counts.pop(pair, None)
            for changed_pair in changed:
                count = pair_counts.get(changed_pair, 0)
                if count > 0:
                    heapq.heappush(heap, (-count, changed_pair))
                else:
                    pair_counts.pop(changed_pair, None)

            if verbose and len(merges) % 500 == 0:
                print(f"merges={len(merges)}/{num_merges}")

        return cls(merges=merges)

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached

        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            best = min(zip(ids, ids[1:]), key=lambda pair: self._ranks.get(pair, float("inf")))
            rank = self._ranks.get(best)
            if rank is None:
                break
            ids = _merge_pair(ids, best, 256 + rank)

        if len(self._cache) < 200_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        """Encode ordinary text. Special tokens are never produced from raw text."""

        ids: list[int] = []
        for chunk in SPLIT_RE.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def encode_with_fallback(self, text: str, fallback_character: str | None = None) -> list[int]:
        return self.encode(text)

    def decode(self, token_ids: list[int], *, skip_special: bool = True) -> str:
        id_to_special = {index: token for token, index in self.special_ids.items()}
        pieces: list[bytes] = []
        for token_id in token_ids:
            if token_id in id_to_special:
                if not skip_special:
                    pieces.append(id_to_special[token_id].encode("utf-8"))
                continue
            pieces.append(self._vocab[token_id])
        return b"".join(pieces).decode("utf-8", errors="replace")

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.tokenizer_type,
            "merges": [list(pair) for pair in self.merges],
            "special_tokens": list(self.special_tokens),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ByteBPETokenizer":
        merges_payload = payload["merges"]
        if not isinstance(merges_payload, list):
            raise TypeError("Invalid BPE tokenizer payload")
        merges = [(int(left), int(right)) for left, right in merges_payload]
        special_tokens = tuple(str(token) for token in payload.get("special_tokens", SPECIAL_TOKENS))
        return cls(merges=merges, special_tokens=special_tokens)
