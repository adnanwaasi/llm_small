"""Tokenizer implementations for the tiny GPT project.

The repo started with a pure character-level tokenizer, then a word-level one.
Chat training now uses the byte-level BPE tokenizer in `tinygpt.bpe`; the older
tokenizers are kept so existing checkpoints still load.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

from tinygpt.bpe import ByteBPETokenizer


TOKEN_PATTERN = re.compile(r"\n|[^\W\d_][^\W_]*(?:['’-][^\W_]+)*|\d+|_|[^\w\s]", re.UNICODE)
PUNCTUATION_WITHOUT_LEADING_SPACE = {".", ",", "!", "?", ":", ";", ")", "]", "}", "%", "'s"}
PUNCTUATION_WITHOUT_TRAILING_SPACE = {"(", "[", "{"}


class BaseTokenizer:
    """Shared tokenizer API used by training, checkpoints, and generation."""

    tokenizer_type: str
    stoi: dict[str, int]
    itos: dict[int, str]

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @property
    def default_start_character(self) -> str:
        return self.default_start_token

    @property
    def default_start_token(self) -> str:
        for candidate in ("\n", " "):
            if candidate in self.stoi:
                return candidate
        return next(iter(self.stoi))

    @property
    def default_start_token_id(self) -> int:
        return self.stoi[self.default_start_token]


@dataclass(slots=True)
class CharacterTokenizer(BaseTokenizer):
    """Legacy character-level tokenizer."""

    stoi: dict[str, int]
    itos: dict[int, str]
    tokenizer_type: str = "character"

    @classmethod
    def from_text(cls, text: str) -> "CharacterTokenizer":
        unique_chars = sorted(set(text))
        stoi = {character: index for index, character in enumerate(unique_chars)}
        itos = {index: character for character, index in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[character] for character in text]

    def encode_with_fallback(self, text: str, fallback_character: str | None = None) -> list[int]:
        if fallback_character is None or fallback_character not in self.stoi:
            fallback_character = self.default_start_character
        fallback_id = self.stoi[fallback_character]
        return [self.stoi.get(character, fallback_id) for character in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.itos[token_id] for token_id in token_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.tokenizer_type,
            "stoi": self.stoi,
            "itos": self.itos,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "CharacterTokenizer":
        stoi_payload = payload["stoi"]
        itos_payload = payload["itos"]
        if not isinstance(stoi_payload, dict) or not isinstance(itos_payload, dict):
            raise TypeError("Invalid character tokenizer payload")
        stoi = {str(key): int(value) for key, value in stoi_payload.items()}
        itos = {int(key): str(value) for key, value in itos_payload.items()}
        return cls(stoi=stoi, itos=itos)


@dataclass(slots=True)
class WordTokenizer(BaseTokenizer):
    """Simple word-level tokenizer with punctuation/newline-aware splitting."""

    stoi: dict[str, int]
    itos: dict[int, str]
    unk_token: str = "<unk>"
    tokenizer_type: str = "word"

    @property
    def default_start_token(self) -> str:
        if "\n" in self.stoi:
            return "\n"
        for token in self.stoi:
            if token != self.unk_token:
                return token
        return self.unk_token

    @classmethod
    def from_text(cls, text: str) -> "WordTokenizer":
        tokens = tokenize_words(text)
        vocab = ["<unk>"] + sorted(set(tokens))
        stoi = {token: index for index, token in enumerate(vocab)}
        itos = {index: token for token, index in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    def encode(self, text: str) -> list[int]:
        tokens = tokenize_words(text)
        return [self.stoi[token] for token in tokens]

    def encode_with_fallback(self, text: str, fallback_character: str | None = None) -> list[int]:
        fallback_token = self.unk_token
        if fallback_character is not None and fallback_character in self.stoi:
            fallback_token = fallback_character
        fallback_id = self.stoi.get(fallback_token, 0)
        return [self.stoi.get(token, fallback_id) for token in tokenize_words(text)]

    def decode(self, token_ids: list[int]) -> str:
        tokens = [self.itos[token_id] for token_id in token_ids if self.itos[token_id] != self.unk_token]
        return detokenize_words(tokens)

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.tokenizer_type,
            "stoi": self.stoi,
            "itos": self.itos,
            "unk_token": self.unk_token,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "WordTokenizer":
        stoi_payload = payload["stoi"]
        itos_payload = payload["itos"]
        if not isinstance(stoi_payload, dict) or not isinstance(itos_payload, dict):
            raise TypeError("Invalid word tokenizer payload")
        stoi = {str(key): int(value) for key, value in stoi_payload.items()}
        itos = {int(key): str(value) for key, value in itos_payload.items()}
        unk_token = str(payload.get("unk_token", "<unk>"))
        return cls(stoi=stoi, itos=itos, unk_token=unk_token)


def tokenize_words(text: str) -> list[str]:
    """Split text into words, punctuation, and explicit newline tokens."""

    return TOKEN_PATTERN.findall(text)


def detokenize_words(tokens: list[str]) -> str:
    """Turn token strings back into a readable sentence-like string."""

    pieces: list[str] = []
    for token in tokens:
        if token == "\n":
            pieces.append("\n")
            continue

        if not pieces or pieces[-1].endswith("\n") or token in PUNCTUATION_WITHOUT_LEADING_SPACE:
            pieces.append(token)
            continue

        if pieces[-1] in PUNCTUATION_WITHOUT_TRAILING_SPACE:
            pieces.append(token)
            continue

        pieces.append(" " + token)

    return "".join(pieces)


def build_tokenizer(text: str, tokenizer_type: str) -> BaseTokenizer:
    """Build the requested tokenizer from corpus text."""

    if tokenizer_type == "character":
        return CharacterTokenizer.from_text(text)
    if tokenizer_type == "word":
        return WordTokenizer.from_text(text)
    raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")


def save_tokenizer(tokenizer: BaseTokenizer | ByteBPETokenizer, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(tokenizer.to_dict()), encoding="utf-8")


def load_tokenizer(path: str | Path) -> BaseTokenizer | ByteBPETokenizer:
    return load_tokenizer_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_tokenizer_from_dict(payload: dict[str, object]) -> BaseTokenizer | ByteBPETokenizer:
    """Load a tokenizer from checkpoint payload data."""

    tokenizer_type = str(payload.get("type", "character"))
    if tokenizer_type == "bpe":
        return ByteBPETokenizer.from_dict(payload)
    if tokenizer_type == "character":
        return CharacterTokenizer.from_dict(payload)
    if tokenizer_type == "word":
        return WordTokenizer.from_dict(payload)
    raise ValueError(f"Unsupported tokenizer payload type: {tokenizer_type}")
