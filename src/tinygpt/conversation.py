"""Shared conversation format used by every data source.

Everything (Instagram, WhatsApp, public datasets, interview answers) is turned
into `Conversation` records with `me`/`them` turns and stored as JSONL:

    {"source": "whatsapp", "turns": [{"role": "them", "text": "u up?"}, {"role": "me", "text": "ya"}]}

Keeping one format means the tokenizer, dataset, and trainer never need to know
where text came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable, Iterator

ROLES = ("me", "them")
DEFAULT_SESSION_GAP_SECONDS = 3 * 60 * 60


@dataclass(slots=True)
class Turn:
    role: str
    text: str

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"Unknown role: {self.role}")


@dataclass(slots=True)
class Conversation:
    turns: list[Turn]
    source: str = "unknown"
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def me_turn_count(self) -> int:
        return sum(1 for turn in self.turns if turn.role == "me")

    def to_json(self) -> str:
        return json.dumps(
            {
                "source": self.source,
                "meta": self.meta,
                "turns": [{"role": turn.role, "text": turn.text} for turn in self.turns],
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "Conversation":
        turns_payload = payload.get("turns", [])
        if not isinstance(turns_payload, list):
            raise TypeError("turns must be a list")
        turns = [Turn(role=str(item["role"]), text=str(item["text"])) for item in turns_payload]
        meta = payload.get("meta", {})
        return cls(turns=turns, source=str(payload.get("source", "unknown")), meta=meta if isinstance(meta, dict) else {})


@dataclass(slots=True)
class RawMessage:
    """One timestamped chat message before it is grouped into conversations."""

    sender: str
    timestamp: float
    text: str


def write_conversations(path: str | Path, conversations: Iterable[Conversation]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for conversation in conversations:
            handle.write(conversation.to_json() + "\n")
            count += 1
    return count


def read_conversations(path: str | Path) -> Iterator[Conversation]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield Conversation.from_dict(json.loads(line))


def sessionize(
    messages: Iterable[RawMessage],
    me_names: set[str],
    *,
    source: str,
    gap_seconds: float = DEFAULT_SESSION_GAP_SECONDS,
    meta: dict[str, object] | None = None,
) -> list[Conversation]:
    """Group one thread's messages into conversations.

    - a silence longer than `gap_seconds` starts a new conversation
    - consecutive messages from the same role are merged into one turn, since
      people send a thought as several short texts
    - conversations where "me" never speaks are dropped (nothing to learn from)
    """

    me_lower = {name.casefold() for name in me_names}
    conversations: list[Conversation] = []
    turns: list[Turn] = []
    last_timestamp: float | None = None

    def flush() -> None:
        if any(turn.role == "me" for turn in turns):
            conversations.append(Conversation(turns=list(turns), source=source, meta=dict(meta or {})))
        turns.clear()

    for message in sorted(messages, key=lambda item: item.timestamp):
        text = message.text.strip()
        if not text:
            continue
        if last_timestamp is not None and message.timestamp - last_timestamp > gap_seconds:
            flush()
        last_timestamp = message.timestamp

        role = "me" if message.sender.casefold() in me_lower else "them"
        if turns and turns[-1].role == role:
            turns[-1].text = f"{turns[-1].text}\n{text}"
        else:
            turns.append(Turn(role=role, text=text))

    flush()
    return conversations
