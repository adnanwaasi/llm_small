"""Parsers for non-Instagram sources of *your* writing.

Supported:

- WhatsApp: "Export chat" -> "Without media" `.txt` files
- Telegram: Telegram Desktop "Export chat history" -> JSON (`result.json`)
- Discord: DiscordChatExporter JSON files
- Plain text: anything you wrote yourself (notes, captions, essays, tweets),
  one paragraph per blank-line-separated block
- Interview: JSONL produced by `scripts/interview.py`
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
from typing import Iterable, Iterator

from tinygpt.conversation import Conversation, RawMessage, Turn, read_conversations, sessionize
from tinygpt.instagram_parser import URL_RE, sanitize_text

WHATSAPP_LINE_RE = re.compile(
    r"^\[?(?P<date>\d{1,4}[./-]\d{1,2}[./-]\d{1,4}),?\s+(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap]\.?[Mm]\.?)?)\]?"
    r"\s*(?:-\s*)?(?P<sender>[^:]+?):\s(?P<text>.*)$"
)
WHATSAPP_SKIP = (
    "<media omitted>",
    "this message was deleted",
    "you deleted this message",
    "messages and calls are end-to-end encrypted",
    "<attached:",
    "null",
    "image omitted",
    "video omitted",
    "sticker omitted",
    "audio omitted",
    "gif omitted",
    "document omitted",
)


def _clean(text: str) -> str:
    return URL_RE.sub("[url]", sanitize_text(text))


def _files(path: str | Path, pattern: str) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    return sorted(root.rglob(pattern))


def _whatsapp_day_first(dates: list[str]) -> bool:
    for date in dates:
        first = int(re.split(r"[./-]", date)[0])
        if first > 12 and first < 32:
            return True
    return False


def _parse_whatsapp_time(date: str, time: str, day_first: bool) -> float:
    parts = [int(part) for part in re.split(r"[./-]", date)]
    if parts[0] > 31:
        year, month, day = parts
    elif day_first:
        day, month, year = parts
    else:
        month, day, year = parts
    if year < 100:
        year += 2000

    clean_time = time.replace(".", "").upper().replace(" ", "")
    for fmt in ("%I:%M:%S%p", "%I:%M%p", "%H:%M:%S", "%H:%M"):
        try:
            parsed = datetime.strptime(clean_time, fmt)
            break
        except ValueError:
            continue
    else:
        parsed = datetime(1900, 1, 1)
    try:
        return datetime(year, month, day, parsed.hour, parsed.minute, parsed.second).timestamp()
    except ValueError:
        return 0.0


def parse_whatsapp_messages(text: str) -> list[RawMessage]:
    # WhatsApp sprinkles U+200E (left-to-right mark) around timestamps on iOS.
    lines = text.replace("‎", "").splitlines()
    matches = [WHATSAPP_LINE_RE.match(line) for line in lines]
    day_first = _whatsapp_day_first([match.group("date") for match in matches if match])

    messages: list[RawMessage] = []
    for line, match in zip(lines, matches):
        if match is None:
            # Multi-line message continuation.
            if messages and line.strip():
                messages[-1].text += "\n" + line
            continue
        messages.append(
            RawMessage(
                sender=match.group("sender").strip(),
                timestamp=_parse_whatsapp_time(match.group("date"), match.group("time"), day_first),
                text=match.group("text"),
            )
        )

    cleaned: list[RawMessage] = []
    for message in messages:
        lowered = message.text.strip().lower()
        if any(lowered.startswith(marker) for marker in WHATSAPP_SKIP):
            continue
        message.text = "\n".join(_clean(part) for part in message.text.splitlines())
        cleaned.append(message)
    return cleaned


def iter_whatsapp_conversations(path: str | Path, me_names: set[str]) -> Iterator[Conversation]:
    for file_path in _files(path, "*.txt"):
        messages = parse_whatsapp_messages(file_path.read_text(encoding="utf-8", errors="replace"))
        yield from sessionize(messages, me_names, source="whatsapp", meta={"file": file_path.name})


def _telegram_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(part if isinstance(part, str) else str(part.get("text", "")) for part in value)
    return ""


def iter_telegram_conversations(path: str | Path, me_names: set[str]) -> Iterator[Conversation]:
    for file_path in _files(path, "*.json"):
        payload = json.loads(file_path.read_text(encoding="utf-8", errors="replace"))
        chats = payload.get("chats", {}).get("list") if isinstance(payload.get("chats"), dict) else None
        if chats is None:
            chats = [payload]
        for chat in chats:
            messages: list[RawMessage] = []
            for message in chat.get("messages", []):
                if message.get("type") != "message" or message.get("media_type"):
                    continue
                text = _clean(_telegram_text(message.get("text")))
                if not text:
                    continue
                timestamp = float(message.get("date_unixtime") or 0)
                messages.append(RawMessage(sender=str(message.get("from") or ""), timestamp=timestamp, text=text))
            yield from sessionize(messages, me_names, source="telegram", meta={"chat": str(chat.get("name", ""))})


def iter_discord_conversations(path: str | Path, me_names: set[str]) -> Iterator[Conversation]:
    for file_path in _files(path, "*.json"):
        payload = json.loads(file_path.read_text(encoding="utf-8", errors="replace"))
        messages: list[RawMessage] = []
        me_lower = {name.casefold() for name in me_names}
        for message in payload.get("messages", []):
            author = message.get("author", {})
            names = {str(author.get("name", "")), str(author.get("nickname", ""))}
            sender = next((name for name in names if name.casefold() in me_lower), str(author.get("name", "")))
            text = _clean(str(message.get("content", "")))
            if not text:
                continue
            try:
                timestamp = datetime.fromisoformat(str(message.get("timestamp"))).timestamp()
            except ValueError:
                timestamp = 0.0
            messages.append(RawMessage(sender=sender, timestamp=timestamp, text=text))
        yield from sessionize(messages, me_names, source="discord", meta={"file": file_path.name})


def iter_text_conversations(path: str | Path, *, min_chars: int = 20) -> Iterator[Conversation]:
    """Turn free-form writing into single-turn `me` monologues."""

    for file_path in [*_files(path, "*.txt"), *_files(path, "*.md")]:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        for block in re.split(r"\n\s*\n", text):
            cleaned = "\n".join(_clean(line) for line in block.splitlines() if line.strip())
            if len(cleaned) >= min_chars:
                yield Conversation(turns=[Turn("me", cleaned)], source="text", meta={"file": file_path.name})


def iter_interview_conversations(path: str | Path) -> Iterable[Conversation]:
    for file_path in _files(path, "*.jsonl"):
        yield from read_conversations(file_path)
