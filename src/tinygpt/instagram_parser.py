"""Utilities for parsing Instagram export JSON files into training examples.

The parser targets the standard Instagram export structure discovered in this repo:

- fuck/your_instagram_activity/messages/inbox/*/message_*.json
- fuck/your_instagram_activity/messages/message_requests/*/message_*.json

Each thread JSON contains a `messages` array with fields such as `sender_name`,
`timestamp_ms`, `content`, `share`, `photos`, and `videos`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Iterable
import unicodedata

from tinygpt.config import ParserConfig
from tinygpt.conversation import Conversation, RawMessage, sessionize


IGNORED_CONTENT_STRINGS = {
    "sent an attachment.",
    "you sent an attachment.",
}

ATTACHMENT_RE = re.compile(r"^.+\s+sent an attachment\.?$", re.IGNORECASE)
REACTION_RE = re.compile(r"^reacted .* to .* message\.?$", re.IGNORECASE)
LIKED_MESSAGE_RE = re.compile(r"^liked a message\.?$", re.IGNORECASE)
GROUP_EVENT_RE = re.compile(
    r"(^.+ added .+ to the group\.?$|^.+ created the group chat\.?$|^.+ left the group\.?$)",
    re.IGNORECASE,
)
CALL_EVENT_RE = re.compile(r"(^you started an audio call\.?$|^audio call ended\.?$)", re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
ZERO_WIDTH_CHARACTERS = {"\u200b", "\u200c", "\u200d", "\ufeff"}
SENTENCE_ENDINGS = (".", "!", "?", "…", '"', "”", "'")
MOJIBAKE_CHARACTERS = {"ð", "â", "Ã", "¤", "�"}


@dataclass(slots=True)
class ParsedMessage:
    """A normalized message record extracted from Instagram export files."""

    thread_path: str
    thread_title: str
    sender_name: str
    timestamp_ms: int
    timestamp_iso: str
    text: str
    media_uris: list[str]
    source_file: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def discover_message_files(root: Path, include_message_requests: bool = True) -> list[Path]:
    """Return all Instagram thread JSON files under the export root."""

    messages_root = root / "your_instagram_activity" / "messages"
    files: list[Path] = []
    files.extend(sorted(messages_root.glob("inbox/*/message_*.json")))
    if include_message_requests:
        files.extend(sorted(messages_root.glob("message_requests/*/message_*.json")))

    return sorted(files)


def iter_parsed_messages(config: ParserConfig) -> Iterable[ParsedMessage]:
    """Yield normalized messages from all matching export files.

    Messages are globally sorted by timestamp so the final corpus follows natural
    conversation order rather than archive file order.
    """

    root = Path(config.input_dir)
    message_files = discover_message_files(root, config.include_message_requests)
    parsed: list[ParsedMessage] = []

    for file_path in message_files:
        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            thread = json.load(handle)

        thread_title = str(thread.get("title", "unknown_thread"))
        thread_path = str(thread.get("thread_path", file_path.parent.name))

        for raw_message in thread.get("messages", []):
            normalized = _normalize_message(
                raw_message=raw_message,
                thread_path=thread_path,
                thread_title=thread_title,
                source_file=file_path,
                lowercase=config.lowercase,
                include_reactions=config.include_reactions,
            )
            if normalized is None:
                continue

            if config.speaker is not None and normalized.sender_name != config.speaker:
                continue

            parsed.append(normalized)

    parsed.sort(key=lambda item: item.timestamp_ms)
    yield from parsed


def iter_instagram_conversations(
    input_dir: str | Path,
    me_names: set[str],
    *,
    include_message_requests: bool = True,
    gap_seconds: float | None = None,
) -> Iterable[Conversation]:
    """Yield `me`/`them` conversations, one thread at a time, split by silence gaps."""

    root = Path(input_dir)
    threads: dict[Path, list[RawMessage]] = {}
    for file_path in discover_message_files(root, include_message_requests):
        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            thread = json.load(handle)

        bucket = threads.setdefault(file_path.parent, [])
        for raw_message in thread.get("messages", []):
            normalized = _normalize_message(
                raw_message=raw_message,
                thread_path=str(file_path.parent.name),
                thread_title="",
                source_file=file_path,
                lowercase=False,
                include_reactions=False,
            )
            if normalized is None or not normalized.text:
                continue
            bucket.append(
                RawMessage(
                    sender=normalized.sender_name,
                    timestamp=normalized.timestamp_ms / 1000,
                    text=normalized.text,
                )
            )

    for thread_dir, messages in threads.items():
        kwargs = {} if gap_seconds is None else {"gap_seconds": gap_seconds}
        yield from sessionize(messages, me_names, source="instagram", meta={"thread": thread_dir.name}, **kwargs)


def collect_speaker_counts(config: ParserConfig) -> Counter[str]:
    """Count how many messages each speaker has in the export."""

    counts: Counter[str] = Counter()
    root = Path(config.input_dir)

    for file_path in discover_message_files(root, config.include_message_requests):
        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            thread = json.load(handle)

        for raw_message in thread.get("messages", []):
            sender_name = raw_message.get("sender_name")
            if isinstance(sender_name, str) and sender_name.strip():
                counts[fix_mojibake(sender_name)] += 1

    return counts


def write_corpus(config: ParserConfig) -> tuple[int, int]:
    """Write normalized JSONL records and plain-text corpus files.

    Returns:
        A tuple of `(message_count, character_count)`.
    """

    output_text = Path(config.output_text)
    output_jsonl = Path(config.output_jsonl)
    output_text.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    messages = list(iter_parsed_messages(config))

    with output_jsonl.open("w", encoding="utf-8") as jsonl_handle:
        for message in messages:
            jsonl_handle.write(message.to_json() + "\n")

    corpus_lines: list[str] = []
    previous_text: str | None = None
    for message in messages:
        text = message.text.strip()
        if not text:
            continue
        if config.clean_training_text and not _is_training_line_usable(text):
            continue
        if text == previous_text:
            continue

        if corpus_lines and previous_text is not None and _should_merge_with_previous(previous_text, text):
            merged_text = f"{previous_text} {text}".strip()
            corpus_lines[-1] = merged_text
            previous_text = merged_text
            continue

        corpus_lines.append(text)
        previous_text = text

    corpus = "\n".join(corpus_lines)
    with output_text.open("w", encoding="utf-8") as text_handle:
        text_handle.write(corpus)

    return len(messages), len(corpus)


def _normalize_message(
    *,
    raw_message: dict[str, object],
    thread_path: str,
    thread_title: str,
    source_file: Path,
    lowercase: bool,
    include_reactions: bool,
) -> ParsedMessage | None:
    sender_name = raw_message.get("sender_name")
    timestamp_ms = raw_message.get("timestamp_ms")
    if not isinstance(sender_name, str) or not isinstance(timestamp_ms, int):
        return None

    text = _extract_text(raw_message, include_reactions=include_reactions)
    if lowercase:
        text = text.lower()

    media_uris = _extract_media_uris(raw_message)
    timestamp_iso = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).isoformat()

    return ParsedMessage(
        thread_path=thread_path,
        thread_title=thread_title,
        sender_name=fix_mojibake(sender_name),
        timestamp_ms=timestamp_ms,
        timestamp_iso=timestamp_iso,
        text=text,
        media_uris=media_uris,
        source_file=str(source_file),
    )


def _extract_text(raw_message: dict[str, object], *, include_reactions: bool) -> str:
    share = raw_message.get("share")
    share_text_value: str | None = None
    if isinstance(share, dict):
        share_text = share.get("share_text")
        if isinstance(share_text, str) and share_text.strip():
            share_text_value = URL_RE.sub("[url]", sanitize_text(share_text))

    content = raw_message.get("content")
    if isinstance(content, str) and content.strip():
        normalized_content = sanitize_text(content)
        if _should_ignore_content(normalized_content):
            return share_text_value or ""
        return normalized_content

    if share_text_value is not None:
        return share_text_value

    media_uris = _extract_media_uris(raw_message)
    if media_uris:
        return ""

    if include_reactions:
        reactions = raw_message.get("reactions")
        if isinstance(reactions, list) and reactions:
            reaction_bits: list[str] = []
            for reaction in reactions:
                if not isinstance(reaction, dict):
                    continue
                emoji = reaction.get("reaction")
                actor = reaction.get("actor")
                if isinstance(emoji, str) and isinstance(actor, str):
                    reaction_bits.append(f"{actor}:{emoji}")
            if reaction_bits:
                return "[reactions] " + ", ".join(reaction_bits)

    return ""


def _extract_media_uris(raw_message: dict[str, object]) -> list[str]:
    media_uris: list[str] = []
    for field_name in ("photos", "videos", "audio_files", "files"):
        value = raw_message.get(field_name)
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict):
                continue
            uri = item.get("uri")
            if isinstance(uri, str) and uri.strip():
                media_uris.append(uri.strip())
    return media_uris


def _should_ignore_content(content: str) -> bool:
    normalized_lower = content.lower()
    if normalized_lower in IGNORED_CONTENT_STRINGS:
        return True

    for pattern in (ATTACHMENT_RE, REACTION_RE, LIKED_MESSAGE_RE, GROUP_EVENT_RE, CALL_EVENT_RE):
        if pattern.match(content):
            return True

    return False


def fix_mojibake(text: str) -> str:
    """Undo Instagram's habit of exporting UTF-8 bytes as latin-1 characters.

    e.g. `\u00f0\u009f\u0098\u0082` in the JSON is really the 😂 emoji.
    """

    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def sanitize_text(text: str) -> str:
    """Normalize export text and strip invisible/control junk."""

    normalized = unicodedata.normalize("NFC", fix_mojibake(text))
    cleaned_characters: list[str] = []
    for character in normalized:
        if character in ZERO_WIDTH_CHARACTERS:
            continue
        if character == "�":
            continue
        category = unicodedata.category(character)
        if category.startswith("C") and character not in {"\n", "\t", " "}:
            continue
        cleaned_characters.append(character)

    cleaned = "".join(cleaned_characters).strip()
    return re.sub(r"\s+", " ", cleaned)


def _should_merge_with_previous(previous_text: str, current_text: str) -> bool:
    """Heuristic for merging fragmented chat lines into longer sentence spans."""

    previous_word_count = len(previous_text.split())
    current_word_count = len(current_text.split())

    if previous_word_count <= 3 or current_word_count <= 3:
        return True

    if not previous_text.rstrip().endswith(SENTENCE_ENDINGS):
        return True

    return False


def _is_training_line_usable(text: str) -> bool:
    """Return whether a line is clean enough for sentence-focused training."""

    stripped = text.strip()
    if not stripped:
        return False

    alphabetic_count = sum(1 for character in stripped if character.isalpha())
    mojibake_count = sum(1 for character in stripped if character in MOJIBAKE_CHARACTERS)

    if alphabetic_count < 3 and len(stripped.split()) <= 2:
        return False

    if mojibake_count >= 2:
        return False

    if len(stripped) > 240 and ("whatsapp" in stripped.lower() or "http" in stripped.lower()):
        return False

    return True
