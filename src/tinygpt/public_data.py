"""Download open conversational datasets for pretraining.

A tiny model cannot learn English from a few thousand personal messages, so we
first teach it general turn-taking chat from public dialogue datasets, then
fine-tune it on your messages.

- DailyDialog (~13k human-written everyday dialogues, short and casual)
- SODA (~1.5M dialogues generated from commonsense situations; we take a slice)

Both are fetched as Parquet from the Hugging Face dataset viewer mirror.
"""

from __future__ import annotations

from pathlib import Path
import random
import re
import shutil
import urllib.request

from tinygpt.conversation import Conversation, Turn

HF_PARQUET = "https://huggingface.co/datasets/{dataset}/resolve/refs%2Fconvert%2Fparquet/{path}"
DATASETS = {
    "dailydialog": {
        "dataset": "roskoN/dailydialog",
        "files": ["full/train/0000.parquet", "full/validation/0000.parquet", "full/test/0000.parquet"],
        "column": "utterances",
    },
    "soda": {
        "dataset": "allenai/soda",
        "files": ["default/train/0000.parquet", "default/train/0001.parquet", "default/train/0002.parquet"],
        "column": "dialogue",
    },
}

_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,!?;:%)])")
_CONTRACTION_RE = re.compile(r"\s*([’'])\s*(s|t|re|ve|ll|d|m)\b", re.IGNORECASE)


def detokenize(text: str) -> str:
    """DailyDialog is pre-tokenized ("Good morning , sir ."); undo that."""

    text = text.replace(" n't", "n't").replace(" ’ ", "'")
    text = _CONTRACTION_RE.sub(r"'\2", text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = re.sub(r"([.!?])(?=[A-Z])", r"\1 ", text)
    return re.sub(r"\s+", " ", text).strip()


def download(url: str, destination: Path) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    partial.rename(destination)
    return destination


def dialogue_to_conversation(utterances: list[str], source: str, rng: random.Random) -> Conversation | None:
    texts = [detokenize(utterance) for utterance in utterances if utterance and utterance.strip()]
    if len(texts) < 2:
        return None
    # Randomize who is "me" so the model learns both sides of the conversation.
    first_role = rng.choice(("me", "them"))
    other_role = "them" if first_role == "me" else "me"
    turns = [Turn(first_role if index % 2 == 0 else other_role, text) for index, text in enumerate(texts)]
    return Conversation(turns=turns, source=source)


def iter_public_conversations(
    name: str,
    cache_dir: str | Path = "data/raw/public",
    *,
    max_conversations: int | None = None,
    max_files: int | None = None,
    seed: int = 0,
):
    import pyarrow.parquet as pq

    spec = DATASETS[name]
    rng = random.Random(seed)
    produced = 0
    files = spec["files"][:max_files] if max_files else spec["files"]
    for relative in files:
        url = HF_PARQUET.format(dataset=spec["dataset"], path=relative)
        local = download(url, Path(cache_dir) / name / relative.replace("/", "_"))
        parquet = pq.ParquetFile(local)
        for batch in parquet.iter_batches(columns=[spec["column"]], batch_size=2048):
            for utterances in batch.column(0).to_pylist():
                conversation = dialogue_to_conversation(utterances or [], name, rng)
                if conversation is None:
                    continue
                yield conversation
                produced += 1
                if max_conversations is not None and produced >= max_conversations:
                    return
