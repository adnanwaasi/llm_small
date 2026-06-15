# Tiny GPT From Instagram Chats

This project trains a small GPT on your Instagram export so it can learn your writing style.

## What it does

- Parses Instagram message export JSON files from `fuck/`
- Normalizes them into structured message records
- Builds a text corpus, optionally filtered to a single speaker
- Trains a tiny decoder-only GPT with PyTorch
- Generates text in the learned style

## Project layout

- `src/tinygpt/config.py` - dataclasses for parser, model, and training config
- `src/tinygpt/instagram_parser.py` - Instagram export parser and corpus builder
- `src/tinygpt/tokenizer.py` - character and word-level tokenizers
- `src/tinygpt/dataset.py` - contiguous text dataset utilities
- `src/tinygpt/model.py` - tiny GPT implementation
- `src/tinygpt/trainer.py` - training and checkpoint helpers
- `scripts/prepare_data.py` - convert Instagram JSON into corpus files
- `scripts/train.py` - train the model
- `scripts/generate.py` - sample text from a trained checkpoint

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 1) Inspect available speaker names

This helps you find the exact display name that represents your messages.

```bash
python scripts/prepare_data.py --input-dir "fuck" --list-speakers
```

## 2) Build a corpus for one speaker

Replace the speaker name with your exact Instagram display name.

```bash
python scripts/prepare_data.py \
  --input-dir "fuck" \
  --speaker "YOUR DISPLAY NAME" \
  --output-text "data/processed/you.txt" \
  --output-jsonl "data/processed/you.jsonl"
```

This writes:

- `data/processed/you.jsonl` - normalized per-message records
- `data/processed/you.txt` - plain text corpus used for training

## 3) Train the tiny GPT

```bash
python scripts/train.py \
  --input-text "data/processed/you.txt" \
  --run-name "you-style" \
  --tokenizer-type "word"
```

Checkpoints are saved under `artifacts/you-style/`.

## 4) Generate text

```bash
python scripts/generate.py \
  --checkpoint "artifacts/you-style/best.pt" \
  --prompt "hey "
```

## Notes

- This is a learning-sized GPT, not a production LLM.
- The default tokenizer is now a simple word-level tokenizer for more readable output. You can still switch back to character-level training with `--tokenizer-type character`.
- If you want the model to imitate *you* specifically, train on only your messages using `--speaker`.
- Your Instagram export is private data. Do not publish `fuck/` or `data/processed/`.
