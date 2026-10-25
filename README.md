# Tiny GPT That Texts Like You

A from-scratch ~16M-parameter GPT that learns **how you write**: your slang,
lowercase, emojis, Hinglish, and how you reply to people.

## Why two stages

A tiny model can't learn English from a few thousand of your messages. That is
why the first version only produced word soup. So training is split:

1. **Pretrain** on ~300k public everyday dialogues (DailyDialog + a slice of
   SODA, downloaded automatically). The model learns English and turn-taking.
2. **Fine-tune** on *your* conversations. The model reads the other person's
   messages as context, but it is only graded on your replies (loss mask), so
   it learns your voice, not your friends'.

```
public dialogues ─┐
                  ├─► BPE tokenizer ─► pretrain (loss on all turns) ─► base.pt
your chats ───────┘                                                    │
your chats ─────────────────────────► fine-tune (loss on <|me|> only) ◄┘ ─► me.pt ─► chat.py
```

## What data you need

Everything becomes `me`/`them` conversations (`data/personal/conversations.jsonl`).
Use any mix of these:

| Source | How to get it | Flag |
|---|---|---|
| Interview (no export needed) | `python scripts/interview.py`. Answer texts the way you'd really reply | default |
| Instagram | Settings → Your activity → Download your information → JSON | `--instagram DIR` |
| WhatsApp | Chat → ⋮ → More → Export chat → Without media (`.txt`) | `--whatsapp PATH` |
| Telegram | Telegram Desktop → Export chat history → JSON | `--telegram PATH` |
| Discord | [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter) → JSON | `--discord PATH` |
| Anything you wrote | captions, notes, posts as `.txt`/`.md` | `--text PATH` |

Rough guide to how much of your own text you need: **5k words** gives faint
style, **20k+** sounds recognizably like you, and **100k+** is where it gets good.
WhatsApp exports are usually the biggest source.

## Setup

```bash
uv venv --python /usr/bin/python3 .venv
uv pip install --python .venv/bin/python -e .
# GPU: uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu126
```

All commands below run from this directory with `.venv` active (or prefix with `PYTHONPATH=src .venv/bin/python`).

## Pipeline

```bash
# 1. public pretraining data (~270 MB JSONL)
python scripts/download_public_data.py

# 2. your data: interview yourself and/or import exports
python scripts/interview.py
python scripts/prepare_data.py --me "Your Name" --whatsapp exports/ --instagram fuck/
#   (find your exact Instagram name with: --instagram fuck/ --list-speakers)

# 3. tokenizer (byte-level BPE, 8k vocab; your text is upweighted)
python scripts/train_tokenizer.py

# 4. pretrain the base model (one time; ~1-2 h on a 4 GB GPU)
python scripts/train.py --stage pretrain --data data/public/dialogues.jsonl --run-name base

# 5. fine-tune on you (minutes; early-stops before memorizing)
python scripts/train.py --stage finetune --data data/personal/conversations.jsonl \
    --init-from artifacts/base/best.pt --run-name me

# 6. talk to it
python scripts/chat.py --checkpoint artifacts/me/best.pt
python scripts/chat.py --checkpoint artifacts/me/best.pt --message "wanna grab food later?" --samples 5
python scripts/generate.py --checkpoint artifacts/me/best.pt --prompt "caption: "
```

Re-run steps 2 and 5 whenever you add more of your data. Pretraining is only needed once.

## Example voice

`examples/claude_voice.jsonl` holds ~200 conversations written by Claude, used to test
the pipeline before any personal data existed. To build a model that talks like Claude:

```bash
python scripts/prepare_data.py --interview examples/claude_voice.jsonl
python scripts/train.py --stage finetune --data data/personal/conversations.jsonl \
    --init-from artifacts/base/best.pt --run-name claude --val-fraction 0.1 --eval-interval 25
python scripts/chat.py --checkpoint artifacts/claude/best.pt
```

## Tests

```bash
python -m pytest
```

## Privacy

Your exports and everything derived from them (`fuck/`, `data/`, `artifacts/`)
are in `.gitignore`. A fine-tuned checkpoint can repeat your messages word for
word, so don't publish it either.

See `architecture` for design details.
