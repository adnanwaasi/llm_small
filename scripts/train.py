"""Train the tiny GPT: pretrain on public dialogues, then fine-tune on you.

    # stage 1: learn English chat in general
    python scripts/train.py --stage pretrain --data data/public/dialogues.jsonl --run-name base

    # stage 2: learn your voice (only your turns are graded)
    python scripts/train.py --stage finetune --data data/personal/conversations.jsonl \\
        --init-from artifacts/base/best.pt --run-name me
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import torch

from tinygpt.config import ModelConfig, TrainingConfig
from tinygpt.conversation import read_conversations
from tinygpt.dataset import ChatStream, split_conversations
from tinygpt.tokenizer import load_tokenizer
from tinygpt.trainer import load_checkpoint, train_model

STAGE_DEFAULTS = {
    "pretrain": {"learning_rate": 6e-4, "max_steps": 10_000, "warmup_steps": 300, "eval_interval": 500, "patience": None},
    "finetune": {"learning_rate": 1e-4, "max_steps": 3_000, "warmup_steps": 50, "eval_interval": 50, "patience": 6},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=sorted(STAGE_DEFAULTS), required=True)
    parser.add_argument("--data", action="append", required=True, help="Conversations JSONL (repeatable)")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json", help="Used for pretrain; finetune reuses the checkpoint's")
    parser.add_argument("--init-from", help="Checkpoint to start from (required for finetune)")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-conversations", type=int, default=None, help="Cap loaded conversations (quick experiments)")
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=None, help="Default 0.0 for pretrain, 0.1 for finetune")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--patience", type=int, help="Early-stop after this many evals without val improvement")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--cache-dir", default="data/cache", help="Where tokenized streams are cached between runs")
    return parser


def _cached_streams(args, tokenizer, train_on: str) -> tuple[ChatStream, ChatStream]:
    key_source = json.dumps(
        {
            "data": [(path, Path(path).stat().st_mtime) for path in args.data],
            "tokenizer": tokenizer.to_dict(),
            "train_on": train_on,
            "max": args.max_conversations,
            "val": args.val_fraction,
            "seed": args.seed,
        },
        sort_keys=True,
    )
    cache_path = Path(args.cache_dir) / f"stream-{hashlib.sha1(key_source.encode()).hexdigest()[:16]}.pt"
    if cache_path.exists():
        print(f"loading tokenized data from {cache_path}")
        payload = torch.load(cache_path, weights_only=False)
        return payload["train"], payload["val"]

    conversations = list(
        itertools.islice(itertools.chain.from_iterable(read_conversations(path) for path in args.data), args.max_conversations)
    )
    if train_on == "me":
        conversations = [conversation for conversation in conversations if conversation.me_turn_count]
    print(f"tokenizing {len(conversations)} conversations", flush=True)
    train_conversations, val_conversations = split_conversations(conversations, args.val_fraction, args.seed)
    if not val_conversations:
        raise SystemExit("Need at least 2 conversations so one can be held out for validation")
    train_stream = ChatStream.from_conversations(tokenizer, train_conversations, train_on=train_on)
    val_stream = ChatStream.from_conversations(tokenizer, val_conversations, train_on=train_on)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"train": train_stream, "val": val_stream}, cache_path)
    return train_stream, val_stream


def main() -> None:
    args = build_parser().parse_args()
    defaults = STAGE_DEFAULTS[args.stage]
    pick = lambda name: getattr(args, name) if getattr(args, name) is not None else defaults[name]  # noqa: E731

    init_state = None
    if args.stage == "finetune":
        if not args.init_from:
            raise SystemExit("finetune needs --init-from <pretrained checkpoint>")
        base_model, tokenizer, model_config, _ = load_checkpoint(args.init_from, device_name="cpu")
        init_state = base_model.state_dict()
        model_config.dropout = 0.1 if args.dropout is None else args.dropout
    else:
        tokenizer = load_tokenizer(args.tokenizer)
        model_config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            context_length=args.context_length,
            embedding_dim=args.embedding_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=0.0 if args.dropout is None else args.dropout,
        )
        if args.init_from:
            init_state = load_checkpoint(args.init_from, device_name="cpu")[0].state_dict()

    training_config = TrainingConfig(
        data_paths=list(args.data),
        tokenizer_path=args.tokenizer,
        stage=args.stage,
        init_from=args.init_from,
        run_name=args.run_name or args.stage,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        max_steps=pick("max_steps"),
        warmup_steps=pick("warmup_steps"),
        eval_interval=pick("eval_interval"),
        eval_batches=args.eval_batches,
        learning_rate=pick("learning_rate"),
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        patience=pick("patience"),
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
    )

    train_stream, val_stream = _cached_streams(args, tokenizer, training_config.train_on)
    best_checkpoint = train_model(training_config, model_config, tokenizer, train_stream, val_stream, init_state=init_state)
    print(f"Best checkpoint saved to {best_checkpoint}")


if __name__ == "__main__":
    main()
