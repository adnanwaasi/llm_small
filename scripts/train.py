"""Train a tiny character-level GPT on a prepared text corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from tinygpt.config import ModelConfig, TrainingConfig
from tinygpt.tokenizer import build_tokenizer
from tinygpt.trainer import train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-text", required=True, help="Path to the plain text corpus")
    parser.add_argument("--run-name", default="tiny-gpt-run", help="Name for the artifact directory")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--tokenizer-type", choices=["word", "character"], default="word")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--sample-length", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output-dir", default="artifacts")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    corpus_path = Path(args.input_text)
    corpus_text = corpus_path.read_text(encoding="utf-8")
    if not corpus_text.strip():
        raise ValueError("Input corpus is empty. Run prepare_data.py first.")

    tokenizer = build_tokenizer(corpus_text, args.tokenizer_type)
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=args.context_length,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    training_config = TrainingConfig(
        input_text=str(corpus_path),
        run_name=args.run_name,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_split=args.train_split,
        sample_length=args.sample_length,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
    )

    best_checkpoint = train_model(training_config, model_config, tokenizer, corpus_text)
    print(f"Best checkpoint saved to {best_checkpoint}")


if __name__ == "__main__":
    main()
