"""Download public dialogue datasets and convert them to conversations JSONL."""

from __future__ import annotations

import argparse
import itertools

from tinygpt.conversation import write_conversations
from tinygpt.public_data import DATASETS, iter_public_conversations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["dailydialog", "soda"], choices=sorted(DATASETS))
    parser.add_argument("--soda-max", type=int, default=300_000, help="Cap SODA dialogues (the full set is ~1.5M)")
    parser.add_argument("--cache-dir", default="data/raw/public")
    parser.add_argument("--output", default="data/public/dialogues.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    streams = []
    for name in args.datasets:
        limit = args.soda_max if name == "soda" else None
        # Each SODA shard holds ~380k dialogues, so only fetch the shards we need.
        max_files = None if limit is None else max(1, -(-limit // 380_000))
        streams.append(
            iter_public_conversations(name, args.cache_dir, max_conversations=limit, max_files=max_files, seed=args.seed)
        )
    count = write_conversations(args.output, itertools.chain(*streams))
    print(f"Wrote {count} public conversations to {args.output}")


if __name__ == "__main__":
    main()
