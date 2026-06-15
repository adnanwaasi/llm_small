"""Parse Instagram export JSON files into normalized training text."""

from __future__ import annotations

import argparse

from tinygpt.config import ParserConfig
from tinygpt.instagram_parser import collect_speaker_counts, write_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="fuck", help="Root directory of the Instagram export")
    parser.add_argument("--output-text", default="data/processed/corpus.txt", help="Where to write the plain text corpus")
    parser.add_argument("--output-jsonl", default="data/processed/messages.jsonl", help="Where to write normalized message records")
    parser.add_argument("--speaker", default=None, help="Filter the corpus to one sender display name")
    parser.add_argument("--exclude-message-requests", action="store_true", help="Skip Instagram message request threads")
    parser.add_argument("--include-reactions", action="store_true", help="Keep reaction-only events as text placeholders")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase extracted text before writing the corpus")
    parser.add_argument("--clean-training-text", action="store_true", help="Drop mojibake-heavy and junky lines for more readable training data")
    parser.add_argument("--list-speakers", action="store_true", help="Print speaker counts and exit")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ParserConfig(
        input_dir=args.input_dir,
        output_text=args.output_text,
        output_jsonl=args.output_jsonl,
        speaker=args.speaker,
        include_message_requests=not args.exclude_message_requests,
        include_reactions=args.include_reactions,
        lowercase=args.lowercase,
        clean_training_text=args.clean_training_text,
    )

    if args.list_speakers:
        counts = collect_speaker_counts(config)
        for speaker_name, count in counts.most_common():
            print(f"{speaker_name}\t{count}")
        return

    message_count, character_count = write_corpus(config)
    print(f"Wrote {message_count} messages and {character_count} characters")
    if args.speaker:
        print(f"Speaker filter: {args.speaker}")


if __name__ == "__main__":
    main()
