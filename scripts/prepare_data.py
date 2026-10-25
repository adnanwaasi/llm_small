"""Collect your messages from every source into one conversations JSONL file."""

from __future__ import annotations

import argparse
from collections import Counter

from tinygpt.config import ParserConfig
from tinygpt.conversation import Conversation, write_conversations
from tinygpt.instagram_parser import collect_speaker_counts, iter_instagram_conversations
from tinygpt.sources import (
    iter_discord_conversations,
    iter_interview_conversations,
    iter_telegram_conversations,
    iter_text_conversations,
    iter_whatsapp_conversations,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--me", action="append", default=[], help="Your display name in chats (repeat for multiple names)")
    parser.add_argument("--instagram", help="Root directory of an Instagram export")
    parser.add_argument("--whatsapp", help="WhatsApp exported .txt file or a directory of them")
    parser.add_argument("--telegram", help="Telegram Desktop result.json or a directory of exports")
    parser.add_argument("--discord", help="DiscordChatExporter .json file or a directory of them")
    parser.add_argument("--text", help="A .txt/.md file or directory of things you wrote yourself")
    parser.add_argument("--interview", default="data/personal/interview.jsonl", help="Answers from scripts/interview.py")
    parser.add_argument("--output", default="data/personal/conversations.jsonl")
    parser.add_argument("--gap-hours", type=float, default=3.0, help="Silence that splits a chat thread into conversations")
    parser.add_argument("--exclude-message-requests", action="store_true", help="Skip Instagram message request threads")
    parser.add_argument("--list-speakers", action="store_true", help="Print Instagram speaker counts and exit")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.list_speakers:
        if not args.instagram:
            raise SystemExit("--list-speakers needs --instagram")
        counts = collect_speaker_counts(
            ParserConfig(input_dir=args.instagram, include_message_requests=not args.exclude_message_requests)
        )
        for speaker_name, count in counts.most_common():
            print(f"{speaker_name}\t{count}")
        return

    chat_sources = [args.instagram, args.whatsapp, args.telegram, args.discord]
    if any(chat_sources) and not args.me:
        raise SystemExit("Chat exports need --me so the parser knows which messages are yours")

    me_names = set(args.me)
    gap_seconds = args.gap_hours * 3600
    conversations: list[Conversation] = []
    if args.instagram:
        conversations.extend(
            iter_instagram_conversations(
                args.instagram,
                me_names,
                include_message_requests=not args.exclude_message_requests,
                gap_seconds=gap_seconds,
            )
        )
    if args.whatsapp:
        conversations.extend(iter_whatsapp_conversations(args.whatsapp, me_names))
    if args.telegram:
        conversations.extend(iter_telegram_conversations(args.telegram, me_names))
    if args.discord:
        conversations.extend(iter_discord_conversations(args.discord, me_names))
    if args.text:
        conversations.extend(iter_text_conversations(args.text))
    try:
        conversations.extend(iter_interview_conversations(args.interview))
    except FileNotFoundError:
        pass

    if not conversations:
        raise SystemExit("No conversations found. Run scripts/interview.py or pass an export.")

    write_conversations(args.output, conversations)
    by_source = Counter(conversation.source for conversation in conversations)
    me_words = sum(len(turn.text.split()) for c in conversations for turn in c.turns if turn.role == "me")
    me_turns = sum(conversation.me_turn_count for conversation in conversations)
    print(f"Wrote {len(conversations)} conversations to {args.output}")
    print(f"your turns={me_turns} your words={me_words} sources={dict(by_source)}")
    if me_words < 20_000:
        print("tip: under ~20k of your words the style will be faint; keep adding interview answers or exports")


if __name__ == "__main__":
    main()
