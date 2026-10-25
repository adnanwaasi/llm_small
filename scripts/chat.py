"""Chat with the model: you play "them", the model replies as "me"."""

from __future__ import annotations

import argparse

import torch

from tinygpt.chat_format import build_reply_prompt
from tinygpt.conversation import Turn
from tinygpt.trainer import load_checkpoint, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--message", help="Reply to one message and exit instead of starting a REPL")
    parser.add_argument("--samples", type=int, default=1, help="How many alternative replies to show")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--history-turns", type=int, default=8, help="How many previous turns to keep as context")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser


def reply(model, tokenizer, history: list[Turn], args, device: torch.device) -> str:
    prompt_ids = build_reply_prompt(tokenizer, history[-args.history_turns :])
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        stop_token_ids={tokenizer.end_id},
        banned_token_ids={tokenizer.bos_id, tokenizer.me_id, tokenizer.them_id},
    )
    return tokenizer.decode(generated[0, len(prompt_ids) :].tolist()).strip()


def main() -> None:
    args = build_parser().parse_args()
    model, tokenizer, _, payload = load_checkpoint(args.checkpoint, device_name=args.device)
    device = resolve_device(args.device)
    if getattr(tokenizer, "tokenizer_type", None) != "bpe":
        raise SystemExit("chat.py needs a checkpoint trained with the BPE chat pipeline; use generate.py for old runs")

    if args.message:
        for _ in range(args.samples):
            print(reply(model, tokenizer, [Turn("them", args.message)], args, device))
        return

    print(f"loaded {args.checkpoint} (step {payload.get('step')}). type a message, /reset to clear, Ctrl-D to quit")
    history: list[Turn] = []
    while True:
        try:
            message = input("them> ").strip()
        except EOFError:
            print()
            return
        if not message:
            continue
        if message == "/reset":
            history.clear()
            continue
        history.append(Turn("them", message))
        answer = reply(model, tokenizer, history, args, device)
        history.append(Turn("me", answer or "..."))
        print(f"me> {answer}")


if __name__ == "__main__":
    main()
