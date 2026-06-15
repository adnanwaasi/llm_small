"""Generate text from a trained tiny GPT checkpoint."""

from __future__ import annotations

import argparse

import torch

from tinygpt.trainer import load_checkpoint, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a saved checkpoint")
    parser.add_argument("--prompt", default="", help="Seed text to start generation")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8, help="Lower is safer and more deterministic")
    parser.add_argument("--top-k", type=int, default=20, help="Only sample from the top-k logits")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Discourage token repetition when > 1.0")
    parser.add_argument("--greedy", action="store_true", help="Use argmax decoding instead of sampling")
    parser.add_argument("--only-new-text", action="store_true", help="Print only the generated continuation, not the prompt")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model, tokenizer, _, _ = load_checkpoint(args.checkpoint, device_name=args.device)
    device = resolve_device(args.device)

    if args.prompt:
        prompt_ids = tokenizer.encode_with_fallback(args.prompt)
    else:
        prompt_ids = [tokenizer.default_start_token_id]

    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(
        prompt_tensor,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        do_sample=not args.greedy,
    )
    generated_ids = generated[0].tolist()
    if args.only_new_text:
        generated_ids = generated_ids[len(prompt_ids) :]
    output = tokenizer.decode(generated_ids)
    print(output)


if __name__ == "__main__":
    main()
