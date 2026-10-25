"""Collect training data in your voice by answering texts like a friend sent them.

No chat export? This is the fastest way to get data. Reply exactly how you
would text: same slang, same lowercase, same emojis, same Hinglish. Do not
"write nicely". Each answer is saved immediately, so quit any time (Ctrl-D) and
resume later; prompts you already answered are skipped.

Commands while answering:  (empty) skip   /quit stop   /undo drop last answer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from tinygpt.conversation import Conversation, Turn

PROMPTS = [
    # small talk
    "hey wyd", "u up?", "how was ur day", "what did u eat today", "bro where are u", "you free this weekend?",
    "good morning", "gn", "kya kar raha hai", "long time no see, how've you been?", "sup", "you alive?",
    "did u sleep at all lol", "what are u up to tonight", "hows college going", "how's work",
    # plans
    "wanna grab food later?", "movie tonight?", "can you pick me up at 6", "let's go on a trip next month",
    "are u coming to the party", "can we reschedule to tomorrow", "gym today or nah", "call me when free",
    "what time should we meet", "i'm outside", "you're late again 😤",
    # reactions
    "i got the job!!", "i failed my exam 😭", "my phone just died and i lost everything", "guess who texted me",
    "i'm so bored", "i can't sleep", "that was the best day ever", "i think i'm getting sick", "i'm so stressed rn",
    "omg did you see what happened", "i broke up with them", "my laptop crashed again", "it's raining so hard here",
    "i just finished the whole series in one night", "i cooked for the first time lol", "i'm moving to a new city",
    # opinions
    "what's your favorite movie and why", "best song right now?", "cats or dogs", "what do u think about ai taking jobs",
    "is pineapple on pizza okay", "what's the most overrated thing", "recommend me a show", "which phone should i buy",
    "what's ur take on social media", "android or iphone", "what's the best food in your city",
    "what do you do when you're sad", "what's your dream job", "would you rather be rich or famous",
    "linux or windows", "what's a skill you wish you had", "favorite game of all time?",
    # you
    "tell me something about yourself nobody knows", "what are you working on these days", "what's stressing you out",
    "what are you excited about", "describe your perfect day", "what's ur biggest flex", "what's ur morning routine",
    "what did u do last weekend", "what's the last thing that made you laugh", "who do you look up to",
    "what are your goals this year", "explain what you study/do like i'm 5", "what's your hot take",
    # help / banter
    "can you help me with my code", "my code works and i don't know why", "i need advice", "lend me 500 pls",
    "you owe me food", "stop ignoring me", "why didn't you reply", "you're so annoying 😂", "ok but hear me out",
    "be honest, how do i look in this pic", "i miss u man", "thank you so much for yesterday",
    "sorry i was rude earlier", "happy birthday!!!", "congrats bro", "lol", "😂😂😂", "hmm", "ok", "fr?", "no way",
    # longer writing
    "write a caption for a beach pic", "write an insta caption for a pic with friends",
    "write a story for your status today", "rant about something that annoyed you this week",
    "explain why your favorite thing is the best", "tell me about a trip you remember",
    "what would you post if you could only post once this year",
]
FOLLOW_UPS = ["why?", "fr?", "lol same", "wait what", "tell me more", "hmm and then?", "nice", "ok and?", "😂", "really?"]


def _load(path: Path) -> list[Conversation]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [Conversation.from_dict(json.loads(line)) for line in handle if line.strip()]


def _save(path: Path, conversations: list[Conversation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(conversation.to_json() + "\n" for conversation in conversations), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="data/personal/interview.jsonl")
    parser.add_argument("--follow-up-rate", type=float, default=0.35, help="Chance of a follow-up text after your reply")
    parser.add_argument("--extra-prompts", help="Text file with more prompts, one per line")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.output)
    rng = random.Random(args.seed)
    conversations = _load(output)

    prompts = list(PROMPTS)
    if args.extra_prompts:
        prompts += [line.strip() for line in Path(args.extra_prompts).read_text(encoding="utf-8").splitlines() if line.strip()]
    answered = {conversation.turns[0].text for conversation in conversations if conversation.turns}
    remaining = [prompt for prompt in prompts if prompt not in answered]
    rng.shuffle(remaining)

    print(__doc__)
    print(f"{len(conversations)} answers saved so far, {len(remaining)} prompts left\n")
    for prompt in remaining:
        try:
            answer = input(f"them> {prompt}\nme>   ").strip()
            if answer == "/quit":
                break
            if answer == "/undo":
                if conversations:
                    conversations.pop()
                    _save(output, conversations)
                    print("(removed last answer)\n")
                continue
            if not answer:
                continue

            turns = [Turn("them", prompt), Turn("me", answer)]
            if rng.random() < args.follow_up_rate:
                follow_up = rng.choice(FOLLOW_UPS)
                second = input(f"them> {follow_up}\nme>   ").strip()
                if second and not second.startswith("/"):
                    turns += [Turn("them", follow_up), Turn("me", second)]
        except EOFError:
            print()
            break

        conversations.append(Conversation(turns=turns, source="interview"))
        _save(output, conversations)
        print()

    words = sum(len(turn.text.split()) for c in conversations for turn in c.turns if turn.role == "me")
    print(f"saved {len(conversations)} answers ({words} of your words) to {output}")


if __name__ == "__main__":
    main()
