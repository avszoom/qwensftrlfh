"""See the model follow instructions (Deliverable 2 / verify Stage 3).

Runs a fixed set of coding instructions through the model so you can eyeball
whether it actually *responds to requests* (vs. the base model that just
autocompletes). Point --adapter at the SFT, DPO, or GRPO output to compare.

Usage:
    python -m src.chat_demo                       # base model (will ramble)
    python -m src.chat_demo --adapter outputs/sft # should now follow instructions
    python -m src.chat_demo --adapter outputs/grpo --prompt "Write a binary search"
"""

from __future__ import annotations

import argparse

from .common import build_chat_prompt, generate, load_model_and_tokenizer
from .config import CFG

DEFAULT_PROMPTS = [
    "Write a Python function that returns the nth Fibonacci number.",
    "Write a function to check whether a string is a palindrome.",
    "Given a list of integers, return the second largest value.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Instruction-following demo")
    p.add_argument("--adapter", type=str, default=None)
    p.add_argument("--prompt", type=str, default=None, help="single custom instruction")
    p.add_argument("--max-new-tokens", type=int, default=CFG.gen_max_new_tokens)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model, tokenizer, device = load_model_and_tokenizer(CFG.base_model, args.adapter)
    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS

    for i, instruction in enumerate(prompts, 1):
        text = build_chat_prompt(tokenizer, instruction)
        response = generate(model, tokenizer, device, text, args.max_new_tokens, 0.0)
        print(f"\n{'#' * 60}\n[{i}] INSTRUCTION: {instruction}\n{'-' * 60}")
        print(response.strip())
    print(f"\n{'#' * 60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
