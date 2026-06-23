"""Run any stage's model on a single prompt (Deliverable 1: 'script to run the model').

Completion mode (base model) by default; --chat for instruction-tuned stages.

Usage:
    python -m src.run_model --prompt "def fibonacci(n):"
    python -m src.run_model --adapter outputs/sft --chat --prompt "Write a function to check if a number is prime"
"""

from __future__ import annotations

import argparse

from .common import build_chat_prompt, generate, load_model_and_tokenizer
from .config import CFG


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Qwen coding model")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--adapter", type=str, default=None)
    p.add_argument("--chat", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=CFG.gen_max_new_tokens)
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model, tokenizer, device = load_model_and_tokenizer(CFG.base_model, args.adapter)

    text = build_chat_prompt(tokenizer, args.prompt) if args.chat else args.prompt
    output = generate(model, tokenizer, device, text, args.max_new_tokens, args.temperature)

    print("\n" + "=" * 60)
    if not args.chat:
        print(args.prompt, end="")
    print(output)
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
