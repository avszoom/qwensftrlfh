"""Build on-policy preference pairs for DPO (the RLHF path).

We sample several solutions from the SFT model on MBPP problems, run each against
its unit tests, and form (chosen = a solution that PASSES, rejected = one that
FAILS). The preference label is grounded in test execution, not human opinion.

Usage:
    python -m src.build_preference_data --num-prompts 200 --samples 4
"""

from __future__ import annotations

import argparse
import json

from datasets import load_dataset
from tqdm import tqdm

from .common import build_chat_prompt, generate, load_model_and_tokenizer
from .config import CFG
from .reward import check_mbpp, extract_code


def instruction_for(problem: dict) -> str:
    sig_hint = problem["test_list"][0] if problem["test_list"] else ""
    return (
        f"{problem['text']}\n\n"
        f"Write a single Python function. It must satisfy: {sig_hint}\n"
        "Return only the function definition."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build DPO preference pairs")
    p.add_argument("--num-prompts", type=int, default=CFG.rl_max_prompts)
    p.add_argument("--samples", type=int, default=4, help="candidates per prompt")
    p.add_argument("--adapter", type=str, default=str(CFG.sft_adapter))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model, tokenizer, device = load_model_and_tokenizer(CFG.base_model, args.adapter)

    ds = load_dataset(CFG.rl_dataset, split="train")
    ds = ds.select(range(min(args.num_prompts, len(ds))))

    pairs = []
    for problem in tqdm(ds, desc="preference pairs"):
        instruction = instruction_for(problem)
        prompt = build_chat_prompt(tokenizer, instruction)

        chosen = rejected = None
        for _ in range(args.samples):
            raw = generate(model, tokenizer, device, prompt,
                           CFG.gen_max_new_tokens, temperature=0.8)
            code = extract_code(raw)
            ok = check_mbpp(code, problem["test_list"], CFG.code_exec_timeout)
            if ok and chosen is None:
                chosen = raw
            elif not ok and rejected is None:
                rejected = raw
            if chosen and rejected:
                break

        if chosen and rejected:
            pairs.append({"prompt": instruction, "chosen": chosen, "rejected": rejected})

    CFG.pref_data.parent.mkdir(parents=True, exist_ok=True)
    with CFG.pref_data.open("w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(pairs)} preference pairs -> {CFG.pref_data}")
    if not pairs:
        print("WARNING: no pairs produced (model passed all or none). "
              "Try more prompts/samples or train SFT longer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
