"""HumanEval pass@1 benchmark.

Works on any stage: base model, SFT adapter, or RL adapter. Two prompting modes:
  - completion mode (default): feed the raw function signature, let the model
    complete it. Correct for BASE models.
  - chat mode (--chat): wrap the problem as an instruction using the chat
    template. Use for SFT / instruct models.

Usage:
    python -m src.benchmark                          # base model, completion mode
    python -m src.benchmark --adapter outputs/sft --chat
    python -m src.benchmark --limit 20               # quick subset
"""

from __future__ import annotations

import argparse
import json

from datasets import load_dataset
from tqdm import tqdm

from .common import (build_chat_prompt, generate, load_model_and_tokenizer, logger)
from .config import CFG, RESULTS
from .reward import check_humaneval, extract_code

STOP_SEQUENCES = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\n@"]


def truncate_completion(text: str) -> str:
    """Cut a base-model completion at the first sign it left the function body."""
    cut = len(text)
    for stop in STOP_SEQUENCES:
        idx = text.find(stop)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HumanEval pass@1 benchmark")
    p.add_argument("--adapter", type=str, default=None, help="path to a LoRA adapter")
    p.add_argument("--chat", action="store_true", help="use chat/instruction prompting")
    p.add_argument("--limit", type=int, default=None, help="evaluate only N problems")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--tag", type=str, default=None, help="label for the results file")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model, tokenizer, device = load_model_and_tokenizer(CFG.base_model, args.adapter)

    ds = load_dataset("openai/openai_humaneval", split="test")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    passed = 0
    records = []
    for problem in tqdm(ds, desc="HumanEval"):
        prompt = problem["prompt"]

        if args.chat:
            instruction = (
                "Complete the following Python function. Return only the full "
                f"function implementation.\n\n```python\n{prompt}\n```"
            )
            raw = generate(model, tokenizer, device, build_chat_prompt(tokenizer, instruction),
                           CFG.gen_max_new_tokens, args.temperature)
            program = extract_code(raw)
        else:
            raw = generate(model, tokenizer, device, prompt,
                           CFG.gen_max_new_tokens, args.temperature)
            program = prompt + truncate_completion(raw)

        ok = check_humaneval(program, problem["test"], problem["entry_point"],
                             CFG.code_exec_timeout)
        passed += int(ok)
        records.append({"task_id": problem["task_id"], "passed": ok})

    total = len(ds)
    pass_at_1 = passed / total if total else 0.0
    tag = args.tag or (args.adapter.replace("/", "_") if args.adapter else "base")
    out = RESULTS / f"humaneval_{tag}.json"
    out.write_text(json.dumps(
        {"tag": tag, "chat": args.chat, "passed": passed, "total": total,
         "pass@1": pass_at_1, "records": records}, indent=2))

    logger.info("=" * 50)
    logger.info("MODEL: %s%s", tag, " (chat)" if args.chat else " (completion)")
    logger.info("pass@1 = %.3f  (%d/%d)", pass_at_1, passed, total)
    logger.info("results -> %s", out)
    logger.info("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
