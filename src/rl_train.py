"""Stage 3 - alignment with execution-grounded rewards.

Two methods (pick with --method):

  grpo  (RLVR): the model generates code, we RUN it against unit tests, and reward
                passing. This is RL from Verifiable Rewards - the technique behind
                modern coding models. Starts from the SFT adapter.

  dpo   (RLHF): Direct Preference Optimization on the (chosen, rejected) pairs built
                by build_preference_data.py. The practical, stable form of RLHF.

Usage:
    python -m src.rl_train --method grpo
    python -m src.rl_train --method dpo
"""

from __future__ import annotations

import argparse
import json

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import build_chat_prompt, get_device, get_dtype, logger
from .config import CFG
from .reward import check_mbpp, check_mbpp_fraction, extract_code


def load_sft_base():
    """Load base model with the SFT adapter merged in, as the RL starting point."""
    device = get_device()
    dtype = get_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(CFG.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(CFG.base_model, torch_dtype=dtype)
    if CFG.sft_adapter.exists():
        logger.info("Merging SFT adapter as RL starting point")
        model = PeftModel.from_pretrained(model, str(CFG.sft_adapter)).merge_and_unload()
    else:
        logger.warning("No SFT adapter found; starting RL from base model")
    model.to(device)
    return model, tokenizer, device, dtype


def lora_config() -> LoraConfig:
    return LoraConfig(
        r=CFG.lora_r, lora_alpha=CFG.lora_alpha, lora_dropout=CFG.lora_dropout,
        target_modules=list(CFG.lora_targets), task_type="CAUSAL_LM",
    )


# --------------------------------------------------------------------- GRPO (RLVR)
def train_grpo(steps=None, max_completion_len=None, num_prompts=None,
               temperature=None, num_generations=None) -> None:
    from trl import GRPOConfig, GRPOTrainer

    steps = steps or CFG.grpo_steps
    max_completion_len = max_completion_len or CFG.gen_max_new_tokens
    num_prompts = num_prompts or CFG.rl_max_prompts
    temperature = temperature or 1.0
    num_generations = num_generations or CFG.grpo_num_generations

    model, tokenizer, device, dtype = load_sft_base()

    raw = load_dataset(CFG.rl_dataset, split="train")
    raw = raw.select(range(min(num_prompts, len(raw))))

    def to_prompt(p):
        hint = p["test_list"][0] if p["test_list"] else ""
        instruction = (f"{p['text']}\n\nWrite a single Python function satisfying: "
                       f"{hint}\nReturn only the function definition.")
        return {"prompt": build_chat_prompt(tokenizer, instruction),
                "test_list": p["test_list"]}

    ds = raw.map(to_prompt, remove_columns=raw.column_names)

    def reward_funcs(completions, test_list=None, **kwargs):
        """Dense reward: fraction of unit tests that pass (0.0-1.0).

        Partial credit gives a learning signal even when no attempt is fully
        correct, avoiding all-zero reward groups (sparse-reward problem).
        """
        rewards = []
        for comp, tests in zip(completions, test_list):
            code = extract_code(comp)
            rewards.append(check_mbpp_fraction(code, tests, CFG.code_exec_timeout))
        return rewards

    args = GRPOConfig(
        output_dir=str(CFG.grpo_adapter),
        learning_rate=CFG.grpo_lr,
        per_device_train_batch_size=num_generations,
        num_generations=num_generations,
        temperature=temperature,
        max_completion_length=max_completion_len,
        max_steps=steps,
        logging_steps=5,
        save_strategy="no",
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        report_to="none",
        seed=CFG.seed,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=args,
        train_dataset=ds,
        peft_config=lora_config(),
        processing_class=tokenizer,
    )
    logger.info("Starting GRPO (RLVR) ...")
    trainer.train()
    trainer.save_model(str(CFG.grpo_adapter))
    tokenizer.save_pretrained(str(CFG.grpo_adapter))
    logger.info("Saved GRPO adapter -> %s", CFG.grpo_adapter)


# ---------------------------------------------------------------------- DPO (RLHF)
def train_dpo() -> None:
    from trl import DPOConfig, DPOTrainer

    if not CFG.pref_data.exists():
        raise FileNotFoundError(
            f"{CFG.pref_data} missing. Run: python -m src.build_preference_data"
        )

    model, tokenizer, device, dtype = load_sft_base()

    rows = [json.loads(l) for l in CFG.pref_data.read_text().splitlines() if l.strip()]
    if not rows:
        raise ValueError("Preference dataset is empty.")
    ds = Dataset.from_list(rows)

    args = DPOConfig(
        output_dir=str(CFG.dpo_adapter),
        learning_rate=CFG.dpo_lr,
        num_train_epochs=CFG.dpo_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        beta=CFG.dpo_beta,
        max_length=CFG.max_seq_len,
        max_prompt_length=CFG.max_seq_len // 2,
        logging_steps=10,
        save_strategy="no",
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        report_to="none",
        seed=CFG.seed,
    )

    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora_config(),
    )
    logger.info("Starting DPO (RLHF) on %d pairs ...", len(rows))
    trainer.train()
    trainer.save_model(str(CFG.dpo_adapter))
    tokenizer.save_pretrained(str(CFG.dpo_adapter))
    logger.info("Saved DPO adapter -> %s", CFG.dpo_adapter)


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 3: GRPO (RLVR) or DPO (RLHF)")
    p.add_argument("--method", choices=["grpo", "dpo"], required=True)
    p.add_argument("--steps", type=int, default=None, help="GRPO: override max steps")
    p.add_argument("--max-completion-len", type=int, default=None, help="GRPO: cap generated tokens")
    p.add_argument("--num-prompts", type=int, default=None, help="GRPO: limit dataset prompts")
    p.add_argument("--temperature", type=float, default=None, help="GRPO: sampling temperature")
    p.add_argument("--num-generations", type=int, default=None, help="GRPO: attempts per prompt")
    args = p.parse_args()
    if args.method == "grpo":
        train_grpo(args.steps, args.max_completion_len, args.num_prompts,
                   args.temperature, args.num_generations)
    else:
        train_dpo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
