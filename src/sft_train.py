"""Stage 2 - Supervised Fine-Tuning (SFT) with LoRA.

Teaches the BASE model the (instruction -> answer) behavior using a code
instruction dataset, so it starts *responding* instead of autocompleting.

Usage:
    python -m src.sft_train
    python -m src.sft_train --max-samples 2000 --epochs 1
"""

from __future__ import annotations

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from .common import get_device, get_dtype, logger
from .config import CFG


def format_example(tokenizer, example: dict) -> str:
    """Render a CodeAlpaca row into a single chat-formatted training string."""
    instruction = example.get("instruction", "").strip()
    inp = example.get("input", "").strip()
    if inp:
        instruction = f"{instruction}\n\n{inp}"
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": example.get("output", "").strip()},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT with LoRA")
    p.add_argument("--max-samples", type=int, default=CFG.sft_max_samples)
    p.add_argument("--epochs", type=int, default=CFG.sft_epochs)
    p.add_argument("--lr", type=float, default=CFG.sft_lr)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = get_device()
    dtype = get_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(CFG.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading dataset %s", CFG.sft_dataset)
    ds = load_dataset(CFG.sft_dataset, split="train")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    ds = ds.map(lambda e: {"text": format_example(tokenizer, e)},
                remove_columns=ds.column_names)

    model = AutoModelForCausalLM.from_pretrained(CFG.base_model, torch_dtype=dtype)
    model.to(device)

    lora = LoraConfig(
        r=CFG.lora_r, lora_alpha=CFG.lora_alpha, lora_dropout=CFG.lora_dropout,
        target_modules=list(CFG.lora_targets), task_type="CAUSAL_LM",
    )

    sft_args = SFTConfig(
        output_dir=str(CFG.sft_adapter),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=CFG.sft_batch_size,
        gradient_accumulation_steps=CFG.sft_grad_accum,
        learning_rate=args.lr,
        max_length=CFG.max_seq_len,
        dataset_text_field="text",
        logging_steps=10,
        save_strategy="no",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        report_to="none",
        seed=CFG.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds,
        peft_config=lora,
        processing_class=tokenizer,
    )
    logger.info("Starting SFT ...")
    trainer.train()
    trainer.save_model(str(CFG.sft_adapter))
    tokenizer.save_pretrained(str(CFG.sft_adapter))
    logger.info("Saved SFT adapter -> %s", CFG.sft_adapter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
