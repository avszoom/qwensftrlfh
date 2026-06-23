"""Shared helpers: device/dtype selection, model loading, generation."""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("qwen")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(model_id: str, adapter: str | None = None):
    """Load a base model (optionally with a trained LoRA adapter) + tokenizer."""
    device = get_device()
    dtype = get_dtype(device)
    logger.info("Loading %s on %s (%s)", model_id, device.type, dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)

    if adapter:
        from peft import PeftModel

        logger.info("Attaching LoRA adapter: %s", adapter)
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()

    model.to(device)
    model.eval()
    return model, tokenizer, device


@torch.no_grad()
def generate(model, tokenizer, device, text: str, max_new_tokens: int,
             temperature: float = 0.0) -> str:
    """Generate a continuation. temperature=0 -> greedy."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    do_sample = temperature > 0
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=0.95 if do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)


def build_chat_prompt(tokenizer, instruction: str) -> str:
    """Wrap an instruction in the model's chat template."""
    messages = [{"role": "user", "content": instruction}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
