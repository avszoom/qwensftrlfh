"""Central configuration: model ids, paths, and training hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
RESULTS = ROOT / "results"
DATA = ROOT / "data"


@dataclass
class Config:
    # --- models ---
    # Start from the BASE model (not -Instruct) so SFT/RL have real headroom.
    base_model: str = "Qwen/Qwen2.5-Coder-0.5B"

    # --- output locations (each stage writes a LoRA adapter here) ---
    sft_adapter: Path = OUTPUTS / "sft"
    dpo_adapter: Path = OUTPUTS / "dpo"
    grpo_adapter: Path = OUTPUTS / "grpo"
    pref_data: Path = DATA / "preference.jsonl"

    # --- LoRA ---
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # --- SFT ---
    sft_dataset: str = "sahil2801/CodeAlpaca-20k"
    sft_max_samples: int = 5000
    sft_epochs: int = 1
    sft_lr: float = 2e-4
    sft_batch_size: int = 2
    sft_grad_accum: int = 8
    max_seq_len: int = 1024

    # --- RL (GRPO=RLVR, DPO=RLHF) ---
    rl_dataset: str = "google-research-datasets/mbpp"
    rl_max_prompts: int = 300
    grpo_lr: float = 1e-5
    grpo_steps: int = 200
    grpo_num_generations: int = 4
    dpo_lr: float = 5e-6
    dpo_epochs: int = 1
    dpo_beta: float = 0.1

    # --- generation / benchmark ---
    gen_max_new_tokens: int = 512
    code_exec_timeout: int = 10

    seed: int = 42


CFG = Config()

for _d in (OUTPUTS, RESULTS, DATA):
    _d.mkdir(parents=True, exist_ok=True)
