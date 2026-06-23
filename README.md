# Qwen Coder: SFT → RLVR/RLHF

Take a small **base** code model (`Qwen/Qwen2.5-Coder-0.5B`) and walk it through the
full post-training pipeline, **measuring HumanEval pass@1 after each stage** so the
gains are visible and benchmark-backed.

```
Base model        →  benchmark            (score A)
  + SFT (Stage 2) →  benchmark + demo     (score B,  follows instructions)
  + RL  (Stage 3) →  benchmark + demo     (score C,  GRPO=RLVR / DPO=RLHF)
```

> **Why start from `-Base`, not `-Instruct`?** The Instruct model is already SFT'd and
> aligned, so there is little headroom to demonstrate. Starting from Base gives a clear
> rising curve across stages. See the project notes for the full reasoning.

## The three deliverables

| Stage | Train | Benchmark | Demo / "see it work" |
|---|---|---|---|
| **1. Base** | — | `src.benchmark` | `src.run_model` |
| **2. SFT** | `src.sft_train` | `src.benchmark --adapter outputs/sft --chat` | `src.chat_demo --adapter outputs/sft` |
| **3. RLVR / RLHF** | `src.rl_train --method grpo` (RLVR) / `--method dpo` (RLHF) | `src.benchmark --adapter outputs/grpo --chat` | `src.chat_demo --adapter outputs/grpo` |

## Concepts

- **Stage 1 — Pretraining (already done by Qwen):** the base model *knows* code but
  only autocompletes; it doesn't follow instructions.
- **Stage 2 — SFT:** LoRA fine-tune on `(instruction → answer)` pairs (CodeAlpaca) so it
  learns to *respond* to requests. Biggest single benchmark jump on a small model.
- **Stage 3 — Alignment with execution-grounded rewards:**
  - **GRPO (RLVR):** generate code → **run it against unit tests** → reward passing.
    The reward is ground truth, not opinion. This is the modern coding-model technique.
  - **DPO (RLHF):** build `(chosen, rejected)` pairs by test execution, then preference-tune.
    The stable, practical form of RLHF.

## Project structure

```
qwensftrlfh/
├── src/
│   ├── config.py                # model id, paths, hyperparameters
│   ├── common.py                # device/dtype, model loading, generation
│   ├── reward.py                # sandboxed code-execution reward (RLVR signal)
│   ├── benchmark.py             # HumanEval pass@1  (Deliverable 1 + 3)
│   ├── run_model.py             # run a model on a prompt (Deliverable 1)
│   ├── sft_train.py             # Stage 2: LoRA SFT
│   ├── chat_demo.py             # instruction-following demo (Deliverable 2 + 3)
│   ├── build_preference_data.py # on-policy pairs for DPO
│   └── rl_train.py              # Stage 3: GRPO (RLVR) / DPO (RLHF)
├── data/        # datasets + preference pairs (gitignored)
├── outputs/     # LoRA adapters per stage (gitignored)
├── results/     # benchmark JSON (gitignored)
├── requirements.txt
└── commands.md  # exact commands for the full pipeline
```

## Setup

```bash
cd qwensftrlfh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

See `commands.md` for the full run sequence.

## Compute notes (honest)

- Tested target: Apple Silicon (**MPS**) or CUDA; auto-detected.
- **SFT** with LoRA runs fine on a Mac.
- **DPO** is light and stable — the recommended Stage-3 path on a Mac.
- **GRPO (RLVR)** is the most exciting but **compute-heavy** (it generates many samples
  *and* executes them every step). It runs on MPS but is slow; a CUDA GPU is strongly
  recommended for a full run. Use `--limit` on the benchmark and small `grpo_steps` to
  iterate.
- `bitsandbytes`/4-bit QLoRA is **CUDA-only** and intentionally not used here, so the
  project runs on Mac.

## Security note

`reward.py` executes model-generated code (standard for HumanEval/MBPP). It runs each
candidate in a subprocess with a timeout. Only run on a trusted machine.
