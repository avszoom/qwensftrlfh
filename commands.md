# Commands — full pipeline

Run from the `qwensftrlfh/` project root with the venv active.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Deliverable 1 — Base model: benchmark + run

```bash
# quick smoke (20 problems) then full HumanEval pass@1 on the BASE model
python -m src.benchmark --limit 20 --tag base_smoke
python -m src.benchmark --tag base

# run the base model on a prompt (completion mode — it autocompletes)
python -m src.run_model --prompt "def is_prime(n):"
```

Baseline score saved to `results/humaneval_base.json`.

---

## Deliverable 2 — Stage 2 SFT: train + verify instruction-following

```bash
# train LoRA SFT (CodeAlpaca)
python -m src.sft_train

# see it FOLLOW INSTRUCTIONS (vs base which rambles)
python -m src.chat_demo --adapter outputs/sft

# benchmark the SFT model (chat/instruction mode)
python -m src.benchmark --adapter outputs/sft --chat --tag sft
```

Compare `results/humaneval_sft.json` to `humaneval_base.json` → expect a jump.

---

## Deliverable 3 — Stage 3 RLVR/RLHF: train + rerun benchmark + verify

### Option A — GRPO (RLVR, reward = passing unit tests)

```bash
python -m src.rl_train --method grpo
python -m src.benchmark --adapter outputs/grpo --chat --tag grpo
python -m src.chat_demo --adapter outputs/grpo
```

### Option B — DPO (RLHF, preference pairs from test execution)

```bash
python -m src.build_preference_data          # sample SFT model, label by tests
python -m src.rl_train --method dpo
python -m src.benchmark --adapter outputs/dpo --chat --tag dpo
python -m src.chat_demo --adapter outputs/dpo
```

---

## Compare all stages

```bash
# the three pass@1 numbers
cat results/humaneval_base.json results/humaneval_sft.json results/humaneval_grpo.json \
  | grep -o '"pass@1": [0-9.]*'
```

Expected trend: **base < sft < grpo/dpo**.

## Iterating cheaply (Mac / limited compute)

```bash
python -m src.benchmark --adapter outputs/sft --chat --limit 30   # subset eval
# edit src/config.py: lower grpo_steps, rl_max_prompts, sft_max_samples
```
