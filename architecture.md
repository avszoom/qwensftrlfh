# Architecture & How We Did It

This file explains the whole Qwen coding project in simple words, so you can pick it up
again later. Read top to bottom.

---

## What we are building

We take a small **base** code model and make it better in stages. After every stage we
run the same coding test (HumanEval) to see if the score went up.

```
Base model            →  test  →  score A
  + Stage 2 (SFT)     →  test  →  score B   (should be higher)
  + Stage 3 (RL)      →  test  →  score C   (should be higher still)
```

The model is **Qwen2.5-Coder-0.5B** (the "Base" version, ~0.5 billion parameters).

---

## Why the Base model and not the Instruct model

Qwen comes in two versions:
- **Base**: only pretrained. It knows code but does not follow instructions well.
- **Instruct**: already tuned by Qwen's team (Stages 2 and 3 done).

We start from **Base** on purpose. If we started from Instruct, it is already polished, so
our training would add little and the score might even drop. Base gives us room to show a
clear improvement at each stage.

---

## The three stages (in simple words)

### Stage 1 — Pretraining (already done by Qwen)
The model has read tons of code. It can **autocomplete** code, but it does not know it is
supposed to answer your questions. This is our starting point.

### Stage 2 — SFT (Supervised Fine-Tuning)
We show the model many examples of **"instruction → correct answer"** (a code instruction
dataset). It learns the *behavior* of responding to a request.
- This is usually the **biggest jump** in score for a small model.
- We use **LoRA**, which trains only a small add-on instead of the whole model (fast, cheap).

### Stage 3 — Alignment with real rewards
Now we teach it to give **good, correct** answers. Two methods, both grounded in running
the code against tests:

- **GRPO (RLVR = RL from Verifiable Rewards):** the model writes code, we **run it against
  unit tests**, and reward it when the tests pass. The reward is the truth ("did it work"),
  not an opinion. This is how modern coding models get strong.
- **DPO (RLHF):** we build pairs of (good answer, bad answer) by running tests, then train
  the model to prefer the good one. This is the lighter, simpler, stable option.

Important idea: Stage 3 can only **sharpen skills the model already has**. It cannot add
knowledge that pretraining never gave it. That is why the base model must be capable enough
to begin with.

---

## How the code is organized

```
src/
├── config.py                # all settings in one place (model id, paths, knobs)
├── common.py                # picks the device (GPU/CPU), loads the model, generates text
├── reward.py                # runs generated code against tests = the reward signal
├── benchmark.py             # the HumanEval test, gives a pass@1 score
├── run_model.py             # run the model on one prompt
├── sft_train.py             # Stage 2 training (SFT + LoRA)
├── chat_demo.py             # check the model follows instructions
├── build_preference_data.py # makes (good, bad) pairs for DPO
└── rl_train.py              # Stage 3 training (GRPO or DPO)
```

What the key pieces do, in one line each:
- **benchmark.py** — asks the model to solve 164 coding problems, runs the answers, counts
  how many pass. The score is **pass@1** (fraction solved on the first try).
- **reward.py** — safely runs model code in a separate process with a time limit, returns
  pass/fail. This is the "verifiable reward".
- **LoRA** — a small trainable add-on; we keep one per stage in `outputs/`.

---

## The deliverables (what we promised)

| Stage | Train it | Score it | See it work |
|-------|----------|----------|-------------|
| 1. Base | (nothing) | `benchmark.py` | `run_model.py` |
| 2. SFT  | `sft_train.py` | `benchmark.py --adapter outputs/sft --chat` | `chat_demo.py --adapter outputs/sft` |
| 3. RL   | `rl_train.py --method grpo` or `--method dpo` | `benchmark.py --adapter outputs/grpo --chat` | `chat_demo.py --adapter outputs/grpo` |

---

## Commands (the full run)

Setup:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Stage 1 — base model:
```bash
python -m src.benchmark --limit 20 --tag base_smoke   # quick check (20 problems)
python -m src.benchmark --tag base                     # full score A
python -m src.run_model --prompt "def is_prime(n):"    # watch it autocomplete
```

Stage 2 — SFT:
```bash
python -m src.sft_train                                 # train
python -m src.chat_demo --adapter outputs/sft           # now it follows instructions
python -m src.benchmark --adapter outputs/sft --chat --tag sft   # score B
```

Stage 3 — RLVR (GRPO):
```bash
python -m src.rl_train --method grpo                    # train with test rewards
python -m src.benchmark --adapter outputs/grpo --chat --tag grpo # score C
python -m src.chat_demo --adapter outputs/grpo          # check it still works
```

Stage 3 — RLHF (DPO), alternative:
```bash
python -m src.build_preference_data                     # make (good, bad) pairs
python -m src.rl_train --method dpo
python -m src.benchmark --adapter outputs/dpo --chat --tag dpo
```

Compare the three scores:
```bash
cat results/humaneval_base.json results/humaneval_sft.json results/humaneval_grpo.json \
  | grep -o '"pass@1": [0-9.]*'
```

---

## What to expect

- Score order should be **base < sft < grpo/dpo**.
- The **SFT** step usually gives the biggest jump. The **RL** step adds a smaller, real gain.
- The gain is only visible if we start from **Base** and use **test-based rewards**.

---

## Compute notes (be realistic)

- The code auto-picks the best device: NVIDIA GPU → Apple GPU (MPS) → CPU.
- **SFT** and **DPO** run fine on a Mac.
- **GRPO (RLVR)** is the heaviest: every step it generates several answers *and* runs them.
  It works on a Mac but is slow — a real NVIDIA GPU is much better for the full run.
- To iterate quickly, use `--limit` on the benchmark and lower the step counts in
  `src/config.py`.

---

## One-line summary

> Start from a base code model → teach it to follow instructions (SFT) → teach it to write
> code that passes tests (RLVR/RLHF) → measure HumanEval after each step to prove it got
> better.
