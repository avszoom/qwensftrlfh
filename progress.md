# Progress Log

A running log of what we did, in order, so it's easy to pick up later.

## Stage 0 — Project setup ✅
- Cloned empty repo, scaffolded the 3-stage pipeline (`src/`).
- Base model chosen: **Qwen/Qwen2.5-Coder-0.5B** (Base, not Instruct — for headroom).
- Wrote `benchmark.py`, `run_model.py`, `sft_train.py`, `chat_demo.py`,
  `reward.py`, `build_preference_data.py`, `rl_train.py`.
- Added `README.md`, `commands.md`, `architecture.md`, `CONCEPTS.md` (in microgpt).

## Stage 1 — Base model benchmark 🟡 in progress
- Fixed dataset id bug: `openai_humaneval` → `openai/openai_humaneval`
  (new `datasets` lib needs `namespace/name`).
- Command: `python -m src.benchmark --limit 20 --tag base_smoke` then full `--tag base`.
- Confirmed how base eval works: **completion mode** (model autocompletes the function
  signature; no instruction needed). SFT/RL stages use `--chat` mode.
- Baseline score (pass@1): _TODO: fill in after run_

## Stage 2 — SFT (LoRA) 🟡 paused (pipeline validated)
- Goal: teach the base model to **follow instructions** using CodeAlpaca.
- ✅ Pipeline validated: smoke run (16 samples) trained + saved a LoRA adapter OK.
- ✅ Fixed trl 1.x API: `SFTConfig(max_seq_length=...)` → `max_length=...`.
- 🟡 Full run (5000 samples, 313 steps, ~8-9 s/step ≈ 45 min on MPS) was **started then
  stopped by user before completion** — no adapter saved yet. First loss ~2.29, token
  acc ~0.66 (healthy).
- Resume/run full training: `python -m src.sft_train`
- Then verify follows instructions: `python -m src.chat_demo --adapter outputs/sft`
- Then benchmark (chat mode): `python -m src.benchmark --adapter outputs/sft --chat --tag sft`
- SFT score (pass@1): _TODO (after full run completes)_

## Stage 3 — RLVR / RLHF ⬜ not started
- GRPO (RLVR): `python -m src.rl_train --method grpo`
- DPO (RLHF): `python -m src.build_preference_data` then `--method dpo`
- RL score (pass@1): _TODO_

---

## Scoreboard
| Stage | Model | pass@1 | Notes |
|-------|-------|--------|-------|
| 1 | base | _TODO_ | completion mode |
| 2 | + SFT | _TODO_ | chat mode |
| 3 | + RL  | _TODO_ | chat mode |

## Known environment notes
- Python 3.14 venv on macOS (Apple MPS).
- Use `python3 -m venv`; inside the venv `python` works.
- Datasets must be `namespace/name`.
