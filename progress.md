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

## Stage 2 — SFT (LoRA) 🟡 in progress
- Goal: teach the base model to **follow instructions** using CodeAlpaca.
- Train: `python -m src.sft_train`
- Verify follows instructions: `python -m src.chat_demo --adapter outputs/sft`
- Benchmark (chat mode): `python -m src.benchmark --adapter outputs/sft --chat --tag sft`
- SFT score (pass@1): _TODO_

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
