# Progress Log

A running log of what we did, in order, so it's easy to pick up later.

## Stage 0 — Project setup ✅
- Cloned empty repo, scaffolded the 3-stage pipeline (`src/`).
- Base model chosen: **Qwen/Qwen2.5-Coder-0.5B** (Base, not Instruct — for headroom).
- Wrote `benchmark.py`, `run_model.py`, `sft_train.py`, `chat_demo.py`,
  `reward.py`, `build_preference_data.py`, `rl_train.py`.
- Added `README.md`, `commands.md`, `architecture.md`, `CONCEPTS.md` (in microgpt).

## Stage 1 — Base model benchmark 🟡 smoke done, full run TODO
- Fixed dataset id bug: `openai_humaneval` → `openai/openai_humaneval`
  (new `datasets` lib needs `namespace/name`).
- Confirmed how base eval works: **completion mode** (model autocompletes the function
  signature; no instruction needed). SFT/RL stages use `--chat` mode.
- ✅ Smoke (20 problems): **pass@1 = 0.40 (8/20)**.
- ⬜ Full 164-problem baseline still TODO: `python -m src.benchmark --tag base`.

## Stage 2 — SFT (LoRA) ✅ complete
- Goal: teach the base model to **follow instructions** using CodeAlpaca.
- ✅ Fixed trl 1.x API: `SFTConfig(max_seq_length=...)` → `max_length=...`.
- ✅ Full run: 5000 samples, 313 steps, ~42 min on MPS. Loss **2.29 → 1.12**,
  token acc ~0.75. Adapter saved to `outputs/sft/` (~35 MB).
- ✅ `chat_demo` confirms it now **follows instructions** with correct code
  (Fibonacci, palindrome, second-largest all correct) — the base just autocompleted.
- ⚠️ Benchmark (20-problem subset, chat): **pass@1 = 0.20 (4/20)** — DROPPED vs base.
  See "Finding" below. (SFT's win is *behavior*, not benchmark.)

## Stage 3 — RLVR (GRPO) 🟡 experiments done; signal weak (0.5B near capability floor)
GRPO smoke runs (20 steps each, MBPP, on SFT model). Key metric: `frac_reward_zero_std`
(fraction of groups where all attempts score the same = NO learning signal; lower is better).

| Experiment | frac_reward_zero_std | Verdict |
|---|---|---|
| sparse reward (1/0), 4 gens, temp 1.0 | 0.8–1.0 | dead (all-fail groups) |
| + temperature 1.2 | 0.8–1.0 | no help |
| + 8 generations | 0.8–1.0 | no help |
| + **dense reward** (fraction of asserts passed) | **→ 0.6** | **best — real signal** |

- GRPO smoke (default) benchmark: pass@1 = **0.05 (1/20)** — 20 noisy steps added noise, not learning.
- **Lesson:** the bottleneck was **sparse reward + a base too weak to ever fully pass**. Dense
  (partial-credit) reward is the correct fix and gave the best signal, but the 0.5B model still
  passes only ~6% of test cases — it's near the floor RLVR needs.
- Timing on MPS: ~13 s/step (4 gens) to ~29 s/step (8 gens). Full 200 steps ≈ 45–90 min (feasible).
- Code: `reward.py:check_mbpp_fraction` (dense), `rl_train.py` now uses it; added CLI knobs
  `--steps --temperature --num-generations --num-prompts --max-completion-len`.
- Next options: (a) longer dense run (100+ steps) to accumulate signal; (b) curate easier MBPP
  subset to raise reward; (c) accept as documented finding (RLVR needs a capable base).

### (original Stage 3 plan)
- GRPO (RLVR): `python -m src.rl_train --method grpo` — the strong path (reward = tests pass),
  but **slow on MPS**: ~75-120 s/step → full 200 steps ≈ **4-7 hr**. Smoke (20 steps) ≈ 35 min.
- DPO (RLHF): `python -m src.build_preference_data` then `--method dpo` — much cheaper
  (~30-45 min), legitimate Stage-3 result.
- RL score (pass@1): _TODO_

---

## Scoreboard (20-problem subset — noisy; full 164 run still TODO)
| Stage | Model | pass@1 | Notes |
|-------|-------|--------|-------|
| 1 | base | 0.40 (8/20) | completion mode (HumanEval's native format) |
| 2 | + SFT | 0.20 (4/20) | chat mode — **dropped** (see finding) |
| 3 | + RL  | _TODO_ | chat mode |

### Finding: SFT improved behavior but LOWERED the benchmark
- SFT (CodeAlpaca) taught instruction-following (chat_demo confirms correct answers),
  but pass@1 fell 0.40 → 0.20. Reasons:
  1. Base is a code-completion specialist; completion mode was its strength.
  2. CodeAlpaca is generic/older — taught *format*, not better *code* (traded skill for behavior).
  3. Chat mode + over-generation (trailing comments) makes exact tests fail.
  4. 20 problems is noisy.
- Lesson: SFT = behavior, not capability. **RLVR (Stage 3) is what should raise pass@1**
  because it rewards code that passes tests.
- Remedies to try: better SFT data (Magicoder-OSS-Instruct), tighter stop/extraction,
  run full 164, then GRPO/RLVR.

## Next up
1. (optional) Full 164-problem benchmark for clean base + SFT numbers.
2. Stage 3: start with a **GRPO smoke (20 steps, ~35 min)** to see reward climb,
   or go the cheaper **DPO** route. Then benchmark `outputs/grpo`/`outputs/dpo` for score C.

## Concepts captured
- `learning.md` — training stages, SFT, RLHF/RLVR, how weights/backprop/nudging work,
  LoRA, RAG vs fine-tuning, serving (llama.cpp / ollama / vLLM).
- microgpt repo `CONCEPTS.md` — model internals (attention, multi-head, context window, KV cache).

## Known environment notes
- Python 3.14 venv on macOS (Apple MPS).
- Use `python3 -m venv`; inside the venv `python` works.
- Datasets must be `namespace/name`.
- Background jobs die if the Claude session/CLI exits — keep it open for long runs.
