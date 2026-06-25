# Learning Notes — Training, Alignment & Serving

Plain-English notes from our discussions. Written so you can re-read later and quickly
remember not just *what* but *why*. (Model internals — attention, KV cache, context window —
are in the MicroGPT repo's `CONCEPTS.md`.)

---

## 1. The three training stages

A chatbot like ChatGPT is not one model — it's a base model with two more training stages
on top.

| Stage | Full name | What it improves | Simple description |
|-------|-----------|------------------|--------------------|
| 1 | Pretraining | **Knowledge** | Read tons of text, learn to predict the next token. Result: a fluent *autocompleter* that doesn't know it should answer you. |
| 2 | **SFT** (Supervised Fine-Tuning) | **Behavior** | Show it many `(instruction → correct answer)` examples. It learns to *respond to requests* instead of just continuing text. |
| 3 | RLHF / RLVR | **Quality** | Teach it to give *good, correct* answers, not just any answer. |

Key rule: **Stage 1 sets the capability ceiling. Stages 2–3 can only surface and sharpen
skills that pretraining already put there — they cannot add new knowledge.** That is why
you start from a capable base model.

A tiny model (like our 14M MicroGPT) can't reason no matter the training — reasoning needs
**scale**.

---

## 2. SFT — Supervised Fine-Tuning

- *Supervised* = we show it the correct answer (labeled examples).
- *Fine-tuning* = adjust an already-trained model a bit, not train from scratch.
- It teaches the **format/behavior** of answering. Usually the **biggest single benchmark
  jump** for a small model.
- In our project: trained on CodeAlpaca so Qwen-Base learns to follow coding instructions.

---

## 3. RLHF vs RLVR (Stage 3)

After SFT the model follows instructions but its answers aren't always *good*. Stage 3 fixes
quality by making the model **prefer good answers over bad ones**.

### RLHF — Reinforcement Learning from Human Feedback
- Show the model's answers to a **human** (or a reward model trained on human ratings).
- Human says "answer A is better than B."
- Model is nudged to produce more A-like answers.
- Reward = **human opinion**. Great for fuzzy things: helpfulness, tone, safety, style.
- Weakness: slow, costly, subjective; "looks good" ≠ "is correct".

### RLVR — Reinforcement Learning from Verifiable Rewards
- Instead of asking a human, **automatically CHECK if the answer is correct**.
- For code: **does it pass the unit tests?** Reward = 1 if pass, 0 if fail.
- Reward = **ground truth, not opinion**. This is the strong technique for **math & code**.
- In our project: `reward.py` runs the model's code against tests = the RLVR signal.

| | RLHF | RLVR |
|---|------|------|
| Who judges? | a human (opinion) | a program (runs/checks the answer) |
| Best for | tone, helpfulness, safety | math, code, logic (anything checkable) |
| Weakness | slow, subjective | only works where a checker exists |

**Why RL at all (vs just more SFT)?** SFT only shows the *one* right answer and says "copy
it." RL lets the model **generate its own attempts and get scored**, so it learns from its
own mistakes and can discover solutions not in any example.

Algorithms you'll see: **PPO, GRPO** (RL-style), **DPO** (a simpler, stable preference
method). We use **GRPO** for RLVR and **DPO** for the RLHF path.

---

## 4. Do these change the model's weights? Yes — all of them do.

Every kind of training does the **same mechanical thing: nudge the weights** (embeddings,
Query/Key/Value matrices, attention output matrix, MLP, LayerNorm scales). "Learning" *is*
"changing weights." Only the **score that guides the nudge** differs:

```
Pretraining : nudge weights → predict next token in raw text
SFT         : nudge weights → reproduce the correct answer
RLHF        : nudge weights → produce answers humans prefer
RLVR        : nudge weights → produce answers that pass the test
```

**How a weight changes (the 3-beat loop, same for all):**
1. **Try** — model produces output with current weights.
2. **Score** — measure how good/bad (loss or reward).
3. **Nudge** — for every weight, figure out "did raising it help or hurt?" and shift it a
   tiny step the better way. (This is *backpropagation + gradient descent*.)

Repeat thousands of times; tiny nudges add up.

The only difference in effort: SFT has the answer ready (compare → nudge). RL must first
**generate an answer, then score it** (run tests / human rank), then nudge — an extra loop.

---

## 5. LoRA — train a small add-on instead of the whole model

The model has ~500M weights. Nudging all of them is expensive and risky on a laptop. **LoRA
(Low-Rank Adaptation):**

> Freeze the original weights. Add a small set of **new** weights next to the Q/K/V (and
> MLP) matrices, and train **only those**.

```
original matrices : FROZEN (never change)
   +
small LoRA add-on : TRAINED (only this is nudged)   → saved as a ~35MB adapter file
```

- Big model untouched; only the tiny adapter learns. Fits on a Mac.
- At runtime, `original + LoRA add-on` behaves like a fully fine-tuned model.
- Each stage produces its own small adapter (`outputs/sft`, `outputs/grpo`, …).

So: RLVR/RLHF **do** change weights — in our setup they change the **LoRA add-on** attached
to Q/K/V and MLP, while the original Qwen weights stay frozen.

---

## 6. Fine-tuning vs RAG — the big practical lesson

**Adding weights (LoRA) ≠ reliably adding facts.** Fine-tuning is great at *skills/style*,
poor at *facts*.

| Want the model to… | Use |
|---|---|
| Answer in your firm's tone / format / JSON schema | **Fine-tune (LoRA)** |
| Speak a domain's style; follow your policies | **Fine-tune (LoRA)** |
| Know specific facts ("Policy Z says…", "X is on team Y") | **RAG** |
| Use info that changes often | **RAG** |

**Why fine-tuning is bad at facts:** facts don't stick reliably (model learns the *style* of
your docs then **hallucinates** specifics); it can forget other skills; you can't easily
update a fact; and there's no source/citation to verify.

### RAG — Retrieval-Augmented Generation
Keep documents in a searchable (vector) database; fetch the relevant pieces at question time
and put them in the prompt:
```
docs → chunks → vector DB
question → search DB → top relevant chunks → "Using this context: <chunks>, answer: ..."
→ model answers FROM the provided text
```
RAG is **accurate, updatable, auditable (citations), cheap (no training), secure**.

**Mental model:**
> Fine-tuning changes how the model *thinks and talks*. RAG changes what it *knows right now*.
> School (learn a skill) vs. an open reference book on the right page (look up a fact).

**Best real-world answer is usually BOTH:** RAG for the facts, optional LoRA for the
tone/format. Most "expert on our internal knowledge" products are **RAG-first**.

---

## 7. Serving engines — running models efficiently

Training uses Python + PyTorch with 16/32-bit floats (heavy). *Running* a model (inference)
on normal hardware needs optimization.

### llama.cpp (engine, C/C++ — library `ggml`)
Runs the same forward pass (tokenize → layers → sample) but small & fast:
- **Quantization** (the big idea): store each weight in fewer bits. 16-bit → 4-bit cuts a
  7B model ~14GB → ~4GB and speeds it up (less data to move). Tiny accuracy loss.
  Names like `Q4_K_M` = bits + scheme. **This is what makes a datacenter model fit a laptop.**
- **GGUF** = one self-describing file (quantized weights + architecture + tokenizer).
- **Backends**: CPU (SIMD), Mac (Metal), NVIDIA (CUDA); can **split layers across CPU/GPU**.
- Uses **mmap** (load weights on demand) and a **KV cache** (reuse past tokens' Key/Value).

### ollama (manager, built ON llama.cpp) = "Docker for models"
Doesn't redo the math; adds usability:
- `ollama run llama3` → downloads the quantized GGUF, caches, runs.
- **Modelfile** (like a Dockerfile): base model + parameters + system prompt → a named model.
- Local **REST + OpenAI-compatible API** at `localhost:11434`.
- Loads/keeps-warm/unloads models for you.

### vLLM (production GPU serving)
Built for **many users at once**, maximizing **throughput** on NVIDIA GPUs:
- **Continuous batching**: add/remove requests from the running batch *every step* so the
  GPU is never idle (vs. naive "wait for the whole batch to finish").
- **PagedAttention**: manage the KV cache like OS virtual memory — small fixed **pages**
  handed out on demand. Almost no wasted GPU memory → far more concurrent users; sequences
  can even share pages (e.g. a shared system prompt).

### ollama vs vLLM
| | ollama | vLLM |
|---|--------|------|
| For | local / single-user | production / many users |
| Optimizes | **latency** (one user fast) | **throughput** (tokens/sec at scale) |
| Hardware | CPU / Mac / modest GPU | **NVIDIA GPU required** |
| Weights | quantized (4-bit) by default | usually fp16/bf16 |
| Killer feature | simplicity | continuous batching + PagedAttention |

Common pattern: **ollama in development, vLLM in production.**

---

## One-page summary

- **Pretraining** = knowledge (ceiling). **SFT** = behavior (follow instructions).
  **RLHF/RLVR** = quality (good/correct answers).
- **RLHF** rewards what *humans* prefer; **RLVR** rewards what *passes a check* (tests) —
  best for code/math.
- All training just **nudges weights** (same mechanism); only the *score* differs.
- **LoRA** trains a small frozen-model add-on → cheap, one adapter per stage.
- **Fine-tune for skills/style; RAG for facts.** Firm-knowledge expert = RAG-first (+ optional LoRA).
- **llama.cpp** = fast quantized engine; **ollama** = easy manager on top (local);
  **vLLM** = high-throughput GPU serving (production).
