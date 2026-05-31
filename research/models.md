# Base Model Survey — Industrial Process Sequence Modeling

> For Zero One Hack 2026 Industrial Track
> Task: next-step prediction, sequence completion, anomaly detection on ~150-step semiconductor sequences

---

## Critical Design Decision: Tokenization

**Option 1 — Step-level tokens (RECOMMENDED)**
Assign integer IDs 0–N to each unique step name. `vocab_size = ~200`. Train from scratch or replace embedding layer. Clean, correct, efficient.

**Option 2 — Subword tokenization**
"MEASURE THICKNESS" → ["ME", "ASURE", "_THICK", "NESS"] (4 tokens). A 150-step sequence becomes 300–600 subword tokens. Wasteful — model must re-learn step-level boundaries.

**Conclusion: Use Option 1.** This means pretrained LLM vocabulary and embeddings add zero value → favors train-from-scratch or replace-embedding approaches.

---

## Category A — Small/Medium LLMs

### GPT-2 Variants

| Model | HF ID | Params | Context | A100 Time | Verdict |
|-------|-------|--------|---------|-----------|---------|
| GPT-2 Small | `openai-community/gpt2` | 117M | 1024 | ~5–15 min | Good |
| GPT-2 Medium | `openai-community/gpt2-medium` | 345M | 1024 | ~15–30 min | Good |
| GPT-2 Large | `openai-community/gpt2-large` | 774M | 1024 | ~30–60 min | Overkill |

Replace embedding + LM head with `vocab_size=200`. Pretrained text weights irrelevant but attention architecture is solid. Use `GPT2Config(vocab_size=200, n_positions=200, ...)` + `GPT2LMHeadModel`.

### SmolLM2

| Model | HF ID | Params | Context | A100 Time | Verdict |
|-------|-------|--------|---------|-----------|---------|
| SmolLM2-135M | `HuggingFaceTB/SmolLM2-135M` | 135M | 2048 | ~5–15 min | ⭐ Recommended |
| SmolLM2-360M | `HuggingFaceTB/SmolLM2-360M` | 360M | 2048 | ~15–40 min | Good |
| SmolLM2-1.7B | `HuggingFaceTB/SmolLM2-1.7B` | 1.7B | 2048 | ~30–60 min | Overkill |

Not gated. Modern Llama-style (RoPE, RMSNorm, SwiGLU). 150 tokens << 2048 context. Swap embed_tokens + lm_head to vocab_size=200, full fine-tune all weights.

### Qwen 2.5

| Model | HF ID | Params | Context | A100 Time | Verdict |
|-------|-------|--------|---------|-----------|---------|
| Qwen2.5-0.5B | `Qwen/Qwen2.5-0.5B` | 0.5B | 128K | ~15–40 min | Good |
| Qwen2.5-1.5B | `Qwen/Qwen2.5-1.5B` | 1.5B | 128K | ~30–60 min | OK |

Not gated. Strong architecture, dark horse candidate. Same vocab-replacement strategy needed.

### Too Large / Gated — Not Recommended

- LLaMA 3.1/3.2 (gated, overkill)
- Mistral 7B (overkill, 4–8h training)
- Phi-3/3.5 (instruction-tuned wrong direction)
- Gemma 2 (gated)

---

## Category B — Encoder-Decoder / Encoder-Only

### T5 / FLAN-T5

| Model | HF ID | Params | Context | A100 Time |
|-------|-------|--------|---------|-----------|
| T5-Small | `google-t5/t5-small` | 60M | 512 | ~5–10 min |
| T5-Base | `google-t5/t5-base` | 220M | 512 | ~10–20 min |
| FLAN-T5-Base | `google/flan-t5-base` | 250M | 512 | ~10–20 min |

**Pros:** Seq2seq is natural for completion task (input: 60% → output: 40%).
**Cons:** 512 subword limit is tight (150 steps × 2–3 subtokens ≈ 400+ tokens). Awkward for next-step prediction. Needs separate model for generation tasks.
**Verdict:** Useful for Task 2 + 3 only. Not recommended as primary model.

### BERT / RoBERTa

| Model | HF ID | Params | A100 Time | Best For |
|-------|-------|--------|-----------|----------|
| BERT-Base | `google-bert/bert-base-uncased` | 110M | ~5–10 min | Task 3 only |
| RoBERTa-Base | `FacebookAI/roberta-base` | 125M | ~5–10 min | Task 3 only |

**Pros:** Excellent anomaly detector (binary classification), fast, well-understood.
**Cons:** Encoder-only — no generation. Cannot do Tasks 1 or 2 alone.
**Verdict:** Best dedicated anomaly detector if you split models per task, but not standalone.

---

## Category C — Train From Scratch (Recommended)

### Custom nanoGPT-style Transformer

| Config | Params | Context | A100 Time | All 3 Tasks |
|--------|--------|---------|-----------|-------------|
| 6L, 6H, d=192 | ~2–5M | Custom | ~2–5 min | ✅ Native |
| 12L, 8H, d=512 | ~20–50M | Custom | ~10–20 min | ✅ Native |

**The most principled approach.** Define `vocab_size = len(unique_steps)` (~200), sequence length 200, train from scratch. No vocabulary mismatch, no irrelevant pretrained weights.

Via HuggingFace: `GPT2Config(vocab_size=200, n_positions=256, n_embd=256, n_layer=8, n_head=8)` + `GPT2LMHeadModel`.
Via nanoGPT: `github.com/karpathy/nanoGPT` — minimal clean implementation.

3K sequences × 130 steps × 20 epochs ≈ 8M tokens. Trains in minutes.

### LSTM / GRU Baseline

| Config | Params | A100 Time | All 3 Tasks |
|--------|--------|-----------|-------------|
| 2L LSTM, hidden=256 | ~1–3M | ~1–3 min | ✅ Native |
| 2L GRU, hidden=512 | ~2–5M | ~2–5 min | ✅ Native |

Pure PyTorch (`torch.nn.LSTM`). Essential baseline — if transformer doesn't beat LSTM meaningfully, problem is data-limited not architecture-limited.

### Mamba / SSM

| Model | HF ID | Params | Notes |
|-------|-------|--------|-------|
| Mamba-130M | `state-spaces/mamba-130m-hf` | 130M | O(n) inference, unlimited context |

Interesting but adds ecosystem complexity. Advantage over transformer at 150-step sequences is minimal. Possible stretch goal.

---

## Full Comparison Table

| Model | Params | Task 1 | Task 2 | Task 3 | A100 Time | Recommended? |
|-------|--------|--------|--------|--------|-----------|--------------|
| **Custom nanoGPT (scratch)** | 2–50M | ✅ | ✅ | ✅ +head | 2–20 min | ⭐⭐⭐ |
| **SmolLM2-135M (replace emb)** | 135M | ✅ | ✅ | ✅ +head | 5–15 min | ⭐⭐⭐ |
| **LSTM/GRU baseline** | 1–5M | ✅ | ✅ | ✅ +head | 1–5 min | ⭐⭐ baseline |
| GPT-2 Small (replace emb) | 117M | ✅ | ✅ | ✅ +head | 5–20 min | ⭐⭐ |
| SmolLM2-360M | 360M | ✅ | ✅ | ✅ +head | 15–40 min | ⭐⭐ |
| Qwen2.5-0.5B | 500M | ✅ | ✅ | ✅ +head | 15–40 min | ⭐⭐ |
| T5-Base (seq2seq) | 220M | ⚠️ | ✅ | ✅ cls | 15–30 min | ⭐ Task 2+3 |
| RoBERTa-Base | 125M | ❌ | ❌ | ✅ | 5–15 min | ⭐ Task 3 |
| LLaMA 3.2 1B | 1B | ✅ | ✅ | ✅ +head | 30–90 min | ❌ gated |
| Mistral 7B | 7B | ✅ | ✅ | ✅ +head | 4–8 hr | ❌ overkill |
| Mamba-130M | 130M | ✅ | ✅ | ✅ +head | 10–25 min | ⭐ stretch |

---

## Final Recommendation

### Primary: Custom Small GPT Transformer (Train From Scratch)
Most technically correct. `vocab_size=200`, `sequence_length=200`, 8–12 transformer layers, d_model=256–512. Trains in <30 min on A100. Covers all 3 tasks with one model. No gated model approvals needed.

**Multi-task inference from one model:**
- Task 1 (next-step): argmax of next-token distribution
- Task 2 (completion): autoregressive decode from prefix until `[END]` token
- Task 3 (anomaly): compute sequence log-likelihood and threshold; OR add `[CLS]` + binary head

### Secondary: SmolLM2-135M (Replace Embeddings)
If you want a pretrained transformer backbone. Swap embedding + LM head to vocab_size=200. Modern architecture, not gated, 2048 context, full fine-tune in <20 min.

### Baseline: LSTM/GRU
Pure PyTorch. 2-layer unidirectional for generation tasks, 2-layer bidirectional for anomaly detection. Trains in 2–5 minutes. Essential for before/after comparison required by judges.

---

## Implementation Notes

1. **Build `step2id` dict** from unique step names across all CSVs. No BPE tokenization.
2. **Data size:** 3K sequences × 130 steps = 390K tokens — tiny. Use dropout 0.1–0.2 + early stopping.
3. **Anomaly scoring (free):** Compute per-sequence perplexity from your trained LM. High perplexity = anomalous sequence. No separate model needed.
4. **For Task 3 classification head:** Add `[CLS]` token at position 0, pool its hidden state, linear layer to 2 classes.
5. **Generate more data:** Use `training_data/generate_sequences.py` to expand from 3K to 30K+ sequences before training.
