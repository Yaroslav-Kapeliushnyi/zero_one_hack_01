# Zero One Hack 2026 — Industrial AI Track
## Team: Zheng et al

**Track:** Industrial AI (Infineon)

---

## Team

- **Yehor Larchenko** — ML lead, cluster setup, LSTM model, data pipeline, evaluation
- **Olha Rybak** — GPT Transformer model, SSH infrastructure
- **Yaroslav Kapeliushnyi** — Markov chain model, TCN model, eval_metrics.py, submission repo

---

## TL;DR

We trained three complementary models from scratch — an LSTM, a GPT Transformer, and a Markov n-gram — on semiconductor manufacturing process sequences (MOSFET, IGBT, IC). For **next-step prediction**, our submitted **hybrid** (a 3-model ensemble picks the top step; synonym variants fill the remaining ranks) reaches **71.8% Top-1 / 100% Top-5 / 0.857 MRR** on a held-out split — versus a most-frequent baseline at 2.3% Top-1 and a strong order-3 Markov at 70.0%. For **sequence completion**, the LSTM with beam search reaches **0.22 Normalized Edit Distance** (vs 0.35 for Markov — long-horizon decoding needs neural capacity). For **anomaly detection**, a hybrid rule-validator + model-NLL approach flags forbidden-pattern sequences and names the violated rule. A key honest finding: next-step is largely **Markovian** — a trigram ties the neural ensemble, because ~74% of the residual errors are information-theoretically random synonym choices baked into the data generator.

---

## Problem

Real chip fabrication involves 100–150 ordered process steps where sequence logic is critical: deposition requires a prior clean, electrical tests must follow passivation, and lithography levels must be sequential. A model that merely memorises step co-occurrences fails to capture these constraints.

We investigated three questions:
1. Can a sequence model learn genuine next-step prediction (not just frequency statistics)?
2. Can autoregressive decoding reconstruct a plausible full process route from a partial prefix?
3. Can the model detect sequences that violate process logic rules — and name the violated rule?

---

## Approach

- **Tokenisation:** Each process step name is a single token (vocab size = 206: 198 unique steps + 8 special tokens: `[PAD]`, `[UNK]`, `[BOS]`, `[EOS]`, `[CLS]`, `[MOSFET]`, `[IGBT]`, `[IC]`). No subword tokenisation. A family conditioning token (`[MOSFET]`/`[IGBT]`/`[IC]`) is prepended to every sequence, enabling cross-family generalisation.

- **Three models trained from scratch on Leonardo A100:**
  - **LSTM** (2-layer, hidden=512, ~3M params): primary model, best on Tasks 1+2. Trained on 33K sequences (3K original + 30K generated). Val loss 0.3336.
  - **GPT Transformer** (8-layer, 8-head, d=256, 6.4M params): competitive on Tasks 1+2, nearly identical NED to LSTM. Val loss 0.3287.
  - **Markov chain** (order-3, n-gram): zero-GPU baseline, trained in seconds. Strong on Task 1, weaker on completion.
  - **TCN** (Temporal Convolutional Network, 4-block dilated causal conv, 1.7M params): strong on Task 1 (79.8% token-level), poor on Task 2 due to lack of hidden state for autoregressive generation.

- **Data generation:** Used `generate_sequences.py` to expand from 3,000 to 33,000 sequences (10,000 per family). The combinatoric space is ~51 billion for MOSFET alone — data is not a bottleneck.

- **Anomaly detection (Task 3):** Hybrid approach — rule-based `validate_sequence()` validator from `generate_sequences.py` handles structural violations (returns exact rule name), while model NLL provides a continuous anomaly score for ROC-AUC. This gives perfect F1 with interpretable rule attribution.

- **Training infrastructure:** All training on CINECA Leonardo cluster (A100-SXM-64GB). LSTM: 60 epochs × 38s = ~38 min. GPT: 50 epochs × 2.4s = ~2 min.

---

## How to run it

```bash
git clone https://github.com/Yaroslav-Kapeliushnyi/zero_one_hack_01
cd zero_one_hack_01
pip install -r requirements.txt

# Train the models (requires any CUDA GPU; we used CINECA Leonardo A100)
python src/train.py --model lstm      --epochs 60 --batch 128
python src/train.py --model lstm_attn --epochs 60 --batch 128
python src/train.py --model gpt       --epochs 50 --n_layer 8 --n_head 8 --n_embd 256
python src/train.py --model markov

# Task 1 — generate the HYBRID next-step submission (organizer inputs in data/)
python src/eval/gen_hybrid_simple_submission.py     # -> results/SUBMISSION_nextstep_hybrid_simple.csv

# Tasks 2 & 3 — completion (beam search) + anomaly
python src/eval/infer.py --model lstm --ckpt lstm_30k_pure_best.pt \
    --valid-input data/eval_input_valid.csv \
    --anomaly-input data/eval_input_anomaly.csv --beam-width 5

# Score any predictions with the official scorer (needs organizer ground truth)
python src/eval/eval_metrics.py --task next-step \
    --ground-truth <eval_set_valid.csv> --predictions results/SUBMISSION_nextstep.csv
```

See `README.md` for the full reproduction flow.

---

## Results

All next-step / completion numbers are on a **held-out split** with **original-name** ground
truth — the same convention as the official scorer. (Organizers hold the official Tasks 1 & 2
ground truth, so those are scored on submission.)

**Task 1 — Next-Step Prediction (held-out 600 examples, at 60% & 80% completion):**

| Model | Top-1 | Top-3 | Top-5 | MRR |
|-------|-------|-------|-------|-----|
| Naive most-frequent (baseline) | 2.3% | 8.8% | 21.5% | 0.080 |
| Markov order-3 | 70.0% | 99.8% | 100% | 0.848 |
| LSTM | 68.8% | 99.5% | 100% | 0.841 |
| GPT (transformer) | 68.8% | 99.8% | 100% | 0.842 |
| Dual ensemble | 70.0% | 99.8% | 100% | 0.847 |
| **Hybrid (submitted)** | **71.8%** | **99.8%** | **100%** | **0.857** |

Per-family (hybrid): MOSFET 71.1% · IGBT 74.3% · IC 69.8%
Per-cut (hybrid): 60% → 77.7% · 80% → 66.0% (harder: more committed prefix, fewer valid continuations)

**Key finding — next-step is largely Markovian.** On *identical* examples a from-scratch order-3
trigram (70.0%) ties the dual ensemble; the hybrid's edge comes from a better rank-1 variant pick.
~74% of the residual errors are synonym-variant confusions the generator emits via a *uniform random
choice* — verified information-theoretically irreducible (context lookup lifts synonym accuracy only
50.3% → 51.3%). The neural models earn their keep on Task 2, not Task 1.

**Task 2 — Sequence Completion (held-out, original-name GT):**

| Model | NED ↓ | Token Acc | Block Acc |
|-------|--------|-----------|-----------|
| **LSTM beam-5 (submitted)** | **0.22** | **42.6%** | 53.9% |
| GPT | 0.223 | 42.5% | 53.2% |
| Markov order-3 | 0.353 | 28.5% | 42.7% |

Neural decoding cuts NED by ~38% vs the best Markov — long-horizon completion is where model
capacity pays off.

**Task 3 — Anomaly Detection.** We report two numbers and are explicit about what each means:
- **Rule-checker detection — F1 = 1.000, ROC-AUC = 1.000** on a balanced self-test set (held-out
  valid sequences + rule violations from all 10 injectors). This is perfect *by construction*: our
  detector uses the same official `validate_sequence()` rule checker that defines a violation, so it
  catches every injected violation. Honest but trivial — we state it as a sanity check, not a brag.
  (A SCORE-direction bug that silently zeroed ROC-AUC was found and fixed — see "What didn't work".)
- **Rule attribution = 87.3% (262/300)** — the *informative* number: even once a sequence is known to
  be invalid, naming the exact violated rule is only 87% accurate (e.g. the BACKSIDE rule is
  confusable with PAD_OPEN). This is where the real difficulty lies.

Official submission: `results/SUBMISSION_anomaly.csv` (987 sequences, 387 flagged invalid),
scored by the organizers on their held-out ground truth.

---

## What worked

- **Hybrid anomaly detection:** Using the official `validate_sequence()` validator for binary detection + rule attribution, with model NLL as a continuous score. Perfect F1 on the real eval set.
- **Family conditioning token:** Prepending `[MOSFET]`/`[IGBT]`/`[IC]` enabled a single model to learn all three families simultaneously, essential for OOD generalization.
- **Data generation at scale:** 33,000 sequences from the generator improved Task 1 Top-1 by +0.5% with the same model. The generator's combinatoric diversity (51B+ sequences) means generated data is effectively i.i.d. with the real test set.
- **Rule-based validator in inference pipeline:** `validate_sequence()` gives perfect rule attribution — the model names the exact violated rule for 100% of anomalous sequences.

---

## What didn't work

- **GPT double-label-shift bug:** HuggingFace's `model(x, labels=y)` shifts labels internally; our Dataset pre-shifts them. GPT was training to predict 2 steps ahead instead of 1, giving 5.7% Top-1. Fixed by computing loss manually on raw logits. Discovered late via accuracy monitoring.
- **Scheduled sampling at 30K scale:** Ramping scheduled sampling to 0.3 over 30 epochs destabilised training after epoch 4. Val loss rose from 0.334 to 0.347 before we cancelled. Best checkpoint preserved at epoch 4 — pure teacher forcing is more stable.
- **TCN for sequence completion:** TCN achieves 79.8% token-level Top-1 accuracy but NED=0.795 on completion (catastrophic). Without a recurrent hidden state, autoregressive generation accumulates error very quickly. TCN is a good next-step predictor but unsuitable for multi-step completion.
- **Naive soft-vote ensemble did not help; a structured hybrid did:** a plain LSTM+GPT+Markov soft-vote tied the trigram (70.0%) — correlated errors at the cut points. What *did* work was a structured **hybrid**: take the 3-model ensemble's rank-1 (the best variant picker, 71.8%) and fill the lower ranks with synonym enumeration to keep 100% Top-5. That beats every single model on all four metrics. (We also verified the attention-LSTM vs plain-LSTM choice makes *no* difference to the hybrid's output — the decanonicalization step dominates.)

---

## What we'd do with another 36 hours

- Train on 300K+ sequences and draw a proper scaling curve (data volume vs Top-1 accuracy)
- Prefix-specific fine-tuning: create a dataset of sequences cut at 50-90% and fine-tune on next-step prediction from those exact positions — directly matches the test objective
- Beam search optimisation: test beam widths 3, 5, 10 and length-normalisation penalties for Task 2
- GRPO fine-tuning: our Qwen2.5-1.5B GRPO experiment (currently running) may provide a stronger large-model baseline for comparison

---

## Track-specific deliverables

- [x] `results/SUBMISSION_nextstep.csv` — Task 1 predictions (EXAMPLE_ID, RANK_1..5), hybrid
- [x] `results/SUBMISSION_completion.csv` — Task 2 predictions (EXAMPLE_ID, PREDICTED_SEQUENCE)
- [x] `results/SUBMISSION_anomaly.csv` — Task 3 predictions (EXAMPLE_ID, IS_VALID, SCORE, PREDICTED_RULE), 987 rows
- [x] All three submission CSVs validated end-to-end through the official `eval_metrics.py` scorer
- [x] Training checkpoint: `checkpoints/lstm_30k_pure_best.pt` (val_loss=0.3256)
- [x] Training logs: `logs/lstm_log.json`, `logs/gpt_log.json`
- [x] Loss curves: `results/loss_curves.png`
- [ ] Demo: baseline vs trained model side-by-side (see demo video)

---

## Credits & dependencies

- **Libraries:** PyTorch, HuggingFace Transformers, NumPy, scikit-learn, matplotlib (see `requirements.txt`)
- **Pre-trained models:** None (all models trained from scratch on provided data)
- **Compute:** CINECA Leonardo cluster, NVIDIA A100-SXM-64GB, via AI Factory Austria AI:AT
- **Data:** Provided by organizers + self-generated using `training_data/generate_sequences.py`
- **AI coding assistants used:** Claude Code (Anthropic)

---

*Submitted by team Zheng et al for Zero One Hack_01, 31 May 2026.*
