# Zero One Hack 2026 — Industrial AI Track
### Learning and Benchmarking Process Logic (Infineon)

Three sequence models trained on semiconductor manufacturing process routes (MOSFET, IGBT, IC) to predict next steps, complete partial sequences, and detect process logic violations.

---

## Quick Start

```bash
git clone https://github.com/Yaroslav-Kapeliushnyi/zero_one_hack_01
cd zero_one_hack_01
pip install -r requirements.txt
```

---

## Training

```bash
# Train LSTM on original 3K sequences
python src/train.py --model lstm --epochs 60 --batch 64

# Train LSTM on 33K sequences (generate extra data first)
# Training data + generator live in tracks/industrial-infineon/training_data/
cd tracks/industrial-infineon/training_data
python generate_sequences.py --family mosfet --count 10000 --output mosfet_10k.csv --seed 10
python generate_sequences.py --family igbt   --count 10000 --output igbt_10k.csv   --seed 11
python generate_sequences.py --family ic     --count 10000 --output ic_10k.csv     --seed 12
cd -
python src/train.py --model lstm --epochs 60 --batch 128 \
    --extra-data mosfet:tracks/industrial-infineon/training_data/mosfet_10k.csv \
                 igbt:tracks/industrial-infineon/training_data/igbt_10k.csv \
                 ic:tracks/industrial-infineon/training_data/ic_10k.csv

# Train Markov chain (no GPU, instant)
python src/train.py --model markov

# Train GPT + attention-LSTM (both used by the Task-1 hybrid ensemble)
python src/train.py --model gpt      --epochs 50 --n_layer 8 --n_head 8 --n_embd 256
python src/train.py --model lstm_attn --epochs 60 --batch 128
```

Checkpoints saved to `checkpoints/`. Training logs to `logs/`.

---

## Inference — reproduce the submission

Place the organizer eval inputs in `data/`:
`data/eval_input_valid.csv` (Tasks 1 & 2, 600 rows) and `data/eval_input_anomaly.csv` (Task 3, 987 rows).

```bash
# Task 1 — HYBRID next-step submission (the submitted model):
#   RANK_1 = 3-model ensemble (LSTM-Attn + GPT + Markov, canonical) -> decanonicalized top variant
#   RANK_2-5 = synonym variants of the predicted step (static lookup)
python src/eval/gen_hybrid_simple_submission.py
#   -> results/SUBMISSION_nextstep_hybrid_simple.csv  (copy to SUBMISSION_nextstep.csv)

# Tasks 2 & 3 — completion (beam search) + anomaly (rule validator + LSTM NLL):
python src/eval/infer.py --model lstm --ckpt lstm_30k_pure_best.pt \
    --valid-input  data/eval_input_valid.csv \
    --anomaly-input data/eval_input_anomaly.csv --beam-width 5

# Self-evaluation on our held-out val split (no organizer GT needed):
python src/eval/infer.py --model dual_ensemble --self-eval --beam-width 5
```

Final submission files in `results/`:
- `SUBMISSION_nextstep.csv`   — Task 1: EXAMPLE_ID, RANK_1..5  (hybrid)
- `SUBMISSION_completion.csv` — Task 2: EXAMPLE_ID, PREDICTED_SEQUENCE  (LSTM beam-5)
- `SUBMISSION_anomaly.csv`    — Task 3: EXAMPLE_ID, IS_VALID, SCORE, PREDICTED_RULE

---

## Evaluation

The organizers hold the ground truth for Tasks 1 & 2, so those are scored on submission.
Score any predictions with the official scorer (`src/eval/eval_metrics.py`):

```bash
python src/eval/eval_metrics.py --task next-step \
    --ground-truth <eval_set_valid.csv> --predictions results/SUBMISSION_nextstep.csv

python src/eval/eval_metrics.py --task completion \
    --ground-truth <eval_set_valid.csv> --predictions results/SUBMISSION_completion.csv

python src/eval/eval_metrics.py --task anomaly \
    --ground-truth <eval_set_forbidden.csv> --predictions results/SUBMISSION_anomaly.csv \
    --valid-supplement <eval_set_valid.csv>
```

---

## Project Structure

```
src/
├── data.py                    # Vocabulary, SequenceDataset
├── train.py                   # Training loop (LSTM, GPT, Markov)
├── models/
│   ├── lstm_baseline.py       # 2-layer LSTM
│   ├── gpt_model.py           # GPT-2 style transformer
│   ├── markov.py              # N-gram Markov chain (order=3)
│   └── tcn.py                 # Temporal Convolutional Network
└── eval/
    ├── eval_metrics.py                # Official metrics (Task 1/2/3)
    ├── infer.py                       # Inference + submission CSV generation
    ├── gen_hybrid_simple_submission.py # Task-1 hybrid submission generator
    └── anomaly_generator.py           # Rule violation injector (all 10 rules)
tracks/industrial-infineon/training_data/
├── MOSFET_variants.csv        # 1K training sequences
├── IGBT_variants.csv
├── IC_variants.csv
└── generate_sequences.py      # Sequence generator + validate_sequence()
data/                          # organizer eval inputs (eval_input_valid.csv, eval_input_anomaly.csv)
checkpoints/                   # Saved model weights
logs/                          # Training loss logs
results/                       # Submission CSVs (SUBMISSION_*.csv)
```

---

## Requirements

- Python 3.11+
- CUDA GPU recommended (A100 used for training)
- See `requirements.txt`

---

## Team

Yehor Larcenko · Olga Rybak · Yaroslav Kapeliushnyi

*Zero One Hack_01 — Vienna, May 2026*
