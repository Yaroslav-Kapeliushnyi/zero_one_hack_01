"""
Train and evaluate simple baselines:
  1. Order-1 Markov (unigram context)
  2. Order-2 Markov (bigram context)
  3. GBM (last 5 steps as integer features, sklearn)
Evaluated on the same 300-sequence val split used in infer.py.
"""

import sys, json, random
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data import build_vocab, SequenceDataset, train_val_split

SPECIAL = {"[PAD]","[UNK]","[BOS]","[EOS]","[CLS]","[MOSFET]","[IGBT]","[IC]"}

vocab   = build_vocab()
dataset = SequenceDataset(vocab)
_, val_ds = train_val_split(dataset)
val_idx_set = set(val_ds.indices)

# ── Build train sequences ─────────────────────────────────────────────────────
train_seqs = []
for idx in range(len(dataset.samples)):
    if idx in val_idx_set:
        continue
    ids   = dataset.samples[idx]
    steps = [vocab.id2step[i] for i in ids if vocab.id2step[i] not in SPECIAL]
    if len(steps) > 2:
        train_seqs.append(steps)
print(f"Train sequences: {len(train_seqs)}")

# ── Build eval pairs (same split as infer.py) ─────────────────────────────────
eval_pairs = []
for idx in val_ds.indices[:300]:
    ids   = dataset.samples[idx]
    steps = [vocab.id2step[i] for i in ids if vocab.id2step[i] not in SPECIAL]
    for frac in (0.6, 0.8):
        cut = max(1, int(len(steps) * frac))
        if cut < len(steps):
            eval_pairs.append((steps[:cut], steps[cut]))
print(f"Eval pairs: {len(eval_pairs)}")

# ── Markov order-1 ────────────────────────────────────────────────────────────
counts1 = defaultdict(lambda: defaultdict(int))
for seq in train_seqs:
    for i in range(len(seq) - 1):
        counts1[seq[i]][seq[i+1]] += 1

def markov1_top1(prefix):
    dist = counts1.get(prefix[-1], {})
    return max(dist, key=dist.get) if dist else None

acc1 = sum(1 for p, t in eval_pairs if markov1_top1(p) == t) / len(eval_pairs)
print(f"Markov order-1  Top-1: {acc1*100:.1f}%")

# ── Markov order-2 ────────────────────────────────────────────────────────────
counts2 = defaultdict(lambda: defaultdict(int))
for seq in train_seqs:
    for i in range(len(seq) - 1):
        ctx = tuple(seq[max(0, i-1):i+1])
        counts2[ctx][seq[i+1]] += 1

def markov2_top1(prefix):
    ctx = tuple(prefix[-2:]) if len(prefix) >= 2 else (prefix[-1],)
    dist = counts2.get(ctx, {})
    return max(dist, key=dist.get) if dist else markov1_top1(prefix)

acc2 = sum(1 for p, t in eval_pairs if markov2_top1(p) == t) / len(eval_pairs)
print(f"Markov order-2  Top-1: {acc2*100:.1f}%")

# ── GBM (last 5 steps as integer features) ────────────────────────────────────
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

all_steps = sorted({s for seq in train_seqs for s in seq})
le = LabelEncoder().fit(all_steps)
WINDOW = 5

def featurize(prefix):
    padded = ["[PAD]"] * WINDOW + list(prefix)
    last_w = padded[-WINDOW:]
    return [int(le.transform([s])[0]) if s in le.classes_ else -1 for s in last_w]

X_train, y_train = [], []
for seq in train_seqs:
    for i in range(1, len(seq)):
        if seq[i] in le.classes_:
            X_train.append(featurize(seq[:i]))
            y_train.append(int(le.transform([seq[i]])[0]))

print(f"GBM samples: {len(X_train)} — fitting...")
gbm = GradientBoostingClassifier(
    n_estimators=100, max_depth=5, learning_rate=0.15,
    subsample=0.8, random_state=42)
gbm.fit(X_train, y_train)

correct_gbm = 0
for prefix, true_next in eval_pairs:
    if true_next not in le.classes_:
        continue
    pred_id   = gbm.predict([featurize(prefix)])[0]
    pred_step = le.inverse_transform([pred_id])[0]
    if pred_step == true_next:
        correct_gbm += 1

acc_gbm = correct_gbm / len(eval_pairs)
print(f"GBM (last-5 steps)  Top-1: {acc_gbm*100:.1f}%")

results = {
    "markov_order1": round(acc1 * 100, 1),
    "markov_order2": round(acc2 * 100, 1),
    "gbm":           round(acc_gbm * 100, 1),
}
out = ROOT / "eval_results" / "baseline_results.json"
out.parent.mkdir(exist_ok=True)
json.dump(results, open(out, "w"), indent=2)
print("\nFinal results:", results)
print(f"Saved to {out}")
