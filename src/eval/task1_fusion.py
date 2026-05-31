"""
Task 1 rank-fusion experiment: Markov order-3  +  dual ensemble.

Both rankers are evaluated on the IDENTICAL 600 held-out examples produced by
build_self_eval(use_original_names=True) — original-name GT = official convention.
Markov is trained on the TRAIN split only (no leakage), original names.

Fusion strategies tried:
  - Dual alone / Markov-3 alone (references)
  - Reciprocal Rank Fusion (RRF), k in {1, 10, 60}
  - Borda count
  - Weighted RRF (dual heavier / markov heavier)

Run on cluster:
  ~/.pixi/bin/pixi run python src/eval/task1_fusion.py
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
from data import (build_vocab, train_val_split, SequenceDataset,
                  load_sequences, FAMILY_FILES, encode_prefix, parse_pipe_sequence)
import eval.infer as I

_SPECIAL = {"[PAD]", "[UNK]", "[BOS]", "[EOS]", "[CLS]", "[MOSFET]", "[IGBT]", "[IC]"}


# ── Markov order-3 on original names, TRAIN split only ────────────────────────
class Markov3:
    def __init__(self):
        self.table = defaultdict(Counter)
        self.fallback = Counter()
        self.order = 3

    def train(self, sequences):
        for seq in sequences:
            for s in seq:
                self.fallback[s] += 1
            for i in range(len(seq) - self.order):
                ctx = tuple(seq[i:i + self.order])
                self.table[ctx][seq[i + self.order]] += 1

    def top5(self, prefix):
        for length in (3, 2, 1):
            if len(prefix) >= length:
                ctx = tuple(prefix[-length:])
                # only the full-order table is keyed by 3-tuples; emulate backoff
                if length == self.order and ctx in self.table and self.table[ctx]:
                    return [s for s, _ in self.table[ctx].most_common(5)]
        return [s for s, _ in self.fallback.most_common(5)]


def build_markov_trainset():
    """TRAIN-split original-name sequences (same split as build_self_eval)."""
    vocab = build_vocab()
    dataset = SequenceDataset(vocab)
    train_ds, _ = train_val_split(dataset)
    all_orig = []
    for fam in ("mosfet", "igbt", "ic"):
        for seq in load_sequences(FAMILY_FILES[fam], apply_canonical=False).values():
            all_orig.append(seq)
    return [all_orig[i] for i in train_ds.indices]


# ── Fusion strategies ─────────────────────────────────────────────────────────
def rrf(list_a, list_b, k, wa=1.0, wb=1.0):
    score = defaultdict(float)
    for r, s in enumerate(list_a):
        score[s] += wa / (k + r + 1)
    for r, s in enumerate(list_b):
        score[s] += wb / (k + r + 1)
    return [s for s, _ in sorted(score.items(), key=lambda x: -x[1])][:5]


def borda(list_a, list_b):
    score = defaultdict(float)
    for r, s in enumerate(list_a):
        score[s] += (5 - r)
    for r, s in enumerate(list_b):
        score[s] += (5 - r)
    return [s for s, _ in sorted(score.items(), key=lambda x: -x[1])][:5]


def score_ranks(rows_ranks):
    """rows_ranks: list of (ranks, truth). Returns Top-1/3/5/MRR."""
    t1 = t3 = t5 = 0
    mrr = 0.0
    n = len(rows_ranks)
    for ranks, truth in rows_ranks:
        if ranks and ranks[0] == truth:
            t1 += 1
        if truth in ranks[:3]:
            t3 += 1
        if truth in ranks[:5]:
            t5 += 1
        if truth in ranks:
            mrr += 1.0 / (ranks.index(truth) + 1)
    return t1 / n, t3 / n, t5 / n, mrr / n


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vocab = build_vocab()

    # Dual ensemble models (reuse verified infer.py machinery)
    lstm_ckpt = "lstm_canonical_best.pt"
    model, _ = I.load_lstm(vocab, device, ckpt_name=lstm_ckpt)
    vocab_orig = I.build_orig_vocab()
    orig_ckpt = "lstm_30k_best.pt"
    orig_model, _ = I.load_lstm(vocab_orig, device, ckpt_name=orig_ckpt)
    family_priors = I.build_family_variant_priors()
    print(f"Dual ensemble: canonical({len(vocab)}) + original({len(vocab_orig)})")

    # Markov-3 (train split, original names)
    mk = Markov3()
    mk.train(build_markov_trainset())
    print("Markov-3 trained on train split (original names)")

    # Same 600 examples + original-name GT
    valid_rows, _ = I.build_self_eval(vocab, use_original_names=True)
    print(f"Eval examples: {len(valid_rows)}\n")

    dual_rows, mk_rows = [], []
    fusion = {f"RRF k={k}": [] for k in (1, 10, 60)}
    fusion["Borda"] = []
    fusion["RRF k=10 dual2x"] = []
    fusion["RRF k=10 mk2x"] = []

    for r in valid_rows:
        truth = r.get("_ACTUAL_NEXT_STEP", "")
        if not truth:
            continue
        family = r["FAMILY"].lower()
        steps = parse_pipe_sequence(r["PARTIAL_SEQUENCE"])

        # dual ensemble top-5
        prefix = encode_prefix(vocab, steps, family)
        prefix_orig = encode_prefix(vocab_orig, steps, family, apply_canonical=False)
        dual = I.predict_next_top5_dual(model, orig_model, vocab, vocab_orig,
                                        prefix, prefix_orig, device, family,
                                        family_priors=family_priors)
        # markov top-5 (original names)
        mkp = mk.top5(steps)

        dual_rows.append((dual, truth))
        mk_rows.append((mkp, truth))
        fusion["RRF k=1"].append((rrf(dual, mkp, 1), truth))
        fusion["RRF k=10"].append((rrf(dual, mkp, 10), truth))
        fusion["RRF k=60"].append((rrf(dual, mkp, 60), truth))
        fusion["Borda"].append((borda(dual, mkp), truth))
        fusion["RRF k=10 dual2x"].append((rrf(dual, mkp, 10, wa=2.0, wb=1.0), truth))
        fusion["RRF k=10 mk2x"].append((rrf(dual, mkp, 10, wa=1.0, wb=2.0), truth))

    def line(name, rows):
        t1, t3, t5, mrr = score_ranks(rows)
        print(f"{name:<22}{t1*100:>7.1f}%{t3*100:>7.1f}%{t5*100:>7.1f}%{mrr:>8.3f}")

    print(f"{'Strategy':<22}{'Top-1':>8}{'Top-3':>8}{'Top-5':>8}{'MRR':>8}")
    print("-" * 54)
    line("Dual alone", dual_rows)
    line("Markov-3 alone", mk_rows)
    print("-" * 54)
    for name, rows in fusion.items():
        line(name, rows)
    print("\n(Same 600 examples, original-name GT. Best fusion vs 70.0% dual = the gain.)")


if __name__ == "__main__":
    main()
