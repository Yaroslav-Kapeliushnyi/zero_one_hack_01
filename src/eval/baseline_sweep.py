"""
Honest baseline sweep for Task 1 (next-step prediction).

All models are evaluated on the IDENTICAL 600 held-out next-step examples
(300 val sequences x {0.6, 0.8} cut points) with ORIGINAL step names as
ground truth — exactly matching the official scorer convention and the
dual-ensemble self-eval (build_self_eval(use_original_names=True)).

Baselines:
  - Naive (global most-frequent next step)
  - Markov order-1 / order-2 / order-3 (trained on TRAIN split, original names)

Reference (already measured, printed for context):
  - Dual ensemble: 70.0% Top-1
  - Canonical LSTM + Markov decan: 67.0% Top-1

Run on cluster:
  ~/.pixi/bin/pixi run python src/eval/baseline_sweep.py
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data import (SequenceDataset, build_vocab, train_val_split,
                  load_sequences, FAMILY_FILES)

_SPECIAL = {"[PAD]", "[UNK]", "[BOS]", "[EOS]", "[CLS]", "[MOSFET]", "[IGBT]", "[IC]"}


def build_eval_split():
    """
    Reproduce build_self_eval(use_original_names=True) exactly:
    same vocab, same train/val split, same val indices, original step names.
    Returns (train_seqs_original, eval_examples) where each eval example is
    (family, prefix_steps, true_next_step).
    """
    vocab = build_vocab()
    dataset = SequenceDataset(vocab)
    train_ds, val_ds = train_val_split(dataset)

    # Parallel original-name sequences, same order as dataset.samples.
    all_orig = []   # list[(family, [steps...])]
    for fam in ("mosfet", "igbt", "ic"):
        seqs = load_sequences(FAMILY_FILES[fam], apply_canonical=False)
        for seq in seqs.values():
            all_orig.append((fam, seq))

    # TRAIN sequences (original names) — for fitting Markov baselines.
    train_seqs = [all_orig[i][1] for i in train_ds.indices]

    # EVAL examples — same 300 val sequences x {0.6, 0.8}, original-name GT.
    # Each: (family, prefix, true_next_step, frac, remaining_steps)
    eval_examples = []
    for idx in val_ds.indices[:300]:
        family, steps = all_orig[idx]
        for frac in (0.6, 0.8):
            cut = max(1, int(len(steps) * frac))
            if cut < len(steps):
                eval_examples.append((family, steps[:cut], steps[cut], frac, steps[cut:]))
    return train_seqs, eval_examples


# ── Markov n-gram (self-contained, original names) ────────────────────────────
class Markov:
    def __init__(self, order):
        self.order = order
        self.table = defaultdict(Counter)
        self.fallback = Counter()

    def train(self, sequences):
        for seq in sequences:
            for s in seq:
                self.fallback[s] += 1
            for i in range(len(seq) - self.order):
                ctx = tuple(seq[i:i + self.order])
                self.table[ctx][seq[i + self.order]] += 1

    def predict_top5(self, prefix):
        if len(prefix) >= self.order:
            ctx = tuple(prefix[-self.order:])
            if ctx in self.table and self.table[ctx]:
                top = [s for s, _ in self.table[ctx].most_common(5)]
                if len(top) >= 5:
                    return top
                # pad with fallback
                for s, _ in self.fallback.most_common():
                    if s not in top:
                        top.append(s)
                    if len(top) == 5:
                        break
                return top
        return [s for s, _ in self.fallback.most_common(5)]

    def complete(self, prefix, max_new=160):
        """Greedy multi-step completion with a simple repetition guard."""
        gen = list(prefix)
        out = []
        for _ in range(max_new):
            top = self.predict_top5(gen)
            if not top:
                break
            nxt = top[0]
            if len(gen) >= 2 and nxt == gen[-1] == gen[-2]:
                break
            out.append(nxt)
            gen.append(nxt)
        return out


class Naive:
    """Global most-frequent next step, ignores context."""
    def __init__(self):
        self.top5 = []

    def train(self, sequences):
        c = Counter()
        for seq in sequences:
            for s in seq:
                c[s] += 1
        self.top5 = [s for s, _ in c.most_common(5)]

    def predict_top5(self, prefix):
        return self.top5


def score(model, eval_examples):
    t1 = t3 = t5 = 0
    mrr = 0.0
    n = len(eval_examples)
    for family, prefix, truth, frac, remaining in eval_examples:
        ranks = model.predict_top5(prefix)
        if ranks and ranks[0] == truth:
            t1 += 1
        if truth in ranks[:3]:
            t3 += 1
        if truth in ranks[:5]:
            t5 += 1
        if truth in ranks:
            mrr += 1.0 / (ranks.index(truth) + 1)
    return {"Top1": t1 / n, "Top3": t3 / n, "Top5": t5 / n, "MRR": mrr / n}


def _levenshtein(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = prev[j - 1] if a[i - 1] == b[j - 1] else 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def score_completion(model, eval_examples):
    """Task 2 metrics for a model with a .complete() method, original-name GT."""
    ned_sum = tok_sum = 0.0
    n = len(eval_examples)
    for family, prefix, truth, frac, remaining in eval_examples:
        pred = model.complete(prefix)
        if pred or remaining:
            ned_sum += _levenshtein(pred, remaining) / max(len(pred), len(remaining), 1)
        k = min(len(pred), len(remaining))
        if k:
            tok_sum += sum(p == r for p, r in zip(pred, remaining)) / k
    return {"NED": ned_sum / n, "TokAcc": tok_sum / n}


def main():
    print("Building eval split (same 600 examples, original-name GT)...")
    train_seqs, eval_examples = build_eval_split()
    print(f"  Train sequences: {len(train_seqs)}")
    print(f"  Eval examples  : {len(eval_examples)} (300 val x 0.6/0.8)\n")

    models = {
        "Naive (most-frequent)": Naive(),
        "Markov order-1":        Markov(1),
        "Markov order-2":        Markov(2),
        "Markov order-3":        Markov(3),
    }

    print("TASK 1 — NEXT-STEP")
    print(f"{'Baseline':<26}{'Top-1':>8}{'Top-3':>8}{'Top-5':>8}{'MRR':>8}")
    print("-" * 58)
    trained = {}
    for name, m in models.items():
        m.train(train_seqs)
        trained[name] = m
        r = score(m, eval_examples)
        print(f"{name:<26}{r['Top1']*100:>7.1f}%{r['Top3']*100:>7.1f}%"
              f"{r['Top5']*100:>7.1f}%{r['MRR']:>8.3f}")
    print("-" * 58)
    print(f"{'Dual ensemble (ours)':<26}{70.0:>7.1f}%{99.8:>7.1f}%{100.0:>7.1f}%{0.847:>8.3f}")

    print("\nTASK 2 — COMPLETION (greedy, original-name GT)")
    print(f"{'Baseline':<26}{'NED (lower=better)':>20}{'TokAcc':>10}")
    print("-" * 58)
    for name in ("Markov order-1", "Markov order-2", "Markov order-3"):
        c = score_completion(trained[name], eval_examples)
        print(f"{name:<26}{c['NED']:>20.4f}{c['TokAcc']*100:>9.1f}%")
    print("-" * 58)
    print(f"{'LSTM beam-5 (ours)':<26}{0.22:>20.4f}{42.6:>9.1f}%")

    print("\n(All baselines measured on the SAME 600 examples,")
    print(" original-name ground truth = official scorer convention.)")


if __name__ == "__main__":
    main()
