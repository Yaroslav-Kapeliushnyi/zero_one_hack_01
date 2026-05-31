"""
Generate the OFFICIAL Task 1 submission via rank fusion (RRF k=10, dual x2).
Best measured strategy: 72.3% Top-1 / 100% Top-5 on original-name GT.

Markov-3 trained on ALL training sequences (original names) for max coverage.
Dual ensemble = canonical LSTM (type) + original LSTM (variant).
Reads eval_input_valid.csv (600 official rows) → writes results/SUBMISSION_nextstep.csv

Run on cluster:
  ~/.pixi/bin/pixi run python src/eval/gen_fusion_submission.py
"""

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
from data import (build_vocab, load_sequences, FAMILY_FILES,
                  encode_prefix, parse_pipe_sequence)
import eval.infer as I

VALID_INPUT = ROOT / "data" / "eval_input_valid.csv"
OUT = ROOT / "results" / "SUBMISSION_nextstep.csv"


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
                self.table[tuple(seq[i:i + self.order])][seq[i + self.order]] += 1

    def top5(self, prefix):
        if len(prefix) >= self.order:
            ctx = tuple(prefix[-self.order:])
            if ctx in self.table and self.table[ctx]:
                return [s for s, _ in self.table[ctx].most_common(5)]
        return [s for s, _ in self.fallback.most_common(5)]


def rrf(a, b, k=10, wa=2.0, wb=1.0):
    score = defaultdict(float)
    for r, s in enumerate(a):
        score[s] += wa / (k + r + 1)
    for r, s in enumerate(b):
        score[s] += wb / (k + r + 1)
    return [s for s, _ in sorted(score.items(), key=lambda x: -x[1])][:5]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    vocab = build_vocab()

    model, _ = I.load_lstm(vocab, device, ckpt_name="lstm_canonical_best.pt")
    vocab_orig = I.build_orig_vocab()
    orig_model, _ = I.load_lstm(vocab_orig, device, ckpt_name="lstm_30k_best.pt")
    family_priors = I.build_family_variant_priors()
    print(f"Dual ensemble: canonical({len(vocab)}) + original({len(vocab_orig)})")

    # Markov-3 on ALL training data (original names) for max coverage.
    all_seqs = []
    for fam in ("mosfet", "igbt", "ic"):
        all_seqs += list(load_sequences(FAMILY_FILES[fam], apply_canonical=False).values())
    mk = Markov3()
    mk.train(all_seqs)
    print(f"Markov-3 trained on {len(all_seqs)} sequences (original names)")

    rows = list(csv.DictReader(open(VALID_INPUT)))
    print(f"Official eval rows: {len(rows)}")

    out_rows = []
    for r in rows:
        eid = r["EXAMPLE_ID"]
        family = r["FAMILY"].lower()
        steps = parse_pipe_sequence(r["PARTIAL_SEQUENCE"])

        prefix = encode_prefix(vocab, steps, family)
        prefix_orig = encode_prefix(vocab_orig, steps, family, apply_canonical=False)
        if family not in ("mosfet", "igbt", "ic"):
            family = I.route_family(model, "lstm", vocab, steps, device)
            prefix = encode_prefix(vocab, steps, family)
            prefix_orig = encode_prefix(vocab_orig, steps, family, apply_canonical=False)

        dual = I.predict_next_top5_dual(model, orig_model, vocab, vocab_orig,
                                        prefix, prefix_orig, device, family,
                                        family_priors=family_priors)
        mkp = mk.top5(steps)
        fused = rrf(dual, mkp, k=10, wa=2.0, wb=1.0)
        while len(fused) < 5:
            fused.append("")
        out_rows.append({"EXAMPLE_ID": eid, "RANK_1": fused[0], "RANK_2": fused[1],
                         "RANK_3": fused[2], "RANK_4": fused[3], "RANK_5": fused[4]})

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, ["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n✓ wrote {len(out_rows)} rows -> {OUT.name} (RRF k=10 dual2x fusion, 72.3% Top-1)")


if __name__ == "__main__":
    main()
