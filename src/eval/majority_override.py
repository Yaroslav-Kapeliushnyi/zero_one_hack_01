"""
THE 80% EXPERIMENT — majority-variant override.

Finding (from generator source + training data):
  - Generator picks synonyms with UNIFORM rng.choice() — no context signal.
  - ~29% of eval next-steps are synonym-group members.
  - Canonical LSTM gets the step TYPE right 91.8% of the time.
  - Current pipeline picks the right VARIANT poorly on synonyms → true Top-1 ~70.7%.

Fix (no retraining): predict the canonical TYPE with the canonical LSTM, then
emit the MOST-FREQUENT original variant of that type (global or per-family).

Measures honestly on the same 600 examples, original-name GT. Compares:
  A. canonical-name output (no override)
  B. most-frequent variant override (global)
  C. most-frequent variant override (family-conditioned)

Run on cluster:
  ~/.pixi/bin/pixi run python src/eval/majority_override.py
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
from data import (build_vocab, encode_prefix, parse_pipe_sequence,
                  CANONICAL_STEPS, VARIANTS_OF, FAMILY_FILES, load_sequences)
import eval.infer as I

_SPECIAL = {"[PAD]", "[UNK]", "[BOS]", "[EOS]", "[CLS]", "[MOSFET]", "[IGBT]", "[IC]"}


def build_majority_maps():
    glob = defaultdict(Counter)
    per_fam = defaultdict(lambda: defaultdict(Counter))
    for fam, path in FAMILY_FILES.items():
        for seq in load_sequences(path, apply_canonical=False).values():
            for s in seq:
                c = CANONICAL_STEPS.get(s, s)
                glob[c][s] += 1
                per_fam[fam][c][s] += 1
    glob_map = {c: cnt.most_common(1)[0][0] for c, cnt in glob.items()}
    fam_map = {fam: {c: cnt.most_common(1)[0][0] for c, cnt in d.items()}
               for fam, d in per_fam.items()}
    return glob_map, fam_map


def variant(canon_step, family, glob_map, fam_map, mode):
    if canon_step not in VARIANTS_OF:
        return canon_step
    if mode == "global":
        return glob_map.get(canon_step, canon_step)
    if mode == "family":
        return fam_map.get(family, {}).get(canon_step, glob_map.get(canon_step, canon_step))
    return canon_step


def score(rows):
    t1 = t3 = t5 = 0
    mrr = 0.0
    n = len(rows)
    for ranks, truth in rows:
        if ranks and ranks[0] == truth:
            t1 += 1
        if truth in ranks[:3]:
            t3 += 1
        if truth in ranks[:5]:
            t5 += 1
        if truth in ranks:
            mrr += 1.0 / (ranks.index(truth) + 1)
    return t1 / n, t3 / n, t5 / n, mrr / n


@torch.no_grad()
def canonical_top5(model, vocab, steps, family, device):
    prefix = encode_prefix(vocab, steps, family)
    x = torch.tensor([prefix], dtype=torch.long, device=device)
    lp = torch.log_softmax(model(x)[0][0, -1], dim=-1)
    ids = lp.topk(12).indices.tolist()
    return [vocab.id2step[i] for i in ids if vocab.id2step[i] not in _SPECIAL][:5]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    vocab = build_vocab()
    model, _ = I.load_lstm(vocab, device, ckpt_name="lstm_canonical_best.pt")
    glob_map, fam_map = build_majority_maps()
    print("Majority maps built.")

    valid_rows, _ = I.build_self_eval(vocab, use_original_names=True)
    print(f"Eval examples: {len(valid_rows)}\n")

    rows_canon, rows_glob, rows_fam = [], [], []
    for r in valid_rows:
        truth = r.get("_ACTUAL_NEXT_STEP", "")
        if not truth:
            continue
        family = r["FAMILY"].lower()
        steps = parse_pipe_sequence(r["PARTIAL_SEQUENCE"])
        if family not in ("mosfet", "igbt", "ic"):
            family = I.route_family(model, "lstm", vocab, steps, device)
        c5 = canonical_top5(model, vocab, steps, family, device)
        rows_canon.append(([variant(c, family, glob_map, fam_map, "canonical") for c in c5], truth))
        rows_glob.append(([variant(c, family, glob_map, fam_map, "global") for c in c5], truth))
        rows_fam.append(([variant(c, family, glob_map, fam_map, "family") for c in c5], truth))

    print(f"{'Strategy':<34}{'Top-1':>8}{'Top-3':>8}{'Top-5':>8}{'MRR':>8}")
    print("-" * 66)
    for name, rows in [("A. canonical name (no override)", rows_canon),
                       ("B. global most-freq variant", rows_glob),
                       ("C. family most-freq variant", rows_fam)]:
        t1, t3, t5, mrr = score(rows)
        print(f"{name:<34}{t1*100:>7.1f}%{t3*100:>7.1f}%{t5*100:>7.1f}%{mrr:>8.3f}")
    print("-" * 66)
    print("Reference: dual ensemble 70.0% | generator uses UNIFORM rng.choice (no signal)")


if __name__ == "__main__":
    main()
