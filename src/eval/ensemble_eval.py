"""
Re-run the LSTM + GPT + Markov soft-vote ensemble (the historical "71.5%" model)
on the SAME 600 held-out examples, scored BOTH ways:
  - canonical GT   (synonym-agnostic — the lineage of the 71.5%/91.83% numbers)
  - original GT     (official convention — directly comparable to dual/Markov 70.0%)

Reuses the verified predict_next_top5_ensemble from infer.py.

Run on cluster:
  ~/.pixi/bin/pixi run python src/eval/ensemble_eval.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
from data import build_vocab, encode_prefix, parse_pipe_sequence, CANONICAL_STEPS
import eval.infer as I


def score(rows_ranks):
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

    # Ensemble components (canonical), exactly as infer.py --model ensemble loads them.
    lstm, _ = I.load_lstm(vocab, device, ckpt_name="lstm_canonical_best.pt")
    markov_ckpt = ("markov_canonical.pkl"
                   if (I.CKPT_DIR / "markov_canonical.pkl").exists() else "markov_order3.pkl")
    markov = I.load_markov(markov_ckpt)
    gpt = None
    for cand in ("gpt_canonical_best.pt", "gpt_best.pt"):
        if (I.CKPT_DIR / cand).exists():
            try:
                gpt, _ = I.load_gpt(vocab, device)
                break
            except Exception as e:
                print(f"  GPT {cand} skipped: {e}")
    print(f"Ensemble: LSTM + {'GPT + ' if gpt else '(no GPT) '}Markov({markov_ckpt})")

    # markov_orig=None → ensemble outputs CANONICAL names (no decanonicalization).
    # We score those canonical predictions against BOTH GT conventions below.
    markov_orig = None

    # Same 600 examples, ORIGINAL-name GT (official convention).
    valid_rows, _ = I.build_self_eval(vocab, use_original_names=True)
    print(f"Eval examples: {len(valid_rows)}\n")

    orig_rows, canon_rows = [], []
    for r in valid_rows:
        truth_orig = r.get("_ACTUAL_NEXT_STEP", "")
        if not truth_orig:
            continue
        truth_canon = CANONICAL_STEPS.get(truth_orig, truth_orig)
        family = r["FAMILY"].lower()
        steps = parse_pipe_sequence(r["PARTIAL_SEQUENCE"])
        prefix = encode_prefix(vocab, steps, family)
        if family not in ("mosfet", "igbt", "ic"):
            family = I.route_family(lstm, "lstm", vocab, steps, device)
            prefix = encode_prefix(vocab, steps, family)
        top5 = I.predict_next_top5_ensemble(
            lstm, markov, vocab, prefix, device, gpt_model=gpt,
            family=family, original_prefix=steps, markov_orig=markov_orig)
        orig_rows.append((top5, truth_orig))
        canon_rows.append((top5, truth_canon))

    print(f"{'GT convention':<26}{'Top-1':>8}{'Top-3':>8}{'Top-5':>8}{'MRR':>8}")
    print("-" * 58)
    t1, t3, t5, mrr = score(canon_rows)
    print(f"{'canonical GT (inflated)':<26}{t1*100:>7.1f}%{t3*100:>7.1f}%{t5*100:>7.1f}%{mrr:>8.3f}")
    t1, t3, t5, mrr = score(orig_rows)
    print(f"{'original GT (official)':<26}{t1*100:>7.1f}%{t3*100:>7.1f}%{t5*100:>7.1f}%{mrr:>8.3f}")
    print("\nReference (same 600, original GT): dual 70.0% | Markov-3 70.0%")
    print("The historical 71.5% was this ensemble on a DIFFERENT (pre-canonical) eval set.")


if __name__ == "__main__":
    main()
