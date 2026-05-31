"""
Generate the HYBRID_SIMPLE Task-1 submission on the OFFICIAL eval input.

This is the architecture shown in the slide diagram:
  RANK_1   = 3-model ensemble (LSTM-Attn + GPT + Markov, canonical) → decanonicalized top variant
  RANK_2-5 = synonym variants of the predicted steps (static VARIANTS_OF lookup, NO model)

No original-vocab LSTM, no dual ensemble. Metric-identical to the dual-tail hybrid on self-eval
(71.8/99.8/100/0.857) but simpler, and the tail never drops a candidate (no else-branch).

Writes results/SUBMISSION_nextstep_hybrid_simple.csv. Does NOT overwrite SUBMISSION_nextstep.csv
(the swap is done explicitly after verification).
Run on cluster:
  ~/.pixi/bin/pixi run --as-is --manifest-path <repo>/pixi.toml python src/eval/gen_hybrid_simple_submission.py
"""
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import eval.infer as infer  # noqa: E402
from data import build_vocab, encode_prefix, CANONICAL_STEPS, VARIANTS_OF  # noqa: E402

CKPT = infer.CKPT_DIR
IN = ROOT / "data" / "eval_input_valid.csv"
OUT = ROOT / "results" / "SUBMISSION_nextstep_hybrid_simple.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KNOWN = ("mosfet", "igbt", "ic")


def hybrid_simple(ens3):
    """RANK_1 = ens3 top-1 (preserved); RANK_2-5 = VARIANTS_OF enumeration of ens3's steps."""
    r1 = ens3[0] if ens3 else ""
    out, seen = [r1], {r1}
    for step in ens3:                       # ens3 rank order
        canon = CANONICAL_STEPS.get(step, step)
        for var in VARIANTS_OF.get(canon, [canon]):
            if var not in seen:
                out.append(var)
                seen.add(var)
            if len(out) == 5:
                return out
    while len(out) < 5:
        out.append("")
    return out


@torch.no_grad()
def main():
    print(f"Device: {DEVICE}")
    vocab_canon = build_vocab()
    print(f"canonical vocab {len(vocab_canon)}")

    lstm_attn, _ = infer.load_lstm_attn(vocab_canon, DEVICE)
    gpt_canon, _ = infer.load_gpt(vocab_canon, DEVICE)
    markov_ck = ("markov_canonical.pkl" if (CKPT / "markov_canonical.pkl").exists()
                 else "markov_order3.pkl")
    markov_ens = infer.load_markov(markov_ck)
    markov_orig = infer.load_markov("markov_orig_names.pkl"
                                    if (CKPT / "markov_orig_names.pkl").exists()
                                    else "markov_order3.pkl")
    print(f"models: lstm_attn + gpt_canonical + {markov_ck} (decanon via markov_orig)")

    rows = list(csv.DictReader(open(IN, newline="")))
    print(f"Official input rows: {len(rows)}")

    out_rows = []
    routed = 0
    for r in rows:
        family = r["FAMILY"].lower()
        steps = infer.parse_pipe_sequence(r["PARTIAL_SEQUENCE"])
        if family not in KNOWN:
            family = infer.route_family(lstm_attn, "lstm_attn", vocab_canon, steps, DEVICE)
            routed += 1
        prefix_canon = encode_prefix(vocab_canon, steps, family)
        ens3 = infer.predict_next_top5_context_aware(
            lstm_attn, gpt_canon, markov_ens, vocab_canon, prefix_canon, DEVICE,
            family=family, original_prefix=steps, markov_orig=markov_orig)
        h = hybrid_simple(ens3)
        out_rows.append({"EXAMPLE_ID": r["EXAMPLE_ID"], "RANK_1": h[0], "RANK_2": h[1],
                         "RANK_3": h[2], "RANK_4": h[3], "RANK_5": h[4]})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, ["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"])
        w.writeheader()
        w.writerows(out_rows)
    empty = sum(1 for r in out_rows if not r["RANK_1"])
    dup = sum(1 for r in out_rows if len({r["RANK_1"], r["RANK_2"], r["RANK_3"],
                                          r["RANK_4"], r["RANK_5"]}) < 5)
    print(f"routed: {routed} | empty RANK_1: {empty} | rows with dup ranks: {dup}")
    print(f"→ wrote {OUT} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
