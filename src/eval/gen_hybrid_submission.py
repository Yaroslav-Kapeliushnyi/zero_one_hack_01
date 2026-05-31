"""
Generate the HYBRID Task-1 submission on the OFFICIAL eval input (data/eval_input_valid.csv).

RANK_1   = ensemble3 (LSTM-Attn + GPT + Markov, canonical+decanon) top-1
RANK_2-5 = dual ensemble's variant-enumerated top-5, with ens3_r1 promoted to front
           (same rule validated in src/eval/task1_hybrid.py: ens3_r1 was in dual top-5
            on 600/600 self-eval rows → truth never dropped).

Writes results/SUBMISSION_nextstep_hybrid.csv (CANDIDATE — does NOT touch SUBMISSION_nextstep.csv).
Run on cluster:
  ~/.pixi/bin/pixi run --as-is --manifest-path <repo>/pixi.toml python src/eval/gen_hybrid_submission.py
"""
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import eval.infer as infer  # noqa: E402
from data import build_vocab, encode_prefix  # noqa: E402

CKPT = infer.CKPT_DIR
IN = ROOT / "data" / "eval_input_valid.csv"
OUT = ROOT / "results" / "SUBMISSION_nextstep_hybrid.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KNOWN = ("mosfet", "igbt", "ic")


def build_hybrid(ens3_top5, dual_top5):
    r1 = ens3_top5[0] if ens3_top5 else ""
    rest = [s for s in dual_top5 if s != r1] if r1 in dual_top5 else dual_top5[:4]
    out = ([r1] + rest)[:5]
    while len(out) < 5:
        out.append("")
    return out


@torch.no_grad()
def main():
    print(f"Device: {DEVICE}")
    vocab_canon = build_vocab()
    vocab_orig = infer.build_orig_vocab()
    print(f"canon vocab {len(vocab_canon)} | orig vocab {len(vocab_orig)}")

    canon_lstm, _ = infer.load_lstm(vocab_canon, DEVICE, ckpt_name="lstm_canonical_best.pt")
    raw_lstm, _ = infer.load_lstm(vocab_orig, DEVICE, ckpt_name="lstm_30k_best.pt")
    lstm_attn, _ = infer.load_lstm_attn(vocab_canon, DEVICE)
    gpt_canon, _ = infer.load_gpt(vocab_canon, DEVICE)
    markov_ck = ("markov_canonical.pkl" if (CKPT / "markov_canonical.pkl").exists()
                 else "markov_order3.pkl")
    markov_ens = infer.load_markov(markov_ck)
    markov_orig = infer.load_markov("markov_orig_names.pkl"
                                    if (CKPT / "markov_orig_names.pkl").exists()
                                    else "markov_order3.pkl")
    family_priors = infer.build_family_variant_priors()

    rows = list(csv.DictReader(open(IN, newline="")))
    print(f"Official input rows: {len(rows)}")

    out_rows = []
    routed = 0
    for r in rows:
        eid = r["EXAMPLE_ID"]
        family = r["FAMILY"].lower()
        steps = infer.parse_pipe_sequence(r["PARTIAL_SEQUENCE"])
        if family not in KNOWN:
            family = infer.route_family(canon_lstm, "lstm", vocab_canon, steps, DEVICE)
            routed += 1

        prefix_canon = encode_prefix(vocab_canon, steps, family)
        prefix_orig = encode_prefix(vocab_orig, steps, family, apply_canonical=False)

        ens3 = infer.predict_next_top5_context_aware(
            lstm_attn, gpt_canon, markov_ens, vocab_canon, prefix_canon, DEVICE,
            family=family, original_prefix=steps, markov_orig=markov_orig)
        dual = infer.predict_next_top5_dual(
            canon_lstm, raw_lstm, vocab_canon, vocab_orig,
            prefix_canon, prefix_orig, DEVICE, family, family_priors=family_priors)
        h = build_hybrid(ens3, dual)
        out_rows.append({"EXAMPLE_ID": eid, "RANK_1": h[0], "RANK_2": h[1],
                         "RANK_3": h[2], "RANK_4": h[3], "RANK_5": h[4]})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, ["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"routed (unknown family): {routed}")
    print(f"→ wrote {OUT} ({len(out_rows)} rows)")
    # quick sanity: any empty RANK_1?
    empty = sum(1 for r in out_rows if not r["RANK_1"])
    print(f"empty RANK_1: {empty}")


if __name__ == "__main__":
    main()
