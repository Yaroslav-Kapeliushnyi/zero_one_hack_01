"""
Task-1 SAME-SPLIT SHOOTOUT — settles "is ~70% an inference bug or genuinely Markovian?"

Every model is scored on BYTE-IDENTICAL examples: the exact 600 the dual ensemble's
self-eval used (train_val_split seed42 random_split on the combined dataset,
val_ds.indices[:300] x {0.6, 0.8} cut points), ORIGINAL-name ground truth.

Models (all decode to ORIGINAL step names so GT comparison is apples-to-apples):
  - naive         : global most-frequent next step (organizer BASELINE)
  - trigram       : Markov order-3, original names, trained on the SAME train split
  - raw_lstm      : lstm_30k_best (orig vocab 206) queried DIRECTLY (no ensemble, no decanon)
                    == the ensemble's own original_model. THIS is the bug-vs-artifact number.
  - gpt           : gpt_best (orig vocab 206) queried directly
  - dual_ensemble : the current SUBMISSION model (predict_next_top5_dual), as run() computes it

Outputs:
  results/shootout_task1.csv          — one row per example, every model's top-1 + correct flag
  results/shootout_task1_summary.json — Top-1/3/5/MRR overall + per cut (60/80) + per family

Run on cluster:
  ~/.pixi/bin/pixi run --manifest-path <repo>/pixi.toml python src/eval/shootout_task1.py
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import eval.infer as infer  # noqa: E402  (reuse exact encode/predict machinery)
from data import (build_vocab, SequenceDataset, train_val_split,  # noqa: E402
                  load_sequences, FAMILY_FILES, encode_prefix)

CKPT = infer.CKPT_DIR
OUT_CSV = ROOT / "results" / "shootout_task1.csv"
OUT_JSON = ROOT / "results" / "shootout_task1_summary.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Build the EXACT 600 examples (same split as build_self_eval) ──────────────
def build_examples():
    """Returns (train_seqs_orig, examples). examples: list of dicts with
    EXAMPLE_ID, FAMILY, CUT, PREFIX_STEPS (original names), TRUTH (original name).
    Reconstructs build_self_eval(use_original_names=True) exactly + train split."""
    vocab_canon = build_vocab()
    dataset = SequenceDataset(vocab_canon)
    train_ds, val_ds = train_val_split(dataset)

    # Parallel original-name sequences, same family order/insertion order as dataset.
    all_orig = []  # [(family, [orig steps])]
    for fam in ("mosfet", "igbt", "ic"):
        seqs = load_sequences(FAMILY_FILES[fam], apply_canonical=False)
        for seq in seqs.values():
            all_orig.append((fam, seq))

    train_seqs = [all_orig[i][1] for i in train_ds.indices]

    examples = []
    for i, idx in enumerate(val_ds.indices[:300]):
        family, steps = all_orig[idx]
        for frac in (0.6, 0.8):
            cut = max(1, int(len(steps) * frac))
            if cut >= len(steps):
                continue  # no next step to predict
            examples.append({
                "EXAMPLE_ID": f"val_{i:04d}_f{int(frac*100)}",
                "FAMILY": family,
                "CUT": int(frac * 100),
                "PREFIX_STEPS": steps[:cut],
                "TRUTH": steps[cut],
            })
    return vocab_canon, train_seqs, examples


# ── Simple n-gram + naive on original names (same train split) ────────────────
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
                self.table[tuple(seq[i:i + self.order])][seq[i + self.order]] += 1

    def top5(self, prefix):
        if len(prefix) >= self.order:
            ctx = tuple(prefix[-self.order:])
            c = self.table.get(ctx)
            if c:
                top = [s for s, _ in c.most_common(5)]
                for s, _ in self.fallback.most_common():
                    if len(top) == 5:
                        break
                    if s not in top:
                        top.append(s)
                return top
        return [s for s, _ in self.fallback.most_common(5)]


class Naive:
    def __init__(self):
        self.t5 = []

    def train(self, sequences):
        c = Counter()
        for seq in sequences:
            for s in seq:
                c[s] += 1
        self.t5 = [s for s, _ in c.most_common(5)]

    def top5(self, prefix):
        return self.t5


# ── Neural top-5 (original-name decode) ───────────────────────────────────────
def neural_top5(model, mtype, vocab, prefix_ids, k=5):
    x = torch.tensor([prefix_ids], dtype=torch.long, device=DEVICE)
    logits = infer._get_logits_last(model, mtype, x)
    top_ids = logits.topk(20).indices.tolist()
    return [vocab.id2step[i] for i in top_ids if vocab.id2step[i] not in infer._SPECIAL][:k]


def load_gpt_orig(vocab_orig):
    from models.gpt_model import build_gpt_model
    ck = torch.load(CKPT / "gpt_best.pt", map_location=DEVICE, weights_only=False)
    a = ck["args"]
    assert ck["model_state"]["transformer.wte.weight"].shape[0] == len(vocab_orig), "gpt vocab mismatch"
    m = build_gpt_model(len(vocab_orig), vocab_orig.bos_id, vocab_orig.eos_id,
                        a["n_layer"], a["n_head"], a["n_embd"]).to(DEVICE)
    m.load_state_dict(ck["model_state"])
    m.eval()
    return m


# ── Scoring ───────────────────────────────────────────────────────────────────
def blank_acc():
    return {"t1": 0, "t3": 0, "t5": 0, "mrr": 0.0, "n": 0}


def add(acc, ranks, truth):
    acc["n"] += 1
    if ranks and ranks[0] == truth:
        acc["t1"] += 1
    if truth in ranks[:3]:
        acc["t3"] += 1
    if truth in ranks[:5]:
        acc["t5"] += 1
    if truth in ranks[:5]:
        acc["mrr"] += 1.0 / (ranks.index(truth) + 1)


def finalize(acc):
    n = max(acc["n"], 1)
    return {"Top1": acc["t1"] / n, "Top3": acc["t3"] / n,
            "Top5": acc["t5"] / n, "MRR": acc["mrr"] / n, "n": acc["n"]}


@torch.no_grad()
def main():
    print(f"Device: {DEVICE}")
    vocab_canon, train_seqs, examples = build_examples()
    print(f"Canonical vocab: {len(vocab_canon)} | train seqs: {len(train_seqs)} | "
          f"eval examples: {len(examples)}")

    vocab_orig = infer.build_orig_vocab()
    print(f"Original vocab: {len(vocab_orig)}")

    # Baselines / trigram
    naive = Naive(); naive.train(train_seqs)
    trigram = Markov(3); trigram.train(train_seqs)

    # Neural models
    raw_lstm, ll = infer.load_lstm(vocab_orig, DEVICE, ckpt_name="lstm_30k_best.pt")
    print(f"  raw_lstm = lstm_30k_best (orig vocab {len(vocab_orig)}, val_loss {ll:.4f})")
    gpt = load_gpt_orig(vocab_orig)
    print(f"  gpt = gpt_best (orig vocab {len(vocab_orig)})")

    canon_lstm, _ = infer.load_lstm(vocab_canon, DEVICE, ckpt_name="lstm_canonical_best.pt")
    family_priors = infer.build_family_variant_priors()
    print(f"  dual_ensemble = canonical({len(vocab_canon)}) + original({len(vocab_orig)})")

    # 3-model context-aware ensemble (the historical "71.5%" LSTM-Attn + GPT + Markov).
    # Replicates run()'s --model ensemble path EXACTLY, but scored vs original-name GT.
    lstm_attn, la_ll = infer.load_lstm_attn(vocab_canon, DEVICE)  # canonical lstm_attn
    try:
        gpt_canon, _ = infer.load_gpt(vocab_canon, DEVICE)        # canonical gpt (196)
    except RuntimeError as e:
        print(f"  ⚠ canonical GPT skipped: {e}")
        gpt_canon = None
    markov_ck = ("markov_canonical.pkl" if (CKPT / "markov_canonical.pkl").exists()
                 else "markov_order3.pkl")
    markov_ens = infer.load_markov(markov_ck)
    markov_orig = infer.load_markov("markov_orig_names.pkl"
                                    if (CKPT / "markov_orig_names.pkl").exists()
                                    else "markov_order3.pkl")
    print(f"  ensemble3 = lstm_attn + gpt_canonical + {markov_ck} (decanon via markov_orig)")

    print(f"  ensemble_plain = lstm_canonical + gpt_canonical + {markov_ck} (no attention, decanon)")

    MODELS = ["naive", "trigram", "raw_lstm", "gpt", "dual_ensemble",
              "ensemble3", "ensemble_plain"]

    # accumulators
    overall = {m: blank_acc() for m in MODELS}
    by_cut = {c: {m: blank_acc() for m in MODELS} for c in (60, 80)}
    by_fam = {f: {m: blank_acc() for m in MODELS} for f in ("mosfet", "igbt", "ic")}

    rows_out = []
    for ex in examples:
        fam, steps, truth, cut = ex["FAMILY"], ex["PREFIX_STEPS"], ex["TRUTH"], ex["CUT"]
        prefix_orig = encode_prefix(vocab_orig, steps, fam, apply_canonical=False)
        prefix_canon = encode_prefix(vocab_canon, steps, fam)  # apply_canonical=True

        preds = {
            "naive": naive.top5(steps),
            "trigram": trigram.top5(steps),
            "raw_lstm": neural_top5(raw_lstm, "lstm", vocab_orig, prefix_orig),
            "gpt": neural_top5(gpt, "gpt", vocab_orig, prefix_orig),
            "dual_ensemble": infer.predict_next_top5_dual(
                canon_lstm, raw_lstm, vocab_canon, vocab_orig,
                prefix_canon, prefix_orig, DEVICE, fam, family_priors=family_priors),
            "ensemble3": infer.predict_next_top5_context_aware(
                lstm_attn, gpt_canon, markov_ens, vocab_canon, prefix_canon, DEVICE,
                family=fam, original_prefix=steps, markov_orig=markov_orig),
            "ensemble_plain": infer.predict_next_top5_ensemble(
                canon_lstm, markov_ens, vocab_canon, prefix_canon, DEVICE,
                gpt_model=gpt_canon, family=fam, original_prefix=steps,
                markov_orig=markov_orig),
        }
        for m in MODELS:
            r = preds[m]
            add(overall[m], r, truth)
            add(by_cut[cut][m], r, truth)
            add(by_fam[fam][m], r, truth)

        row = {"EXAMPLE_ID": ex["EXAMPLE_ID"], "FAMILY": fam, "CUT": cut, "TRUTH": truth}
        for m in MODELS:
            top1 = preds[m][0] if preds[m] else ""
            row[f"{m}_top1"] = top1
            row[f"{m}_correct"] = int(top1 == truth)
        # keep top-5 for the models we diff
        for m in ("trigram", "raw_lstm", "dual_ensemble", "ensemble3", "ensemble_plain"):
            row[f"{m}_top5"] = "|".join(preds[m][:5])
        rows_out.append(row)

    # write per-example CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    summary = {
        "n_examples": len(examples),
        "overall": {m: finalize(overall[m]) for m in MODELS},
        "by_cut": {str(c): {m: finalize(by_cut[c][m]) for m in MODELS} for c in (60, 80)},
        "by_family": {fa: {m: finalize(by_fam[fa][m]) for m in MODELS} for fa in by_fam},
        "ckpts": {"raw_lstm": "lstm_30k_best.pt", "gpt": "gpt_best.pt",
                  "dual_canon": "lstm_canonical_best.pt", "dual_orig": "lstm_30k_best.pt"},
        "split": "train_val_split(seed42 random_split) val_ds.indices[:300] x {0.6,0.8}, original-name GT",
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))

    # console table
    def tbl(title, d):
        print(f"\n=== {title} ===")
        print(f"  {'model':<16}{'Top-1':>8}{'Top-3':>8}{'Top-5':>8}{'MRR':>8}{'n':>6}")
        for m in MODELS:
            v = d[m]
            print(f"  {m:<16}{v['Top1']*100:>7.1f}%{v['Top3']*100:>7.1f}%"
                  f"{v['Top5']*100:>7.1f}%{v['MRR']:>8.3f}{v['n']:>6}")

    tbl("OVERALL (600-ex same split, original-name GT)", summary["overall"])
    for c in (60, 80):
        tbl(f"CUT {c}%", summary["by_cut"][str(c)])
    for fa in ("mosfet", "igbt", "ic"):
        tbl(f"FAMILY {fa}", summary["by_family"][fa])

    print(f"\n→ wrote {OUT_CSV}")
    print(f"→ wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
