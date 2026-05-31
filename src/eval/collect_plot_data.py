"""
Collect plot data for the submission dashboard — LOCAL, no torch needed.

Produces (results/plots_data/):
  - loss_curves.json           : train/val loss per epoch for each neural model (from logs/)
  - task1_context_sweep.json   : Task-1 Top-1/3/5/MRR per family for
                                   * Naive most-frequent BASELINE
                                   * Markov order 1,2,3,5,8 (context-length sweep)
                                 → reproduces the Bayes ceiling empirically AND tests
                                   whether the recoverable signal is LONG-RANGE.

All metrics on ORIGINAL vocabulary, original-name GT (official convention).
Single deterministic split (seed 42, 90/10) shared with every other script.
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import random

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "tracks" / "industrial-infineon" / "training_data"
LOGS = BASE / "logs"
OUT = BASE / "results" / "plots_data"
OUT.mkdir(parents=True, exist_ok=True)

FILES = {"MOSFET": "MOSFET_variants.csv", "IGBT": "IGBT_variants.csv", "IC": "IC_variants.csv"}
ORDERS = [1, 2, 3, 5, 8]


def load_family():
    fam_seqs = defaultdict(list)
    for fam, fn in FILES.items():
        rows = defaultdict(list)
        with open(DATA / fn, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows[r["SEQUENCE_ID"]].append(r["STEP"].strip())
        for sid, steps in rows.items():
            fam_seqs[fam].append(steps)
    return fam_seqs


def split(seqs, seed=42):
    seqs = sorted(seqs, key=lambda s: s[0] if s else "")
    random.Random(seed).shuffle(seqs)
    n_val = len(seqs) // 10
    return seqs[n_val:], seqs[:n_val]


def build_markov(train, order):
    """context (last `order` steps) -> Counter(next_step). Plus a global next-step Counter."""
    table = defaultdict(Counter)
    glob = Counter()
    for steps in train:
        for i in range(1, len(steps)):
            ctx = tuple(steps[max(0, i - order):i])
            table[ctx][steps[i]] += 1
            glob[steps[i]] += 1
    return table, glob


def predict_markov(tables, glob_top, steps, i, order):
    """Backoff from `order` down to 1, then global most-frequent. Return top-5 list."""
    for o in range(order, 0, -1):
        if o not in tables:
            continue
        ctx = tuple(steps[max(0, i - o):i])
        c = tables[o].get(ctx)
        if c:
            top = [s for s, _ in c.most_common(5)]
            if len(top) < 5:
                top += [s for s in glob_top if s not in top][:5 - len(top)]
            return top[:5]
    return glob_top[:5]


def naive_topk(glob_top):
    return glob_top[:5]


def score(preds_truths):
    t1 = t3 = t5 = 0
    mrr = 0.0
    n = len(preds_truths)
    for ranks, truth in preds_truths:
        if ranks and ranks[0] == truth:
            t1 += 1
        if truth in ranks[:3]:
            t3 += 1
        if truth in ranks[:5]:
            t5 += 1
        if truth in ranks:
            mrr += 1.0 / (ranks.index(truth) + 1)
    return {"top1": t1 / n, "top3": t3 / n, "top5": t5 / n, "mrr": mrr / n, "n": n}


def task1():
    fam_seqs = load_family()
    result = {}
    for fam, seqs in fam_seqs.items():
        train, val = split(seqs)
        # build markov tables for all orders + global frequency on TRAIN
        tables = {o: build_markov(train, o)[0] for o in ORDERS}
        glob = build_markov(train, 1)[1]
        glob_top = [s for s, _ in glob.most_common()]

        # naive baseline = predict globally most-frequent step
        naive = [(naive_topk(glob_top), steps[i])
                 for steps in val for i in range(1, len(steps))]
        fam_res = {"naive_most_frequent": score(naive)}
        for o in ORDERS:
            pt = [(predict_markov(tables, glob_top, steps, i, o), steps[i])
                  for steps in val for i in range(1, len(steps))]
            fam_res[f"markov_order{o}"] = score(pt)
        result[fam] = fam_res
        print(f"\n=== {fam} (train={len(train)} val={len(val)}, "
              f"eval positions={fam_res['naive_most_frequent']['n']}) ===")
        print(f"  {'model':<22}{'Top-1':>8}{'Top-3':>8}{'Top-5':>8}{'MRR':>8}")
        for k, v in fam_res.items():
            print(f"  {k:<22}{v['top1']*100:>7.1f}%{v['top3']*100:>7.1f}%"
                  f"{v['top5']*100:>7.1f}%{v['mrr']:>8.3f}")
    (OUT / "task1_context_sweep.json").write_text(json.dumps(result, indent=2))
    print(f"\n→ wrote {OUT/'task1_context_sweep.json'}")
    return result


def task1_boundary(fracs=(0.2, 0.4, 0.6, 0.8)):
    """Official convention: ONE next-step prediction at each completion-fraction
    boundary (matches how our neural model's 70% was measured). Per family."""
    fam_seqs = load_family()
    result = {}
    for fam, seqs in fam_seqs.items():
        train, val = split(seqs)
        tables = {o: build_markov(train, o)[0] for o in ORDERS}
        glob = build_markov(train, 1)[1]
        glob_top = [s for s, _ in glob.most_common()]
        per_frac = {}
        for frac in fracs:
            naive, mk = [], {o: [] for o in ORDERS}
            for steps in val:
                if len(steps) < 3:
                    continue
                i = max(1, int(round(frac * len(steps))))
                if i >= len(steps):
                    continue
                truth = steps[i]
                naive.append((glob_top[:5], truth))
                for o in ORDERS:
                    mk[o].append((predict_markov(tables, glob_top, steps, i, o), truth))
            per_frac[f"{int(frac*100)}%"] = {
                "naive_most_frequent": score(naive),
                **{f"markov_order{o}": score(mk[o]) for o in ORDERS},
            }
        result[fam] = per_frac
        print(f"\n=== {fam} — boundary convention (Top-1 by completion fraction) ===")
        hdr = "  ".join(f"{int(f*100)}%" for f in fracs)
        print(f"  {'model':<22}{hdr:>26}")
        models = ["naive_most_frequent"] + [f"markov_order{o}" for o in ORDERS]
        for m in models:
            cells = "  ".join(f"{per_frac[f'{int(f*100)}%'][m]['top1']*100:>5.1f}%" for f in fracs)
            print(f"  {m:<22}{cells:>26}")
    (OUT / "task1_boundary.json").write_text(json.dumps(result, indent=2))
    print(f"\n→ wrote {OUT/'task1_boundary.json'}")
    return result


def loss_curves():
    curves = {}
    name_map = {"lstm_log.json": "LSTM (3K)",
                "lstm_30k_pure_log.json": "LSTM (30K)",
                "gpt_log.json": "GPT Transformer"}
    for fn, label in name_map.items():
        p = LOGS / fn
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        curves[label] = [{"epoch": d["epoch"], "train": d.get("train"),
                          "val": d.get("val"), "val_acc": d.get("val_acc")} for d in data]
    (OUT / "loss_curves.json").write_text(json.dumps(curves, indent=2))
    print(f"→ wrote {OUT/'loss_curves.json'}  (models: {list(curves)})")
    return curves


if __name__ == "__main__":
    print("#" * 70)
    print("# TASK 1 — per-family context sweep (original vocab, original GT)")
    print("#" * 70)
    task1()
    print("\n" + "#" * 70)
    print("# TASK 1 — boundary convention (apples-to-apples with neural 70%)")
    print("#" * 70)
    task1_boundary()
    print("\n" + "#" * 70)
    print("# LOSS CURVES")
    print("#" * 70)
    loss_curves()
