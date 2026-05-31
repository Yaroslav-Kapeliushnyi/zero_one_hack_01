"""
Task-1 HYBRID — best-of-both, computed locally from results/shootout_task1.csv.

RANK_1  = ensemble3's top-1 (the better variant picker: 71.8% Top-1)
RANK_2-5 = the dual ensemble's variant-enumerated top-5 (which has 100% Top-5 coverage),
           with ensemble3's RANK_1 promoted to the front.

Rule (maximizes Top-5 retention):
  - if ens3_r1 in dual_top5:  [ens3_r1] + [s in dual_top5 if s != ens3_r1]   (keeps all 5 → 100% Top-5)
  - else:                     [ens3_r1] + dual_top5[:4]                       (prepend, drop dual rank-5)

Scored vs original-name GT on the SAME 600. Writes results/task1_hybrid.json.
Does NOT touch shootout_task1.csv or SUBMISSION_*.csv.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CSV = ROOT / "results" / "shootout_task1.csv"
OUT = ROOT / "results" / "task1_hybrid.json"


def build_hybrid(ens3_top5, dual_top5):
    r1 = ens3_top5[0] if ens3_top5 else ""
    if r1 in dual_top5:
        rest = [s for s in dual_top5 if s != r1]
    else:
        rest = dual_top5[:4]
    return ([r1] + rest)[:5]


def score(rows, model_or_fn):
    t1 = t3 = t5 = 0
    mrr = 0.0
    n = len(rows)
    for r in rows:
        truth = r["TRUTH"]
        ranks = model_or_fn(r)
        if ranks and ranks[0] == truth:
            t1 += 1
        if truth in ranks[:3]:
            t3 += 1
        if truth in ranks[:5]:
            t5 += 1
        if truth in ranks[:5]:
            mrr += 1.0 / (ranks.index(truth) + 1)
    return {"Top1": t1 / n, "Top3": t3 / n, "Top5": t5 / n, "MRR": mrr / n, "n": n}


def main():
    rows = list(csv.DictReader(open(CSV)))
    assert len(rows) == 600

    def split5(r, m):
        return [s for s in r[f"{m}_top5"].split("|") if s]

    def hybrid(r):
        return build_hybrid(split5(r, "ensemble3"), split5(r, "dual_ensemble"))

    def dual(r):
        return split5(r, "dual_ensemble")

    def ens3(r):
        return split5(r, "ensemble3")

    def tri(r):
        return split5(r, "trigram")

    models = {"trigram": tri, "dual_ensemble": dual, "ensemble3": ens3, "HYBRID": hybrid}

    # how often is ens3_r1 in dual_top5 (the safe case)?
    safe = sum(1 for r in rows if split5(r, "ensemble3")[0] in split5(r, "dual_ensemble"))
    # sanity: where the unsafe branch drops dual rank-5, did we ever lose the truth?
    lost = 0
    for r in rows:
        e5, d5 = split5(r, "ensemble3"), split5(r, "dual_ensemble")
        h = build_hybrid(e5, d5)
        if r["TRUTH"] in d5 and r["TRUTH"] not in h:
            lost += 1

    overall = {m: score(rows, fn) for m, fn in models.items()}

    def subset(pred):
        return [r for r in rows if pred(r)]
    by_cut = {c: {m: score(subset(lambda r, c=c: int(r["CUT"]) == c), fn)
                  for m, fn in models.items()} for c in (60, 80)}
    by_fam = {f: {m: score(subset(lambda r, f=f: r["FAMILY"] == f), fn)
                  for m, fn in models.items()} for f in ("mosfet", "igbt", "ic")}

    result = {
        "n": 600,
        "hybrid_rule": "RANK_1=ensemble3 top-1; RANK_2-5=dual variant list (ens3_r1 promoted)",
        "ens3_r1_in_dual_top5": f"{safe}/600",
        "truth_lost_by_dropping_dual_rank5": lost,
        "overall": overall,
        "by_cut": {str(c): by_cut[c] for c in (60, 80)},
        "by_family": by_fam,
    }
    OUT.write_text(json.dumps(result, indent=2))

    def tbl(title, d):
        print(f"\n=== {title} ===")
        print(f"  {'model':<16}{'Top-1':>8}{'Top-3':>8}{'Top-5':>8}{'MRR':>8}")
        for m in ("trigram", "dual_ensemble", "ensemble3", "HYBRID"):
            v = d[m]
            print(f"  {m:<16}{v['Top1']*100:>7.1f}%{v['Top3']*100:>7.1f}%"
                  f"{v['Top5']*100:>7.1f}%{v['MRR']:>8.3f}")

    print(f"ens3 RANK_1 already in dual top-5: {safe}/600 "
          f"({safe/6:.1f}% safe case) | truth ever lost: {lost}")
    tbl("OVERALL (600, original-name GT)", overall)
    for c in (60, 80):
        tbl(f"CUT {c}%", by_cut[c])
    for f in ("mosfet", "igbt", "ic"):
        tbl(f"FAMILY {f}", by_fam[f])
    print(f"\n→ wrote {OUT}")


if __name__ == "__main__":
    main()
