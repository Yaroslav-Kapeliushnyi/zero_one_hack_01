"""
Simplified hybrid — drops the original-vocab LSTM entirely.

  RANK_1  = ensemble3 top-1 (unchanged)
  RANK_2-5 = PLAIN variant-enumeration: walk ensemble3's predicted steps in rank order,
             expand each to its synonym group (VARIANTS_OF), dedup, fill to 5.
             No original LSTM, no dual ensemble — coverage comes from enumeration alone.

Compares against the current hybrid (which fills RANK_2-5 from the dual ensemble = canonical
LSTM + ORIGINAL LSTM) on the SAME 600 (results/shootout_task1.csv). Tests whether the original
LSTM contributes anything beyond what deterministic enumeration already gives.

Writes results/task1_hybrid_simple.json. Touches no submission files.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CSV = ROOT / "results" / "shootout_task1.csv"
OUT = ROOT / "results" / "task1_hybrid_simple.json"

# synonym groups (verbatim from src/data.py)
CANONICAL_STEPS = {
    "STRIP PHOTORESIST": "STRIP RESIST", "STRIP RESIST LEVEL 2": "STRIP RESIST",
    "PASSIVATION ETCH PAD OPENING": "PASSIVATION ETCH",
    "MEASURE SURFACE PLANARITY": "MEASURE PLANARITY",
    "OPEN BOND PAD WINDOW": "OPEN PAD WINDOW",
    "CMP INTERLAYER DIELECTRIC": "CMP DIELECTRIC",
    "OPEN PAD WINDOW LITHO": "PAD WINDOW LITHO",
    "VIA ETCH THROUGH DIELECTRIC": "VIA ETCH", "DIELECTRIC ETCH VIA": "VIA ETCH",
    "MEASURE FILM THICKNESS": "MEASURE DIELECTRIC THICKNESS",
}
VARIANTS_OF = {}
for _v, _c in CANONICAL_STEPS.items():
    VARIANTS_OF.setdefault(_c, [_c]).append(_v)


def split5(r, m):
    return [s for s in r[f"{m}_top5"].split("|") if s]


def hybrid_dual(ens3, dual):
    """current hybrid: ens3 r1 + dual variant list."""
    r1 = ens3[0] if ens3 else ""
    rest = [s for s in dual if s != r1] if r1 in dual else dual[:4]
    return ([r1] + rest)[:5]


def hybrid_simple(ens3):
    """ens3 r1 (preserved exactly) + plain VARIANTS_OF enumeration of ens3's predicted steps."""
    r1 = ens3[0] if ens3 else ""
    out, seen = [r1], {r1}
    for step in ens3:                       # ens3 rank order
        canon = CANONICAL_STEPS.get(step, step)
        for var in VARIANTS_OF.get(canon, [canon]):
            if var not in seen:
                out.append(var); seen.add(var)
            if len(out) == 5:
                return out
    while len(out) < 5:
        out.append("")
    return out


def score(rows, fn):
    t1 = t3 = t5 = 0; mrr = 0.0; n = len(rows)
    for r in rows:
        truth = r["TRUTH"]; ranks = fn(r)
        if ranks and ranks[0] == truth: t1 += 1
        if truth in ranks[:3]: t3 += 1
        if truth in ranks[:5]: t5 += 1
        if truth in ranks[:5]: mrr += 1.0 / (ranks.index(truth) + 1)
    return {"Top1": t1/n, "Top3": t3/n, "Top5": t5/n, "MRR": mrr/n}


def main():
    rows = list(csv.DictReader(open(CSV)))
    assert len(rows) == 600

    fns = {
        "dual_ensemble": lambda r: split5(r, "dual_ensemble"),
        "HYBRID (dual tail, +orig LSTM)": lambda r: hybrid_dual(split5(r, "ensemble3"), split5(r, "dual_ensemble")),
        "HYBRID_SIMPLE (enum tail, no orig LSTM)": lambda r: hybrid_simple(split5(r, "ensemble3")),
    }
    overall = {m: score(rows, fn) for m, fn in fns.items()}

    # do the two hybrids ever disagree?
    h_dual = [hybrid_dual(split5(r, "ensemble3"), split5(r, "dual_ensemble")) for r in rows]
    h_simp = [hybrid_simple(split5(r, "ensemble3")) for r in rows]
    r1_diff = sum(1 for a, b in zip(h_dual, h_simp) if a[0] != b[0])
    full_diff = sum(1 for a, b in zip(h_dual, h_simp) if a != b)
    # where does each hybrid place truth (rank) — compare MRR source
    simple_top5_miss = sum(1 for r, h in zip(rows, h_simp) if r["TRUTH"] not in h)

    result = {"overall": overall,
              "hybrid_dual_vs_simple_RANK1_differ": r1_diff,
              "hybrid_dual_vs_simple_full_differ": full_diff,
              "simple_top5_misses": simple_top5_miss}
    OUT.write_text(json.dumps(result, indent=2))

    print(f"RANK_1 identical between the two hybrids (by construction): "
          f"{600 - r1_diff}/600 same")
    print(f"full top-5 differs: {full_diff}/600   | simple Top-5 misses: {simple_top5_miss}/600\n")
    print(f"  {'model':<42}{'Top-1':>8}{'Top-3':>8}{'Top-5':>8}{'MRR':>8}")
    for m, v in overall.items():
        print(f"  {m:<42}{v['Top1']*100:>7.1f}%{v['Top3']*100:>7.1f}%"
              f"{v['Top5']*100:>7.1f}%{v['MRR']:>8.3f}")
    print(f"\n→ wrote {OUT}")


if __name__ == "__main__":
    main()
