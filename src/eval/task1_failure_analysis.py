"""
Task-1 FAILURE ANALYSIS on results/shootout_task1.csv — LOCAL, no torch/cluster.

Buckets the 180 dual_ensemble errors into:
  (a) synonym-sibling confusion  : canon(top1) == canon(truth)        → irreducible coin-flip
  (c) genuine ranking miss       : not (a), truth in top5 (rank 2-5)  → RECOVERABLE (rerank)
  (b) boundary / unseen          : not (a), truth NOT in top5         → mostly irreducible
      (sub-split by whether truth is an optional-category step: MEASURE/INSPECT/CLEAN)

Then characterizes the 70 examples where trigram beats the ensemble (trigram right,
ensemble wrong): by family, by cut, by step-category, by whether the ensemble error was
a synonym-sibling, and where truth ranked in the ensemble's top-5.

Decisive router test: partition the 600 examples into 6 (family x cut) cells. A static
router that picks the better model PER CELL scores sum(max(trigram, ensemble)) per cell.
Compare to 420 (either model alone) and to the 490 oracle (best-of-both per example).
If the per-cell router ~= 420 → the wins are within-cell random → no router helps.

Writes results/task1_failure_buckets.json.
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CSV = ROOT / "results" / "shootout_task1.csv"
OUT = ROOT / "results" / "task1_failure_buckets.json"

# Synonym groups copied verbatim from src/data.py CANONICAL_STEPS (avoids torch import).
CANONICAL_STEPS = {
    "STRIP PHOTORESIST":              "STRIP RESIST",
    "STRIP RESIST LEVEL 2":           "STRIP RESIST",
    "PASSIVATION ETCH PAD OPENING":   "PASSIVATION ETCH",
    "MEASURE SURFACE PLANARITY":      "MEASURE PLANARITY",
    "OPEN BOND PAD WINDOW":           "OPEN PAD WINDOW",
    "CMP INTERLAYER DIELECTRIC":      "CMP DIELECTRIC",
    "OPEN PAD WINDOW LITHO":          "PAD WINDOW LITHO",
    "VIA ETCH THROUGH DIELECTRIC":    "VIA ETCH",
    "DIELECTRIC ETCH VIA":            "VIA ETCH",
    "MEASURE FILM THICKNESS":         "MEASURE DIELECTRIC THICKNESS",
}

OPTIONAL_CATS = {"MEASURE", "INSPECT", "CLEAN"}


def canon(step):
    return CANONICAL_STEPS.get(step, step)


def category(step):
    """Coarse step category = first word (MEASURE/ETCH/STRIP/CMP/DEPOSIT/...)."""
    return step.split()[0] if step else "(empty)"


def main():
    rows = list(csv.DictReader(open(CSV)))
    n = len(rows)
    assert n == 600, f"expected 600, got {n}"

    def top5(r, m):
        return [s for s in r[f"{m}_top5"].split("|") if s]

    # ── A. Bucket the 180 ensemble errors ────────────────────────────────────
    errors = [r for r in rows if int(r["dual_ensemble_correct"]) == 0]
    buckets = {"a_synonym_sibling": [], "c_genuine_ranking_miss": [],
               "b_boundary_unseen": []}
    b_optional = b_nonoptional = 0
    for r in errors:
        truth = r["TRUTH"]
        t1 = r["dual_ensemble_top1"]
        t5 = top5(r, "dual_ensemble")
        if canon(t1) == canon(truth):
            buckets["a_synonym_sibling"].append(r["EXAMPLE_ID"])
        elif truth in t5:                       # rank 2-5 (t1 already != truth)
            buckets["c_genuine_ranking_miss"].append(r["EXAMPLE_ID"])
        else:
            buckets["b_boundary_unseen"].append(r["EXAMPLE_ID"])
            if category(truth) in OPTIONAL_CATS or category(t1) in OPTIONAL_CATS:
                b_optional += 1
            else:
                b_nonoptional += 1

    bucket_counts = {k: len(v) for k, v in buckets.items()}
    assert sum(bucket_counts.values()) == len(errors)

    # ── B. Trigram beats ensemble (70) ───────────────────────────────────────
    tri_wins = [r for r in rows
                if int(r["trigram_correct"]) == 1 and int(r["dual_ensemble_correct"]) == 0]
    ens_wins = [r for r in rows
                if int(r["dual_ensemble_correct"]) == 1 and int(r["trigram_correct"]) == 0]

    def breakdown(subset):
        by_fam = Counter(r["FAMILY"] for r in subset)
        by_cut = Counter(r["CUT"] for r in subset)
        by_cat = Counter(category(r["TRUTH"]) for r in subset)
        # for tri-wins: was the ensemble error a synonym sibling? where did truth rank in ens top5?
        sibling = 0
        truth_rank_in_ens = Counter()  # rank (1-5) or 'not_in_top5'
        for r in subset:
            truth = r["TRUTH"]
            if canon(r["dual_ensemble_top1"]) == canon(truth):
                sibling += 1
            t5 = top5(r, "dual_ensemble")
            truth_rank_in_ens[str(t5.index(truth) + 1) if truth in t5 else "not_in_top5"] += 1
        return {
            "n": len(subset),
            "by_family": dict(by_fam),
            "by_cut": {str(k): v for k, v in by_cut.items()},
            "by_truth_category": dict(by_cat.most_common()),
            "ensemble_err_was_synonym_sibling": sibling,
            "truth_rank_in_ensemble_top5": dict(truth_rank_in_ens),
        }

    tri_win_bd = breakdown(tri_wins)
    ens_win_bd = breakdown(ens_wins)

    # Baseline distribution of all 600 for non-uniformity comparison
    base_fam = Counter(r["FAMILY"] for r in rows)
    base_cut = Counter(r["CUT"] for r in rows)

    # ── C. Router test: per (family,cut) cell, pick better model ─────────────
    cells = defaultdict(lambda: {"n": 0, "trigram": 0, "ensemble": 0, "oracle": 0})
    for r in rows:
        key = f"{r['FAMILY']}_cut{r['CUT']}"
        c = cells[key]
        c["n"] += 1
        tc = int(r["trigram_correct"]); ec = int(r["dual_ensemble_correct"])
        c["trigram"] += tc
        c["ensemble"] += ec
        c["oracle"] += int(tc or ec)
    # static per-cell router = pick whichever model has higher Top-1 in that cell
    router_score = sum(max(c["trigram"], c["ensemble"]) for c in cells.values())
    oracle_score = sum(c["oracle"] for c in cells.values())
    tri_total = sum(c["trigram"] for c in cells.values())
    ens_total = sum(c["ensemble"] for c in cells.values())

    cell_table = {}
    for k, c in sorted(cells.items()):
        better = "trigram" if c["trigram"] > c["ensemble"] else (
                 "ensemble" if c["ensemble"] > c["trigram"] else "tie")
        cell_table[k] = {**c, "better": better}

    result = {
        "source_csv": str(CSV.relative_to(ROOT)),
        "n_examples": n,
        "ensemble_top1_correct": ens_total,
        "ensemble_errors": len(errors),
        "error_buckets": {
            "a_synonym_sibling_IRREDUCIBLE": bucket_counts["a_synonym_sibling"],
            "b_boundary_unseen_mostly_irreducible": bucket_counts["b_boundary_unseen"],
            "b_split": {"optional_category_truth_or_pred": b_optional,
                        "non_optional": b_nonoptional},
            "c_genuine_ranking_miss_RECOVERABLE": bucket_counts["c_genuine_ranking_miss"],
        },
        "trigram_beats_ensemble": tri_win_bd,
        "ensemble_beats_trigram": ens_win_bd,
        "baseline_dist": {"by_family": dict(base_fam),
                          "by_cut": {str(k): v for k, v in base_cut.items()}},
        "router_test": {
            "trigram_alone": tri_total,
            "ensemble_alone": ens_total,
            "static_family_cut_router": router_score,
            "oracle_best_of_both": oracle_score,
            "router_gain_over_best_single": router_score - max(tri_total, ens_total),
            "oracle_gain_over_best_single": oracle_score - max(tri_total, ens_total),
            "cells": cell_table,
        },
    }
    OUT.write_text(json.dumps(result, indent=2))

    # ── console report ────────────────────────────────────────────────────────
    print(f"Ensemble: {ens_total}/600 = {ens_total/6:.1f}% Top-1, {len(errors)} errors\n")
    print("ERROR BUCKETS (of 180):")
    print(f"  (a) synonym-sibling   [IRREDUCIBLE] : {bucket_counts['a_synonym_sibling']:>3}")
    print(f"  (b) boundary/unseen   [mostly irred]: {bucket_counts['b_boundary_unseen']:>3}"
          f"   (optional-cat {b_optional} / other {b_nonoptional})")
    print(f"  (c) genuine rank miss [RECOVERABLE] : {bucket_counts['c_genuine_ranking_miss']:>3}")
    print(f"      → recoverable share of errors    : "
          f"{bucket_counts['c_genuine_ranking_miss']}/{len(errors)} "
          f"= {bucket_counts['c_genuine_ranking_miss']/len(errors)*100:.1f}%\n")

    print("TRIGRAM BEATS ENSEMBLE (70):")
    print(f"  by family: {tri_win_bd['by_family']}   (baseline {dict(base_fam)})")
    print(f"  by cut   : {tri_win_bd['by_cut']}   (baseline {dict(base_cut)})")
    print(f"  ensemble err was synonym-sibling: {tri_win_bd['ensemble_err_was_synonym_sibling']}/70")
    print(f"  truth rank in ensemble top5     : {tri_win_bd['truth_rank_in_ensemble_top5']}")
    print(f"  top truth categories            : {dict(list(tri_win_bd['by_truth_category'].items())[:6])}\n")

    print("ROUTER TEST (decisive):")
    print(f"  trigram alone            : {tri_total}/600 = {tri_total/6:.1f}%")
    print(f"  ensemble alone           : {ens_total}/600 = {ens_total/6:.1f}%")
    print(f"  static (family,cut) router: {router_score}/600 = {router_score/6:.1f}%"
          f"   (gain {router_score-max(tri_total,ens_total):+d})")
    print(f"  oracle (best per example): {oracle_score}/600 = {oracle_score/6:.1f}%"
          f"   (gain {oracle_score-max(tri_total,ens_total):+d})")
    print("\n  per-cell (n / trigram / ensemble / better):")
    for k, c in cell_table.items():
        print(f"    {k:<16} n={c['n']:>3}  tri={c['trigram']:>3}  ens={c['ensemble']:>3}  → {c['better']}")

    print(f"\n→ wrote {OUT}")


if __name__ == "__main__":
    main()
