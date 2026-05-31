#!/usr/bin/env python3
"""
eval_metrics.py — Evaluation Script for Industrial AI Track

Calculates all required metrics for:
  1. Next-Step Prediction   (--task next-step)
  2. Sequence Completion    (--task completion)
  3. Anomaly Detection      (--task anomaly)

Includes per-family and per-truncation-point breakdowns.

Usage:
  python eval_metrics.py --task next-step  --ground-truth gt.csv --predictions pred.csv
  python eval_metrics.py --task completion --ground-truth gt.csv --predictions pred.csv
  python eval_metrics.py --task anomaly    --ground-truth gt.csv --predictions pred.csv
"""

import argparse
import csv
import math
from collections import defaultdict


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def read_csv(filepath):
    with open(filepath, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def edit_distance(s1, s2):
    """Levenshtein distance between two token lists."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = (dp[i-1][j-1] if s1[i-1] == s2[j-1]
                        else 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]))
    return dp[m][n]


def calculate_roc_auc(y_true, y_scores):
    """ROC-AUC in pure Python (trapezoidal rule)."""
    if not y_true or not y_scores:
        return 0.0
    pairs = sorted(zip(y_true, y_scores), key=lambda x: x[1], reverse=True)
    num_pos = sum(1 for y, _ in pairs if y == 1)
    num_neg = len(pairs) - num_pos
    if num_pos == 0 or num_neg == 0:
        return 0.0
    auc, fp, tp, fp_prev, tp_prev = 0.0, 0, 0, 0, 0
    for i, (label, _) in enumerate(pairs):
        if label == 1: tp += 1
        else:          fp += 1
        if i == len(pairs) - 1 or pairs[i][1] != pairs[i+1][1]:
            auc += (fp - fp_prev) * (tp + tp_prev) / 2.0
            fp_prev, tp_prev = fp, tp
    return auc / (num_pos * num_neg)


def _print_breakdown(title, stats, fmt_fn):
    """Print a per-group breakdown table."""
    if not stats:
        return
    print(f"\n  Breakdown by {title}:")
    for key in sorted(stats):
        fmt_fn(key, stats[key])


# ---------------------------------------------------------------------------
# Task 1: Next-Step Prediction
# ---------------------------------------------------------------------------

def eval_next_step(gt_data, pred_data):
    """
    GT  columns : SEQUENCE_ID, ACTUAL_NEXT_STEP, [FAMILY], [COMPLETION_FRACTION]
    Pred columns: SEQUENCE_ID, PRED_1 .. PRED_5
    """
    gt_map = {r['SEQUENCE_ID']: r for r in gt_data}

    # Overall counters
    top1 = top3 = top5 = mrr_sum = total = 0

    # Per-family and per-fraction counters
    by_family   = defaultdict(lambda: [0, 0, 0, 0.0, 0])  # t1,t3,t5,mrr,n
    by_fraction = defaultdict(lambda: [0, 0, 0, 0.0, 0])

    for row in pred_data:
        sid = row['SEQUENCE_ID']
        if sid not in gt_map:
            continue
        gt  = gt_map[sid]
        actual = gt['ACTUAL_NEXT_STEP']
        preds  = [row.get(f'PRED_{i}', '') for i in range(1, 6)]

        h1 = int(preds[0] == actual)
        h3 = int(actual in preds[:3])
        h5 = int(actual in preds[:5])
        try:    rr = 1.0 / (preds.index(actual) + 1)
        except: rr = 0.0

        top1 += h1; top3 += h3; top5 += h5; mrr_sum += rr; total += 1

        fam  = gt.get('FAMILY', '')
        frac = gt.get('COMPLETION_FRACTION', '')
        if fam:
            c = by_family[fam]
            c[0]+=h1; c[1]+=h3; c[2]+=h5; c[3]+=rr; c[4]+=1
        if frac:
            c = by_fraction[frac]
            c[0]+=h1; c[1]+=h3; c[2]+=h5; c[3]+=rr; c[4]+=1

    if total == 0:
        print("No matching IDs."); return

    print("\n=== Task 1: Next-Step Prediction ===")
    print(f"  Sequences : {total}")
    print(f"  Top-1 Acc : {top1/total:.4f}")
    print(f"  Top-3 Acc : {top3/total:.4f}")
    print(f"  Top-5 Acc : {top5/total:.4f}")
    print(f"  MRR       : {mrr_sum/total:.4f}")

    def fmt_ns(key, c):
        n = c[4]
        print(f"    {key:15s} Top-1={c[0]/n:.4f}  Top-3={c[1]/n:.4f}  "
              f"Top-5={c[2]/n:.4f}  MRR={c[3]/n:.4f}  (n={n})")

    _print_breakdown("family",            by_family,   fmt_ns)
    _print_breakdown("completion %",      by_fraction, fmt_ns)
    print("=" * 40)


# ---------------------------------------------------------------------------
# Task 2: Sequence Completion
# ---------------------------------------------------------------------------

def _get_3grams(seq):
    if len(seq) < 3:
        return set()
    return set(tuple(seq[i:i+3]) for i in range(len(seq) - 2))


def eval_completion(gt_data, pred_data):
    """
    GT  columns : SEQUENCE_ID, REMAINING_STEPS (pipe-sep), [FAMILY], [COMPLETION_FRACTION]
    Pred columns: SEQUENCE_ID, PREDICTED_STEPS (pipe-sep)
    """
    gt_map = {r['SEQUENCE_ID']: r for r in gt_data}

    exact = ned = tok_acc = blk_acc = total = 0
    by_family   = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0])  # exact,ned,tok,blk,n
    by_fraction = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0])

    for row in pred_data:
        sid = row['SEQUENCE_ID']
        if sid not in gt_map:
            continue
        gt       = gt_map[sid]
        actual   = [s for s in gt['REMAINING_STEPS'].split('|') if s]
        pred_raw = row.get('PREDICTED_STEPS', '')
        predicted = [s for s in pred_raw.split('|') if s] if pred_raw else []

        e  = int(actual == predicted)
        d  = edit_distance(actual, predicted)
        nd = d / max(len(actual), len(predicted), 1)
        ta = (sum(a==p for a,p in zip(actual, predicted)) / len(actual)
              if actual else 0.0)
        ab = _get_3grams(actual)
        pb = _get_3grams(predicted)
        ba = len(ab & pb) / len(ab) if ab else (1.0 if not pb else 0.0)

        exact += e; ned += nd; tok_acc += ta; blk_acc += ba; total += 1

        fam  = gt.get('FAMILY', '')
        frac = gt.get('COMPLETION_FRACTION', '')
        if fam:
            c = by_family[fam]
            c[0]+=e; c[1]+=nd; c[2]+=ta; c[3]+=ba; c[4]+=1
        if frac:
            c = by_fraction[frac]
            c[0]+=e; c[1]+=nd; c[2]+=ta; c[3]+=ba; c[4]+=1

    if total == 0:
        print("No matching IDs."); return

    print("\n=== Task 2: Sequence Completion ===")
    print(f"  Sequences          : {total}")
    print(f"  Exact Match Rate   : {exact/total:.4f}")
    print(f"  Norm Edit Distance : {ned/total:.4f}  (lower = better)")
    print(f"  Token Accuracy     : {tok_acc/total:.4f}")
    print(f"  Block Accuracy     : {blk_acc/total:.4f}")

    def fmt_cp(key, c):
        n = c[4]
        print(f"    {key:15s} ExactMatch={c[0]/n:.4f}  NED={c[1]/n:.4f}  "
              f"TokAcc={c[2]/n:.4f}  BlkAcc={c[3]/n:.4f}  (n={n})")

    _print_breakdown("family",       by_family,   fmt_cp)
    _print_breakdown("completion %", by_fraction, fmt_cp)
    print("=" * 40)


# ---------------------------------------------------------------------------
# Task 3: Anomaly Detection
# ---------------------------------------------------------------------------

def eval_anomaly(gt_data, pred_data):
    """
    GT  columns : SEQUENCE_ID, IS_ANOMALY (1=bad, 0=valid), [VIOLATED_RULE], [FAMILY]
    Pred columns: SEQUENCE_ID, ANOMALY_SCORE (float), [PREDICTED_RULE]
    """
    gt_map = {r['SEQUENCE_ID']: r for r in gt_data}

    y_true, y_scores, y_pred = [], [], []
    rule_correct = rule_total = 0
    by_family = defaultdict(lambda: {'yt': [], 'ys': [], 'yp': []})

    for row in pred_data:
        sid = row['SEQUENCE_ID']
        if sid not in gt_map:
            continue
        gt    = gt_map[sid]
        score = float(row.get('ANOMALY_SCORE', 0.5))
        label = int(gt['IS_ANOMALY'])
        pred_label = int(score >= 0.5)

        y_true.append(label); y_scores.append(score); y_pred.append(pred_label)

        fam = gt.get('FAMILY', '')
        if fam:
            by_family[fam]['yt'].append(label)
            by_family[fam]['ys'].append(score)
            by_family[fam]['yp'].append(pred_label)

        # Rule attribution (only for actual anomalies)
        if label == 1 and gt.get('VIOLATED_RULE', ''):
            rule_total += 1
            if row.get('PREDICTED_RULE', '') == gt['VIOLATED_RULE']:
                rule_correct += 1

    total = len(y_true)
    if total == 0:
        print("No matching IDs."); return

    tp = sum(1 for yt,yp in zip(y_true,y_pred) if yt==1 and yp==1)
    tn = sum(1 for yt,yp in zip(y_true,y_pred) if yt==0 and yp==0)
    fp = sum(1 for yt,yp in zip(y_true,y_pred) if yt==0 and yp==1)
    fn = sum(1 for yt,yp in zip(y_true,y_pred) if yt==1 and yp==0)

    acc  = (tp+tn)/total
    prec = tp/(tp+fp) if (tp+fp) else 0.0
    rec  = tp/(tp+fn) if (tp+fn) else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    auc  = calculate_roc_auc(y_true, y_scores)

    print("\n=== Task 3: Anomaly Detection ===")
    print(f"  Sequences          : {total}")
    print(f"  Binary Accuracy    : {acc:.4f}")
    print(f"  Precision          : {prec:.4f}")
    print(f"  Recall             : {rec:.4f}")
    print(f"  F1 Score           : {f1:.4f}")
    print(f"  ROC-AUC            : {auc:.4f}")
    if rule_total > 0:
        print(f"  Rule Attribution   : {rule_correct/rule_total:.4f}  ({rule_correct}/{rule_total})")
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted 1  Predicted 0")
    print(f"    Actual 1  | {tp:11d}  {fn:11d}")
    print(f"    Actual 0  | {fp:11d}  {tn:11d}")

    if by_family:
        print(f"\n  Breakdown by family:")
        for fam in sorted(by_family):
            d   = by_family[fam]
            yt, ys, yp = d['yt'], d['ys'], d['yp']
            n   = len(yt)
            ftp = sum(1 for a,p in zip(yt,yp) if a==1 and p==1)
            ftn = sum(1 for a,p in zip(yt,yp) if a==0 and p==0)
            ffp = sum(1 for a,p in zip(yt,yp) if a==0 and p==1)
            ffn = sum(1 for a,p in zip(yt,yp) if a==1 and p==0)
            facc = (ftp+ftn)/n if n else 0
            fprec = ftp/(ftp+ffp) if (ftp+ffp) else 0
            frec  = ftp/(ftp+ffn) if (ftp+ffn) else 0
            ff1   = 2*fprec*frec/(fprec+frec) if (fprec+frec) else 0
            fauc  = calculate_roc_auc(yt, ys)
            print(f"    {fam:8s}  Acc={facc:.4f}  P={fprec:.4f}  R={frec:.4f}  "
                  f"F1={ff1:.4f}  AUC={fauc:.4f}  (n={n})")
    print("=" * 40)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Official evaluation script — Industrial AI Track")
    parser.add_argument("--task", choices=["next-step", "completion", "anomaly"],
                        required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--predictions",  required=True)
    args = parser.parse_args()

    gt   = read_csv(args.ground_truth)
    pred = read_csv(args.predictions)

    if args.task == "next-step":
        eval_next_step(gt, pred)
    elif args.task == "completion":
        eval_completion(gt, pred)
    elif args.task == "anomaly":
        eval_anomaly(gt, pred)


if __name__ == "__main__":
    main()
