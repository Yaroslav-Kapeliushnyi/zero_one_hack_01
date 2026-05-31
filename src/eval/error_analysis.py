"""
Error analysis on Task 1 next-step prediction failures.
Runs on our val set to understand WHY the model fails.

Questions:
  1. Which families fail most?
  2. Which steps are hardest to predict?
  3. How often is the correct answer in top-5 but not top-1? (ranking issue)
  4. How often are multiple valid next steps possible? (grammar ambiguity)
  5. Does failure correlate with how far back the "deciding" context is?
  6. Which steps does the model confuse for which?

Usage:
    python src/eval/error_analysis.py
"""

import sys, json
from pathlib import Path
from collections import defaultdict, Counter

import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data import build_vocab, SequenceDataset, train_val_split, encode_prefix

CKPT_DIR = ROOT / "checkpoints"
SPECIAL = {"[PAD]","[UNK]","[BOS]","[EOS]","[CLS]","[MOSFET]","[IGBT]","[IC]"}


def load_ensemble(vocab, device):
    from models.lstm_baseline import LSTMModel
    from models.gpt_model import build_gpt_model
    from models.markov import MarkovModel

    ckpt = torch.load(CKPT_DIR / "lstm_30k_pure_best.pt", map_location=device, weights_only=False)
    a = ckpt["args"]
    lstm = LSTMModel(len(vocab), a["embed"], a["hidden"], a["layers"],
                     a["dropout"], vocab.pad_id).to(device)
    lstm.load_state_dict(ckpt["model_state"])
    lstm.eval()

    ckpt2 = torch.load(CKPT_DIR / "gpt_best.pt", map_location=device, weights_only=False)
    a2 = ckpt2["args"]
    gpt = build_gpt_model(len(vocab), vocab.bos_id, vocab.eos_id,
                          a2["n_layer"], a2["n_head"], a2["n_embd"]).to(device)
    gpt.load_state_dict(ckpt2["model_state"])
    gpt.eval()

    markov = MarkovModel.load(CKPT_DIR / "markov_order3.pkl")
    return lstm, gpt, markov


@torch.no_grad()
def ensemble_top5(lstm, gpt, markov, vocab, prefix_ids, device):
    x = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    lstm_lp = torch.log_softmax(lstm(x)[0][0, -1], dim=-1).cpu()
    gpt_lp  = torch.log_softmax(gpt(x).logits[0, -1], dim=-1).cpu()

    steps = [vocab.id2step[i] for i in prefix_ids if vocab.id2step[i] not in SPECIAL]
    markov_lp = torch.full((len(vocab),), -20.0)
    for rank, step in enumerate(markov.predict_next_top_k(steps, k=len(vocab))):
        if step in vocab.step2id:
            markov_lp[vocab.step2id[step]] = -rank * 0.5

    combined = 0.45 * lstm_lp + 0.35 * gpt_lp + 0.20 * markov_lp
    top_ids  = combined.topk(10).indices.tolist()
    return [vocab.id2step[i] for i in top_ids if vocab.id2step[i] not in SPECIAL][:5], combined


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    vocab   = build_vocab()
    dataset = SequenceDataset(vocab)
    _, val_ds = train_val_split(dataset)

    lstm, gpt, markov = load_ensemble(vocab, device)

    # ── Collect all training next-step counts for ambiguity detection ─────────
    train_next = defaultdict(Counter)   # last_3_steps → Counter(next_step)
    for idx in range(len(dataset.samples)):
        if idx in set(val_ds.indices):
            continue
        ids   = dataset.samples[idx]
        steps = [vocab.id2step[i] for i in ids if vocab.id2step[i] not in SPECIAL]
        for i in range(3, len(steps)):
            ctx = tuple(steps[i-3:i])
            train_next[ctx][steps[i]] += 1

    # ── Evaluate ───────────────────────────────────────────────────────────────
    results = []
    print("Evaluating 300 val sequences × 60%/80% = 600 examples...")

    for seq_i, idx in enumerate(val_ds.indices[:300]):
        ids   = dataset.samples[idx]
        steps = [vocab.id2step[i] for i in ids if vocab.id2step[i] not in SPECIAL]
        family = vocab.id2step[ids[1]].strip("[]").lower()

        for frac in (0.6, 0.8):
            cut = max(1, int(len(steps) * frac))
            if cut >= len(steps):
                continue

            prefix   = steps[:cut]
            true_next = steps[cut]
            prefix_ids = encode_prefix(vocab, prefix, family)

            top5, scores = ensemble_top5(lstm, gpt, markov, vocab, prefix_ids, device)
            top1_correct = (top5[0] == true_next) if top5 else False
            in_top5      = true_next in top5

            # Ambiguity: how many distinct next steps follow this 3-gram in training?
            ctx3 = tuple(prefix[-3:])
            next_counts = train_next.get(ctx3, Counter())
            n_valid_next = len(next_counts)
            total_ctx    = sum(next_counts.values())
            true_freq    = next_counts.get(true_next, 0) / max(total_ctx, 1)

            # How far back is the last occurrence of true_next in prefix?
            last_seen_at = None
            for j in range(len(prefix)-1, -1, -1):
                if prefix[j] == true_next:
                    last_seen_at = len(prefix) - j
                    break

            # Model confidence for true_next
            true_id   = vocab.step2id.get(true_next, vocab.unk_id)
            true_score = scores[true_id].item() if true_id < len(scores) else -99

            results.append({
                "family":       family,
                "frac":         frac,
                "prefix_len":   len(prefix),
                "seq_len":      len(steps),
                "true_next":    true_next,
                "pred_top1":    top5[0] if top5 else "",
                "top1_correct": top1_correct,
                "in_top5":      in_top5,
                "n_valid_next": n_valid_next,
                "true_freq":    round(true_freq, 3),
                "true_score":   round(true_score, 3),
                "last_seen_at": last_seen_at,
                "top5":         top5,
            })

        if (seq_i + 1) % 50 == 0:
            print(f"  {seq_i+1}/300 sequences done...")

    # ── Analysis ───────────────────────────────────────────────────────────────
    total   = len(results)
    correct = sum(r["top1_correct"] for r in results)
    top5_ok = sum(r["in_top5"]      for r in results)

    print(f"\n{'='*60}")
    print(f"OVERALL: Top-1={correct/total*100:.1f}%  Top-5={top5_ok/total*100:.1f}%  ({total} examples)")

    # 1. By family
    print(f"\n── By family ──")
    for fam in ("mosfet", "igbt", "ic"):
        sub = [r for r in results if r["family"] == fam]
        if sub:
            acc = sum(r["top1_correct"] for r in sub) / len(sub)
            print(f"  {fam:8s}: Top-1={acc*100:.1f}%  (n={len(sub)})")

    # 2. By completion fraction
    print(f"\n── By completion fraction ──")
    for frac in (0.6, 0.8):
        sub = [r for r in results if r["frac"] == frac]
        acc = sum(r["top1_correct"] for r in sub) / len(sub)
        print(f"  {frac:.0%} given: Top-1={acc*100:.1f}%  (n={len(sub)})")

    # 3. Failures in top-5 vs not in top-5
    failures = [r for r in results if not r["top1_correct"]]
    in_top5_but_wrong = sum(r["in_top5"] for r in failures)
    not_in_top5       = sum(not r["in_top5"] for r in failures)
    print(f"\n── Failure breakdown ({len(failures)} failures) ──")
    print(f"  Correct answer in top-5 but not #1: {in_top5_but_wrong}  ({in_top5_but_wrong/len(failures)*100:.1f}%) → ranking problem")
    print(f"  Correct answer not in top-5 at all: {not_in_top5}  ({not_in_top5/len(failures)*100:.1f}%) → knowledge gap")

    # 4. Ambiguity analysis
    print(f"\n── Ambiguity (n_valid_next = distinct next steps seen after same 3-gram) ──")
    for n in (1, 2, 3, 5, 10):
        sub = [r for r in results if r["n_valid_next"] <= n]
        acc = sum(r["top1_correct"] for r in sub) / max(len(sub), 1)
        print(f"  n_valid_next ≤ {n:2d}: acc={acc*100:.1f}%  (n={len(sub)})")

    unambiguous = [r for r in results if r["n_valid_next"] == 1]
    ambiguous   = [r for r in results if r["n_valid_next"] > 3]
    if unambiguous:
        acc_u = sum(r["top1_correct"] for r in unambiguous) / len(unambiguous)
        print(f"\n  Unambiguous (1 valid next): acc={acc_u*100:.1f}%  (n={len(unambiguous)})")
    if ambiguous:
        acc_a = sum(r["top1_correct"] for r in ambiguous) / len(ambiguous)
        print(f"  Highly ambiguous (>3 valid): acc={acc_a*100:.1f}%  (n={len(ambiguous)})")

    # 5. Most common failures (what step did we predict instead?)
    print(f"\n── Top 15 most confused pairs (true → predicted) ──")
    confusions = Counter()
    for r in failures:
        if r["pred_top1"]:
            confusions[(r["true_next"], r["pred_top1"])] += 1
    for (true, pred), cnt in confusions.most_common(15):
        print(f"  {cnt:3d}×  '{true}'")
        print(f"         predicted: '{pred}'")

    # 6. Hardest steps to predict
    print(f"\n── Hardest steps to predict (min 5 occurrences, worst acc) ──")
    step_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        s = step_stats[r["true_next"]]
        s["total"]   += 1
        s["correct"] += int(r["top1_correct"])
    hard = [(step, s) for step, s in step_stats.items() if s["total"] >= 5]
    hard.sort(key=lambda x: x[1]["correct"] / x[1]["total"])
    for step, s in hard[:15]:
        acc = s["correct"] / s["total"]
        print(f"  {acc*100:5.1f}%  '{step}'  ({s['total']} examples)")

    # 7. Save full results
    out = ROOT / "eval_results" / "error_analysis.json"
    out.parent.mkdir(exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nFull results saved → {out}")


if __name__ == "__main__":
    run()
