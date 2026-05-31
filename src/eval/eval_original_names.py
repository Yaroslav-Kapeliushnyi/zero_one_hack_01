"""
True self-eval: original GT vs original predictions.

Loads val sequences with apply_canonical=False so GT uses original names
(STRIP PHOTORESIST, not STRIP RESIST). Runs canonical LSTM + markov_orig_names
tie-breaker so predictions also use original names. Apples to apples.

Usage:
    python src/eval/eval_original_names.py
"""

import sys, csv
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data import (Vocabulary, FAMILY_TOKEN, CANONICAL_STEPS, FAMILY_FILES,
                  load_sequences, train_val_split, encode_prefix)

CKPT_DIR = ROOT / "checkpoints"
SPECIAL = {"[PAD]","[UNK]","[BOS]","[EOS]","[CLS]","[MOSFET]","[IGBT]","[IC]"}
SPECIAL_IDS = None  # set after vocab built


# ── Build ORIGINAL-name vocab (apply_canonical=False) ────────────────────────
def build_orig_vocab():
    """Vocab over original names — same as canonical but retains all 198 steps."""
    from data import Vocabulary, SPECIAL_TOKENS
    seqs_orig = []
    for fam, path in FAMILY_FILES.items():
        d = load_sequences(path, apply_canonical=False)
        seqs_orig.extend(d.values())
    vocab = Vocabulary()
    vocab.build(seqs_orig)
    return vocab, seqs_orig


# ── Load val split with ORIGINAL names ───────────────────────────────────────
def load_orig_val(vocab_orig, seed=42):
    """Returns list of (original_steps, family_str) for the val split."""
    import random
    all_seqs, all_fams = [], []
    for fam, path in FAMILY_FILES.items():
        d = load_sequences(path, apply_canonical=False)
        for steps in d.values():
            all_seqs.append(steps)
            all_fams.append(fam)

    rng = random.Random(seed)
    indices = list(range(len(all_seqs)))
    rng.shuffle(indices)
    cut = int(len(indices) * 0.1)
    val_indices = set(indices[:cut])

    val_seqs = [(all_seqs[i], all_fams[i]) for i in val_indices]
    return val_seqs


# ── Build canonical vocab (for LSTM) ─────────────────────────────────────────
def build_canon_vocab():
    from data import build_vocab
    return build_vocab()


# ── Decanonicalize using original Markov ─────────────────────────────────────
def decanonicalize(step, original_prefix, markov_orig):
    from data import VARIANTS_OF
    variants = VARIANTS_OF.get(step)
    if not variants:
        return step
    if len(variants) == 1:
        return variants[0]
    ranked = markov_orig.predict_next_top_k(original_prefix, k=len(markov_orig.vocab) + 1)
    rank_of = {v: len(ranked) for v in variants}
    for rank, s in enumerate(ranked):
        if s in rank_of:
            rank_of[s] = rank
    return min(rank_of, key=rank_of.get)


@torch.no_grad()
def predict_top5_original(model, vocab_canon, prefix_ids, device,
                           original_prefix, markov_orig):
    """Predict top-5, decanonicalise to original names."""
    x = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    logits, _ = model(x)
    logits = logits[0, -1]
    top_ids = logits.topk(10).indices.tolist()
    canonical = [vocab_canon.id2step[i] for i in top_ids
                 if vocab_canon.id2step[i] not in SPECIAL][:5]
    return [decanonicalize(s, original_prefix, markov_orig) for s in canonical]


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Canonical vocab + LSTM
    vocab_canon = build_canon_vocab()
    from models.lstm_baseline import LSTMModel
    ckpt = torch.load(CKPT_DIR / "lstm_canonical_best.pt",
                      map_location=device, weights_only=False)
    a = ckpt["args"]
    model = LSTMModel(len(vocab_canon), a["embed"], a["hidden"],
                      a["layers"], a["dropout"], vocab_canon.pad_id).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded LSTM canonical (val_loss={ckpt['val_loss']:.4f})")

    # Original-name Markov for tie-breaker
    from models.markov import MarkovModel
    markov_orig = MarkovModel.load(CKPT_DIR / "markov_orig_names.pkl")
    print(f"Loaded Markov orig (vocab={len(markov_orig.vocab)})")

    # Val set with original names
    _, orig_seqs = build_orig_vocab()
    val_seqs = load_orig_val(None)
    print(f"Val set: {len(val_seqs)} sequences (original names)")

    # Evaluate
    results = defaultdict(lambda: {"top1": 0, "top3": 0, "top5": 0, "total": 0})
    total_top1 = total_top3 = total_top5 = total = 0

    for seq_idx, (steps, family) in enumerate(val_seqs[:300]):
        for frac in (0.6, 0.8):
            cut = max(1, int(len(steps) * frac))
            if cut >= len(steps):
                continue

            original_prefix = steps[:cut]           # original names for Markov
            true_next       = steps[cut]             # original name GT

            # Encode with canonical mapping for LSTM
            canonical_prefix = [CANONICAL_STEPS.get(s, s) for s in original_prefix]
            prefix_ids = encode_prefix(vocab_canon, canonical_prefix, family,
                                       apply_canonical=False)  # already canonical

            top5 = predict_top5_original(model, vocab_canon, prefix_ids, device,
                                         original_prefix, markov_orig)

            t1 = int(top5[0] == true_next if top5 else False)
            t3 = int(true_next in top5[:3])
            t5 = int(true_next in top5)

            total_top1 += t1; total_top3 += t3; total_top5 += t5; total += 1
            results[family]["top1"]  += t1
            results[family]["top3"]  += t3
            results[family]["top5"]  += t5
            results[family]["total"] += 1

        if (seq_idx + 1) % 50 == 0:
            print(f"  {seq_idx+1}/300 — running Top-1: {total_top1/total*100:.1f}%")

    print(f"\n{'='*55}")
    print(f"TRUE self-eval (original GT vs original predictions)")
    print(f"{'='*55}")
    print(f"Top-1: {total_top1/total*100:.2f}%")
    print(f"Top-3: {total_top3/total*100:.2f}%")
    print(f"Top-5: {total_top5/total*100:.2f}%")
    print(f"Total examples: {total}")
    print()
    for fam, r in sorted(results.items()):
        n = r["total"]
        print(f"  {fam:8s}: Top-1={r['top1']/n*100:.1f}%  "
              f"Top-3={r['top3']/n*100:.1f}%  "
              f"Top-5={r['top5']/n*100:.1f}%  (n={n})")


if __name__ == "__main__":
    run()
