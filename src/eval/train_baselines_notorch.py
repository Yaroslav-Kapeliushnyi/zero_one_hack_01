"""
Baseline evaluation — NO torch dependency.
Loads sequences directly from CSV files.
"""
import csv, json, sys, random
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "data"

# ── Load sequences from CSV files ─────────────────────────────────────────────
def load_sequences_from_csvs(data_dir):
    seqs = []
    for fam in ("mosfet", "igbt", "ic"):
        for fname in data_dir.glob(f"*{fam}*.csv"):
            try:
                rows = list(csv.DictReader(open(fname)))
                # long format: SEQUENCE_ID, STEP
                if rows and "STEP" in rows[0]:
                    cur_id, cur_steps = None, []
                    for r in rows:
                        sid = r.get("SEQUENCE_ID", r.get("sequence_id", ""))
                        step = r["STEP"].strip().strip('"')
                        if step and not step.startswith("#"):
                            if sid != cur_id:
                                if cur_steps:
                                    seqs.append(cur_steps)
                                cur_id, cur_steps = sid, []
                            cur_steps.append(step)
                    if cur_steps:
                        seqs.append(cur_steps)
            except Exception:
                pass
    return seqs

seqs = load_sequences_from_csvs(DATA)
if not seqs:
    # Try training_data in repo
    for alt in [ROOT/"training_data", ROOT/"repo"/"tracks"/"industrial-infineon"/"training_data"]:
        seqs = load_sequences_from_csvs(alt)
        if seqs:
            break

print(f"Loaded {len(seqs)} sequences")

if len(seqs) < 10:
    print("Not enough data — loading from pixi data dir")
    # Fall back: read from the hackathon data dir directly
    for p in Path("/leonardo_scratch/large/usertrain/CLUSTER_USER/hackathon/data").glob("*.csv"):
        try:
            rows = list(csv.DictReader(open(p)))
            if rows and "STEP" in rows[0]:
                cur_id, cur_steps = None, []
                for r in rows:
                    sid = r.get("SEQUENCE_ID","")
                    step = r["STEP"].strip().strip('"')
                    if step and sid != cur_id:
                        if cur_steps: seqs.append(cur_steps)
                        cur_id, cur_steps = sid, []
                    elif step:
                        cur_steps.append(step)
                if cur_steps: seqs.append(cur_steps)
        except Exception:
            pass
    print(f"After fallback: {len(seqs)} sequences")

# ── Train/val split ───────────────────────────────────────────────────────────
random.seed(42)
random.shuffle(seqs)
cut = int(len(seqs) * 0.9)
train_seqs, val_seqs = seqs[:cut], seqs[cut:]
print(f"Train: {len(train_seqs)}  Val: {len(val_seqs)}")

# ── Eval pairs ────────────────────────────────────────────────────────────────
eval_pairs = []
for seq in val_seqs[:300]:
    for frac in (0.6, 0.8):
        cut_i = max(1, int(len(seq) * frac))
        if cut_i < len(seq):
            eval_pairs.append((seq[:cut_i], seq[cut_i]))
print(f"Eval pairs: {len(eval_pairs)}")

# ── Markov order-1 ────────────────────────────────────────────────────────────
counts1 = defaultdict(lambda: defaultdict(int))
for seq in train_seqs:
    for i in range(len(seq)-1):
        counts1[seq[i]][seq[i+1]] += 1

def m1(prefix):
    d = counts1.get(prefix[-1], {})
    return max(d, key=d.get) if d else None

acc1 = sum(m1(p) == t for p, t in eval_pairs) / len(eval_pairs)
print(f"Markov order-1  Top-1: {acc1*100:.1f}%")

# ── Markov order-2 ────────────────────────────────────────────────────────────
counts2 = defaultdict(lambda: defaultdict(int))
for seq in train_seqs:
    for i in range(len(seq)-1):
        ctx = tuple(seq[max(0,i-1):i+1])
        counts2[ctx][seq[i+1]] += 1

def m2(prefix):
    ctx = tuple(prefix[-2:]) if len(prefix) >= 2 else (prefix[-1],)
    d = counts2.get(ctx, {})
    return max(d, key=d.get) if d else m1(prefix)

acc2 = sum(m2(p) == t for p, t in eval_pairs) / len(eval_pairs)
print(f"Markov order-2  Top-1: {acc2*100:.1f}%")

# ── GBM ───────────────────────────────────────────────────────────────────────
try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import LabelEncoder
    import numpy as np

    all_steps = sorted({s for seq in train_seqs for s in seq})
    le = LabelEncoder().fit(all_steps)
    W = 5

    def feat(prefix):
        pad = ["<PAD>"] * W + list(prefix)
        return [int(le.transform([s])[0]) if s in le.classes_ else -1
                for s in pad[-W:]]

    X, y = [], []
    for seq in train_seqs:
        for i in range(1, len(seq)):
            if seq[i] in le.classes_:
                X.append(feat(seq[:i]))
                y.append(int(le.transform([seq[i]])[0]))

    print(f"GBM: {len(X)} training samples — fitting...")
    gbm = GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                     learning_rate=0.15, subsample=0.8, random_state=42)
    gbm.fit(X, y)

    correct = sum(
        le.inverse_transform([gbm.predict([feat(p)])[0]])[0] == t
        for p, t in eval_pairs if t in le.classes_
    )
    acc_gbm = correct / len(eval_pairs)
    print(f"GBM (last-5 steps)  Top-1: {acc_gbm*100:.1f}%")
except ImportError:
    acc_gbm = 0.0
    print("sklearn not available — skipping GBM")

results = {
    "markov_order1": round(acc1*100, 1),
    "markov_order2": round(acc2*100, 1),
    "gbm":           round(acc_gbm*100, 1),
}
out = Path("/leonardo_scratch/large/usertrain/CLUSTER_USER/hackathon/eval_results/baseline_results.json")
out.parent.mkdir(exist_ok=True)
json.dump(results, open(out, "w"), indent=2)
print("Results:", results)
