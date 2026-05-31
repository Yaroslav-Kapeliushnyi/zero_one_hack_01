"""
THE DECISIVE TEST: is the synonym variant predictable from the PRECEDING step(s)?

Descriptions revealed:
  STRIP RESIST      -> after poly/metal/pad etch
  STRIP PHOTORESIST -> after via/oxide etch
So the previous step (or last 1-3 steps) may determine the variant.

For each synonym group, we measure: given the (prev1) or (prev2,prev1) context,
how often is the variant DETERMINISTIC? If P(majority variant | context) >> 50%,
then a context lookup table recovers real points. We compute the ACHIEVABLE
accuracy on synonym targets using a context->majority-variant table built on a
train split and evaluated on a held-out split (no leakage).
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path("/Users/yehor_larcenko/Desktop/hackathon2/tracks/industrial-infineon/training_data")
FILES = {"mosfet": "MOSFET_variants.csv", "igbt": "IGBT_variants.csv", "ic": "IC_variants.csv"}

# our synonym groups (the 8 from data.py)
GROUPS = {
    "STRIP RESIST": {"STRIP RESIST", "STRIP PHOTORESIST", "STRIP RESIST LEVEL 2"},
    "PASSIVATION ETCH": {"PASSIVATION ETCH", "PASSIVATION ETCH PAD OPENING"},
    "MEASURE PLANARITY": {"MEASURE PLANARITY", "MEASURE SURFACE PLANARITY"},
    "OPEN PAD WINDOW": {"OPEN PAD WINDOW", "OPEN BOND PAD WINDOW"},
    "CMP DIELECTRIC": {"CMP DIELECTRIC", "CMP INTERLAYER DIELECTRIC"},
    "PAD WINDOW LITHO": {"PAD WINDOW LITHO", "OPEN PAD WINDOW LITHO"},
    "VIA ETCH": {"VIA ETCH", "VIA ETCH THROUGH DIELECTRIC", "DIELECTRIC ETCH VIA"},
    "MEASURE DIELECTRIC THICKNESS": {"MEASURE DIELECTRIC THICKNESS", "MEASURE FILM THICKNESS"},
}
VARIANT2GROUP = {v: g for g, members in GROUPS.items() for v in members}

# load all sequences with family
seqs = []
for fam, fn in FILES.items():
    rows = defaultdict(list)
    for r in csv.DictReader(open(BASE / fn, encoding="utf-8")):
        rows[r["SEQUENCE_ID"]].append(r["STEP"].strip())
    for sid, steps in rows.items():
        seqs.append((fam, steps))

print(f"total sequences: {len(seqs)}")

# split 90/10 deterministically
seqs.sort(key=lambda x: x[1][0] if x[1] else "")
import random
random.Random(42).shuffle(seqs)
n_val = len(seqs) // 10
train, val = seqs[n_val:], seqs[:n_val]
print(f"train={len(train)} val={len(val)}\n")

def contexts(steps, i, fam):
    """Return context keys of increasing specificity for position i (the target)."""
    prev1 = steps[i-1] if i >= 1 else "<BOS>"
    prev2 = steps[i-2] if i >= 2 else "<BOS>"
    return {
        "prev1": (fam, prev1),
        "prev2": (fam, prev2, prev1),
    }

for ctx_name in ("prev1", "prev2"):
    # Build context -> Counter(variant) on train
    table = defaultdict(Counter)
    base_rate = defaultdict(Counter)  # group -> Counter(variant) global
    for fam, steps in train:
        for i, s in enumerate(steps):
            if s in VARIANT2GROUP:
                g = VARIANT2GROUP[s]
                key = contexts(steps, i, fam)[ctx_name]
                table[key][s] += 1
                base_rate[g][s] += 1

    # Evaluate on val: predict majority variant for the context; fall back to global majority
    global_major = {g: c.most_common(1)[0][0] for g, c in base_rate.items()}
    tot = hit_ctx = hit_global = 0
    for fam, steps in val:
        for i, s in enumerate(steps):
            if s in VARIANT2GROUP:
                g = VARIANT2GROUP[s]
                tot += 1
                key = contexts(steps, i, fam)[ctx_name]
                if key in table and table[key]:
                    pred = table[key].most_common(1)[0][0]
                else:
                    pred = global_major[g]
                if pred == s:
                    hit_ctx += 1
                if global_major[g] == s:
                    hit_global += 1
    print(f"[{ctx_name}] synonym targets={tot}")
    print(f"   global-majority baseline : {100*hit_global/tot:.1f}%")
    print(f"   CONTEXT lookup           : {100*hit_ctx/tot:.1f}%  <-- {ctx_name}")
    print()
