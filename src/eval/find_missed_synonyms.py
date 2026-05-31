"""
Find synonym groups we MISSED: step names with identical/near-identical official
descriptions that are NOT already merged in data.py CANONICAL_STEPS.
A missed merge = free accuracy points (we're splitting prob mass for no reason).
"""
import csv, sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

BASE = Path("/Users/yehor_larcenko/Desktop/hackathon2/tracks/industrial-infineon/training_data")
FILES = ["MOSFET_Longdescr.csv", "IGBT_Longdescr.csv", "IC_Longdescr.csv"]

step_desc = {}
for fn in FILES:
    fp = BASE / fn
    if not fp.exists():
        continue
    for r in csv.DictReader(open(fp, encoding="utf-8")):
        s = r["STEP"].strip()
        d = (r.get("STEP_DESCRIPTION") or "").strip()
        if d:
            step_desc[s] = d

sys.path.insert(0, "/Users/yehor_larcenko/Desktop/hackathon2/src")
from data import CANONICAL_STEPS, VARIANTS_OF

def same_group(a, b):
    ca = CANONICAL_STEPS.get(a, a)
    cb = CANONICAL_STEPS.get(b, b)
    return ca == cb

def sim(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

steps = sorted(step_desc)
print(f"steps with descriptions: {len(steps)}")
print("\n=== STEP-NAME PAIRS WITH NEAR-IDENTICAL DESCRIPTIONS, NOT YET MERGED ===")
found = 0
for i in range(len(steps)):
    for j in range(i + 1, len(steps)):
        a, b = steps[i], steps[j]
        r = sim(step_desc[a], step_desc[b])
        if r >= 0.90 and not same_group(a, b):
            found += 1
            print(f"\n  sim={r:.2f}  (NOT merged)")
            print(f"    [{a}]  {step_desc[a][:90]}")
            print(f"    [{b}]  {step_desc[b][:90]}")
print(f"\nTotal missed candidate merges (sim>=0.90): {found}")
if found == 0:
    print("=> Our CANONICAL_STEPS map is COMPLETE. No free points from missed synonyms.")
