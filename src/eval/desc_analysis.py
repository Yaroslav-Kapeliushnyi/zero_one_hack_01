"""Are our 'synonyms' the same process? Compare official STEP / DESCRIPTION."""
import csv
from difflib import SequenceMatcher
from pathlib import Path

BASE = Path("/Users/yehor_larcenko/Desktop/hackathon2/tracks/industrial-infineon/training_data")
FILES = ["MOSFET_Longdescr.csv", "IGBT_Longdescr.csv", "IC_Longdescr.csv"]

desc = {}
for fn in FILES:
    fp = BASE / fn
    if not fp.exists():
        continue
    for r in csv.reader(open(fp, encoding="utf-8-sig")):
        if len(r) >= 2 and r[0] != "STEP":
            desc.setdefault(r[0].strip(), r[1].strip())

groups = {
    "STRIP RESIST": ["STRIP RESIST", "STRIP PHOTORESIST", "STRIP RESIST LEVEL 2"],
    "PASSIVATION ETCH": ["PASSIVATION ETCH", "PASSIVATION ETCH PAD OPENING"],
    "MEASURE PLANARITY": ["MEASURE PLANARITY", "MEASURE SURFACE PLANARITY"],
    "OPEN PAD WINDOW": ["OPEN PAD WINDOW", "OPEN BOND PAD WINDOW"],
    "CMP DIELECTRIC": ["CMP DIELECTRIC", "CMP INTERLAYER DIELECTRIC"],
    "PAD WINDOW LITHO": ["PAD WINDOW LITHO", "OPEN PAD WINDOW LITHO"],
    "VIA ETCH": ["VIA ETCH", "VIA ETCH THROUGH DIELECTRIC", "DIELECTRIC ETCH VIA"],
    "MEASURE DIELECTRIC THICKNESS": ["MEASURE DIELECTRIC THICKNESS", "MEASURE FILM THICKNESS"],
}

def sim(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

out = [f"total steps in library: {len(desc)}", ""]
for canon, members in groups.items():
    out.append("=" * 78)
    out.append(f"GROUP: {canon}")
    present = [m for m in members if m in desc]
    for m in members:
        d = desc.get(m, "<<NOT IN LIBRARY>>")
        out.append(f"  [{m}]")
        out.append(f"     {d}")
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            a, b = present[i], present[j]
            r = sim(desc[a], desc[b])
            v = "SAME" if r > 0.85 else ("RELATED" if r > 0.5 else "DIFFERENT!")
            out.append(f"  ~ {a} vs {b}: sim={r:.2f} -> {v}")
    out.append("")

Path("/tmp/desc_report.txt").write_text("\n".join(out))
print("\n".join(out))
