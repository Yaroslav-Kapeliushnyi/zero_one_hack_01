# GitHub Repo Findings — zero_one_hack_01
> https://github.com/Lumos-Data/zero_one_hack_01
> Partner: Infineon

---

## Available Data Files

| File | Description | Size |
|------|-------------|------|
| `syntheticMOSFET.csv` | 1 reference MOSFET sequence (126 steps) | 2.6 KB |
| `syntheticIGBT.csv` | 1 reference IGBT sequence (151 steps) | 3.3 KB |
| `syntheticIC.csv` | 1 reference IC sequence (107 steps) | 2.3 KB |
| `MOSFET_variants.csv` | 1,000 pre-generated MOSFET sequences | 3.5 MB |
| `IGBT_variants.csv` | 1,000 pre-generated IGBT sequences | 4.2 MB |
| `IC_variants.csv` | 1,000 pre-generated IC sequences | 3.2 MB |
| `*_Longdescr.csv` | Human-readable step descriptions | ~10 KB each |
| `*_longdescription_parameters.csv` | Extended metadata per step | ~8 KB each |

**CSV Format:** `SEQUENCE_ID, STEP` (one row per step, long format)

---

## Process Grammar Backbone (All Families)

```
PREFIX
→ INITIAL_MEASUREMENTS
→ PRE_PROCESS_CLEAN
→ FAMILY_SPECIFIC_PREP
→ FIRST_OXIDATION
→ PROCESS_CYCLES {3..6 cycles}
→ ILD_BLOCK
→ VIA_BLOCK
→ METAL_BLOCK
→ PASSIVATION_BLOCK
→ BACKSIDE_BLOCK
→ FINAL_INSPECTION
→ TEST_SUITE
→ SUFFIX
```

**Combinatoric space:**
- MOSFET: ~51 billion valid sequences
- IGBT: ~13 trillion valid sequences
- IC: ~6 billion valid sequences

---

## The 10 Forbidden Rules (= Anomaly Ground Truth)

These are the rules the anomaly eval set tests. Models must learn all 10.

| Rule ID | Description |
|---------|-------------|
| RULE_DEP_NO_CLEAN | Deposition requires a prior clean within 12 steps |
| RULE_METAL_ETCH_NO_LITHO | Metal etch requires EXPOSE LITHO + DEVELOP within 15 steps |
| RULE_ETCH_NO_MASK | Etch requires DEVELOP PHOTORESIST within 12 steps |
| RULE_LITHO_LEVEL_SKIP | Litho levels must be sequential (1,2,3,...) — no skipping |
| RULE_IMPLANT_NO_MASK | Implant requires oxide etch or litho develop within 15 steps |
| RULE_CMP_NO_DEP | CMP requires deposition/fill within 6 steps |
| RULE_PAD_OPEN_BEFORE_DEP | Pad window must open after DEPOSIT PASSIVATION + CURE |
| RULE_TEST_BEFORE_PASSIVATION | Electrical tests must appear after CURE PASSIVATION |
| RULE_SHIP_BEFORE_TEST | SHIP LOT must appear after WAFER SORT TEST |
| RULE_BACKSIDE_BEFORE_PASSIVATION | DEPOSIT BACKSIDE METAL must appear after CURE PASSIVATION |

---

## generate_sequences.py — Key Functions

```python
validate_sequence(steps)         # checks all 10 rules, returns Violation objects
generate_sequence(family, rng)   # produces one valid sequence
generate_dataset(family, count, seed)  # produces N unique sequences

# CLI usage:
python generate_sequences.py --family mosfet --count 500 --seed 42
python generate_sequences.py --validate mysequences.csv --family mosfet
python generate_sequences.py --family ic --estimate-only
```

**Vocabulary step categories exposed:**
`DEPOSITION_STEPS, CLEAN_STEPS, ETCH_STEPS, IMPLANT_STEPS, CMP_STEPS, PAD_WINDOW_STEPS, ELECTRICAL_TEST_STEPS, BACKSIDE_METAL_STEPS`

---

## Submission Requirements (CRITICAL)

**Deadline: Sunday 10:00 AM — no extensions**

**Via Tally form:**
- Team name
- Public GitHub repo (MIT licensed)
- PDF slides (max 10 slides / 3-min pitch)
- Demo video (max 2 minutes)

**Mandatory in repo:**
- `MIT LICENSE` at root
- `README.md` with setup + run instructions
- `REPORT.md`:
  - Problem statement
  - Technical approach (3–5 bullets)
  - Reproducible exact commands
  - Quantified results vs. baseline
  - Honest retrospective (what worked/failed)
  - Next steps
  - All libraries, models, APIs used
- `requirements.txt`
- Zero hardcoded secrets

**Track-specific output files to submit:**

| File | Content |
|------|---------|
| `nextstep.csv` | Top-5 predictions per eval example |
| `completion.csv` | Full predicted sequences |
| `anomaly.csv` | Anomaly scores + rule attribution |

---

## Immediate Action Items

1. Clone the repo and download all CSV files
2. Read `generation_rules.md` in full (28.7 KB — the ground truth for our model)
3. Run `generate_sequences.py` to generate more training data (target: 5K–10K per family)
4. Set up the eval pipeline early using `eval_metrics.py`
5. Create GitHub repo with MIT license for final submission
