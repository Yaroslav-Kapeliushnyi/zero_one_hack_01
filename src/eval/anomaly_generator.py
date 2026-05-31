"""
Anomaly generator — injects verified rule violations into valid sequences.
Uses generate_sequences.py validate_sequence() to confirm each violation.
Covers all 10 rule types from generation_rules.md.
"""

import random
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).parent.parent.parent

# ── Step category sets (mirrors generate_sequences.py) ───────────────────────
DEPOSITION_STEPS = {
    "THERMAL OXIDATION", "GATE OXIDE GROWTH", "DEPOSIT PAD OXIDE",
    "EPITAXIAL DEPOSITION", "DEPOSIT POLYSILICON", "DEPOSIT SPACER DIELECTRIC",
    "DEPOSIT FIELD OXIDE", "DEPOSIT GATE OXIDE OR DIELECTRIC",
    "DEPOSIT INTERLAYER DIELECTRIC", "DEPOSIT INTERLEVEL DIELECTRIC",
    "DEPOSIT BARRIER METAL", "DEPOSIT METAL SEED", "DEPOSIT METAL 1",
    "DEPOSIT TOP METAL", "DEPOSIT BACKSIDE METAL", "DEPOSIT TUNGSTEN SEED",
    "DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER", "DEPOSIT BACKSIDE PROTECTION",
}

CLEAN_STEPS = {
    "PRE CLEAN WAFER", "WAFER CLEAN PRE PROCESS", "WAFER SURFACE CLEAN",
    "RCA CLEAN 1", "RCA CLEAN 2", "WET CLEAN RCA1", "WET CLEAN RCA2",
    "HF DIP", "OXIDE STRIP", "SURFACE PREP FOR DEPOSITION",
    "FRONTSIDE CLEAN", "BACKSIDE CLEAN", "FRONTSIDE CLEAN FINAL",
    "BACKSIDE CLEAN FINAL", "WAFER CLEAN PRE-GRIND", "DRY WAFER",
    "DRY WAFER BACKSIDE", "CLEAN AFTER ETCH", "CLEAN AFTER OXIDE ETCH",
    "CLEAN AFTER POLY ETCH", "CLEAN AFTER VIA ETCH", "CLEAN AFTER METAL ETCH",
    "CLEAN AFTER WINDOW ETCH", "CLEAN AFTER FIELD ETCH", "CLEAN PAD OPENING",
    "BACKSIDE ETCH CLEAN", "BACKSIDE RINSE",
}

ETCH_STEPS = {
    "OXIDE ETCH", "OXIDE ETCH DRY", "POLYSILICON ETCH", "POLYSILICON ETCH DRY",
    "ETCH SILICON OR OXIDE WINDOW", "FIELD OXIDE ETCH",
    "VIA ETCH", "VIA ETCH THROUGH DIELECTRIC", "DIELECTRIC ETCH VIA",
    "METAL ETCH", "METAL ETCH DRY",
    "PASSIVATION ETCH PAD OPENING", "PASSIVATION ETCH",
}

METAL_ETCH_STEPS = {"METAL ETCH", "METAL ETCH DRY"}

IMPLANT_STEPS = {
    "IMPLANT WELL", "IMPLANT SOURCE DRAIN", "IMPLANT SOURCE REGION",
    "IMPLANT LDD", "IMPLANT P BODY", "IMPLANT N BUFFER",
    "IMPLANT CHANNEL STOP", "IMPLANT DRAIN / CATHODE REGION", "IMPLANT N-TYPE",
}

CMP_STEPS = {
    "CMP DIELECTRIC", "CMP INTERLAYER DIELECTRIC", "CMP METAL", "CMP VIA FILL",
}

ELECTRICAL_TEST_STEPS = {
    "PARAMETRIC TEST", "ELECTRICAL PARAMETRIC TEST",
    "THRESHOLD VOLTAGE TEST", "BREAKDOWN VOLTAGE TEST",
    "LEAKAGE TEST", "SWITCHING TEST",
}

PAD_OPEN_STEPS = {"OPEN PAD WINDOW", "OPEN BOND PAD WINDOW", "PAD WINDOW LITHO"}


def _find(steps, targets):
    """Return index of first step in targets, or -1."""
    for i, s in enumerate(steps):
        if s in targets:
            return i
    return -1


def _find_all(steps, targets):
    return [i for i, s in enumerate(steps) if s in targets]


def _find_clean_before(steps, dep_idx, window=12):
    """Find a clean step within window positions before dep_idx."""
    start = max(0, dep_idx - window)
    for i in range(dep_idx - 1, start - 1, -1):
        if steps[i] in CLEAN_STEPS:
            return i
    return -1


# ── Rule injection functions ──────────────────────────────────────────────────

def inject_rule_dep_no_clean(steps: List[str]) -> Optional[List[str]]:
    """Remove the clean step before a deposition step."""
    dep_indices = [i for i, s in enumerate(steps) if s in DEPOSITION_STEPS]
    random.shuffle(dep_indices)
    for dep_idx in dep_indices:
        clean_idx = _find_clean_before(steps, dep_idx, window=12)
        if clean_idx >= 0:
            bad = steps[:clean_idx] + steps[clean_idx+1:]
            return bad
    return None


def inject_rule_etch_no_mask(steps: List[str]) -> Optional[List[str]]:
    """Remove DEVELOP PHOTORESIST before an etch step."""
    etch_indices = [i for i, s in enumerate(steps) if s in ETCH_STEPS]
    random.shuffle(etch_indices)
    for etch_idx in etch_indices:
        start = max(0, etch_idx - 12)
        develop_idx = -1
        for i in range(etch_idx - 1, start - 1, -1):
            if steps[i] in ("DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"):
                develop_idx = i
                break
        if develop_idx >= 0:
            bad = steps[:develop_idx] + steps[develop_idx+1:]
            return bad
    return None


def inject_rule_metal_etch_no_litho(steps: List[str]) -> Optional[List[str]]:
    """Remove EXPOSE LITHO before a metal etch step."""
    metal_indices = [i for i, s in enumerate(steps) if s in METAL_ETCH_STEPS]
    random.shuffle(metal_indices)
    for m_idx in metal_indices:
        start = max(0, m_idx - 15)
        expose_idx = -1
        for i in range(m_idx - 1, start - 1, -1):
            if steps[i].startswith("EXPOSE LITHO"):
                expose_idx = i
                break
        if expose_idx >= 0:
            bad = steps[:expose_idx] + steps[expose_idx+1:]
            return bad
    return None


def inject_rule_litho_level_skip(steps: List[str]) -> Optional[List[str]]:
    """Swap ALIGN MASK LEVEL 1 and ALIGN MASK LEVEL 2 to create a skip."""
    idx1 = _find(steps, {"ALIGN MASK LEVEL 1"})
    idx2 = _find(steps, {"ALIGN MASK LEVEL 2"})
    if idx1 >= 0 and idx2 >= 0 and idx1 < idx2:
        bad = list(steps)
        bad[idx1], bad[idx2] = bad[idx2], bad[idx1]
        return bad
    return None


def inject_rule_implant_no_mask(steps: List[str]) -> Optional[List[str]]:
    """Remove oxide etch / develop before an implant step."""
    impl_indices = [i for i, s in enumerate(steps) if s in IMPLANT_STEPS]
    random.shuffle(impl_indices)
    mask_steps = {"OXIDE ETCH", "OXIDE ETCH DRY", "ETCH SILICON OR OXIDE WINDOW",
                  "DEVELOP PHOTORESIST"}
    for impl_idx in impl_indices:
        start = max(0, impl_idx - 15)
        mask_idx = -1
        for i in range(impl_idx - 1, start - 1, -1):
            if steps[i] in mask_steps:
                mask_idx = i
                break
        if mask_idx >= 0:
            bad = steps[:mask_idx] + steps[mask_idx+1:]
            return bad
    return None


def inject_rule_cmp_no_dep(steps: List[str]) -> Optional[List[str]]:
    """Remove the deposition step immediately before a CMP step."""
    cmp_indices = [i for i, s in enumerate(steps) if s in CMP_STEPS]
    fill_steps = DEPOSITION_STEPS | {"FILL VIA METAL", "FILL VIA TUNGSTEN"}
    random.shuffle(cmp_indices)
    for cmp_idx in cmp_indices:
        start = max(0, cmp_idx - 6)
        dep_idx = -1
        for i in range(cmp_idx - 1, start - 1, -1):
            if steps[i] in fill_steps:
                dep_idx = i
                break
        if dep_idx >= 0:
            bad = steps[:dep_idx] + steps[dep_idx+1:]
            return bad
    return None


def inject_rule_pad_open_before_dep(steps: List[str]) -> Optional[List[str]]:
    """Move OPEN PAD WINDOW before DEPOSIT PASSIVATION."""
    pad_idx = _find(steps, PAD_OPEN_STEPS)
    dep_pass_idx = _find(steps, {"DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER"})
    if pad_idx >= 0 and dep_pass_idx >= 0 and pad_idx > dep_pass_idx:
        # Already correct order; move pad before deposition (violation)
        pad_step = steps[pad_idx]
        bad = steps[:pad_idx] + steps[pad_idx+1:]
        insert_at = max(0, dep_pass_idx - 1)
        bad = bad[:insert_at] + [pad_step] + bad[insert_at:]
        return bad
    return None


def inject_rule_test_before_passivation(steps: List[str]) -> Optional[List[str]]:
    """Move an electrical test step before CURE PASSIVATION."""
    cure_idx = _find(steps, {"CURE PASSIVATION"})
    test_indices = _find_all(steps, ELECTRICAL_TEST_STEPS)
    if cure_idx < 0 or not test_indices:
        return None
    # Find a test step that appears AFTER cure (valid position), move it before
    after = [i for i in test_indices if i > cure_idx]
    if not after:
        return None
    test_idx = after[0]
    test_step = steps[test_idx]
    bad = steps[:test_idx] + steps[test_idx+1:]
    insert_at = max(0, cure_idx - 2)
    bad = bad[:insert_at] + [test_step] + bad[insert_at:]
    return bad


def inject_rule_ship_before_test(steps: List[str]) -> Optional[List[str]]:
    """Move SHIP LOT before WAFER SORT TEST."""
    ship_idx = _find(steps, {"SHIP LOT"})
    test_idx  = _find(steps, {"WAFER SORT TEST"})
    if ship_idx < 0 or test_idx < 0 or ship_idx <= test_idx:
        return None
    # ship already after test (valid). Move ship before test.
    bad = steps[:ship_idx] + steps[ship_idx+1:]
    test_idx_new = bad.index("WAFER SORT TEST")
    bad = bad[:test_idx_new] + ["SHIP LOT"] + bad[test_idx_new:]
    return bad


def inject_rule_backside_before_passivation(steps: List[str]) -> Optional[List[str]]:
    """Move DEPOSIT BACKSIDE METAL before CURE PASSIVATION."""
    back_idx = _find(steps, {"DEPOSIT BACKSIDE METAL"})
    cure_idx  = _find(steps, {"CURE PASSIVATION"})
    if back_idx < 0 or cure_idx < 0 or back_idx <= cure_idx:
        return None
    # Already valid (backside after cure). Move backside before cure.
    bad = steps[:back_idx] + steps[back_idx+1:]
    cure_idx_new = bad.index("CURE PASSIVATION")
    bad = bad[:cure_idx_new] + ["DEPOSIT BACKSIDE METAL"] + bad[cure_idx_new:]
    return bad


INJECTORS = {
    "RULE_DEP_NO_CLEAN":               inject_rule_dep_no_clean,
    "RULE_ETCH_NO_MASK":               inject_rule_etch_no_mask,
    "RULE_METAL_ETCH_NO_LITHO":        inject_rule_metal_etch_no_litho,
    "RULE_LITHO_LEVEL_SKIP":           inject_rule_litho_level_skip,
    "RULE_IMPLANT_NO_MASK":            inject_rule_implant_no_mask,
    "RULE_CMP_NO_DEP":                 inject_rule_cmp_no_dep,
    "RULE_PAD_OPEN_BEFORE_DEP":        inject_rule_pad_open_before_dep,
    "RULE_TEST_BEFORE_PASSIVATION":    inject_rule_test_before_passivation,
    "RULE_SHIP_BEFORE_TEST":           inject_rule_ship_before_test,
    "RULE_BACKSIDE_BEFORE_PASSIVATION": inject_rule_backside_before_passivation,
}


def generate_anomaly_set(valid_sequences: List[Tuple[str, str, List[str]]],
                         validator_fn,
                         n_per_rule: int = 40,
                         seed: int = 42) -> List[dict]:
    """
    Generate a balanced anomaly test set.

    Args:
        valid_sequences: list of (example_id, family, steps)
        validator_fn:    validate_sequence() from generate_sequences.py
        n_per_rule:      target number of anomalies per rule type
        seed:            random seed

    Returns:
        list of dicts with keys: EXAMPLE_ID, FAMILY, SEQUENCE, IS_ANOMALY,
                                  VIOLATED_RULE
    """
    rng = random.Random(seed)
    rows = []
    seqs = list(valid_sequences)

    for rule_name, inject_fn in INJECTORS.items():
        rng.shuffle(seqs)
        count = 0
        for eid, family, steps in seqs:
            if count >= n_per_rule:
                break
            bad = inject_fn(list(steps))
            if bad is None:
                continue
            # Verify violation is actually detected
            violations = validator_fn(bad)
            rule_names = [v.rule for v in violations]
            if rule_name in rule_names:
                rows.append({
                    "EXAMPLE_ID":   f"anom_{rule_name}_{count:03d}",
                    "FAMILY":       family,
                    "SEQUENCE":     "|".join(bad),
                    "IS_ANOMALY":   1,
                    "VIOLATED_RULE": rule_name,
                })
                count += 1

        print(f"  {rule_name:40s} → {count} anomalies generated")

    return rows
