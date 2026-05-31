"""
Constraint decoder — rule-based next-step filter.
Given a sequence prefix, returns the set of step names that are NOT
blocked by any of the 10 process grammar rules.

Used to mask invalid logits during inference (neuro-symbolic hybrid):
    all 198 steps → [rule filter] → valid candidates → [LSTM logits] → top-5

Each rule check is O(window_size) — runs in microseconds per prefix.
"""

from typing import List, Set

# ── Step category sets (from generation_rules.md) ────────────────────────────
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
    "BACKSIDE ETCH CLEAN", "BACKSIDE RINSE", "RAPID THERMAL ANNEAL",
    "EPITAXY ANNEAL", "ANNEAL OXIDE", "GATE OXIDE PREP",
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

CMP_STEPS = {"CMP DIELECTRIC", "CMP INTERLAYER DIELECTRIC", "CMP METAL", "CMP VIA FILL"}

FILL_STEPS = {"FILL VIA METAL", "FILL VIA TUNGSTEN"}

ELECTRICAL_TEST_STEPS = {
    "PARAMETRIC TEST", "ELECTRICAL PARAMETRIC TEST",
    "THRESHOLD VOLTAGE TEST", "BREAKDOWN VOLTAGE TEST",
    "LEAKAGE TEST", "SWITCHING TEST",
}

PAD_OPEN_STEPS = {"OPEN PAD WINDOW", "OPEN BOND PAD WINDOW", "PAD WINDOW LITHO"}

DEVELOP_STEPS = {"DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"}

MASK_FOR_IMPLANT = {"OXIDE ETCH", "OXIDE ETCH DRY", "ETCH SILICON OR OXIDE WINDOW",
                    "DEVELOP PHOTORESIST"}

PASSIVATION_DEPOSIT = {"DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER"}


def get_valid_next_steps(prefix_steps: List[str], vocab_steps: Set[str]) -> Set[str]:
    """
    Returns the set of step names from vocab_steps that are NOT blocked
    by any of the 10 process grammar rules given the current prefix.

    Window-based rules (DEP_NO_CLEAN, ETCH_NO_MASK, etc.) are only enforced
    once the prefix is long enough for the window to be meaningful. At sequence
    start, blocking all deposition/etch steps causes false negatives because
    the very first steps are typically deposition (THERMAL OXIDATION).

    Runs in <1ms per call.
    """
    blocked: Set[str] = set()

    n = len(prefix_steps)
    recent_6  = prefix_steps[-6:]
    recent_12 = prefix_steps[-12:]
    recent_15 = prefix_steps[-15:]

    # RULE_DEP_NO_CLEAN — only enforce once we've had enough context (≥12 steps)
    # At sequence start, THERMAL OXIDATION is often the very first step
    if n >= 12:
        has_clean_12 = any(s in CLEAN_STEPS for s in recent_12)
        if not has_clean_12:
            blocked.update(DEPOSITION_STEPS)

    # RULE_ETCH_NO_MASK — only enforce once litho cycle could have happened (≥12 steps)
    if n >= 12:
        has_develop_12 = any(s in DEVELOP_STEPS for s in recent_12)
        if not has_develop_12:
            blocked.update(ETCH_STEPS)

    # RULE_METAL_ETCH_NO_LITHO — needs a full litho cycle (≥15 steps)
    if n >= 15:
        has_expose_15 = any(s.startswith("EXPOSE LITHO") for s in recent_15)
        has_develop_15 = any(s in DEVELOP_STEPS for s in recent_15)
        if not (has_expose_15 and has_develop_15):
            blocked.update(METAL_ETCH_STEPS)

    # RULE_CMP_NO_DEP — CMP needs recent deposition/fill (≥6 steps)
    if n >= 6:
        has_dep_6 = any(s in DEPOSITION_STEPS or s in FILL_STEPS for s in recent_6)
        if not has_dep_6:
            blocked.update(CMP_STEPS)

    # RULE_IMPLANT_NO_MASK — implant needs oxide etch/develop (≥15 steps)
    if n >= 15:
        has_mask_15 = any(s in MASK_FOR_IMPLANT for s in recent_15)
        if not has_mask_15:
            blocked.update(IMPLANT_STEPS)

    # Global ordering rules — apply always regardless of prefix length

    # RULE_TEST_BEFORE_PASSIVATION — tests require CURE PASSIVATION already seen
    cure_seen = "CURE PASSIVATION" in prefix_steps
    if not cure_seen:
        blocked.update(ELECTRICAL_TEST_STEPS)

    # RULE_SHIP_BEFORE_TEST — SHIP LOT requires WAFER SORT TEST already seen
    sort_seen = "WAFER SORT TEST" in prefix_steps
    if not sort_seen:
        blocked.add("SHIP LOT")

    # RULE_BACKSIDE_BEFORE_PASSIVATION — backside metal requires CURE PASSIVATION
    if not cure_seen:
        blocked.add("DEPOSIT BACKSIDE METAL")

    # RULE_PAD_OPEN_BEFORE_DEP — pad opening requires DEPOSIT PASSIVATION + CURE
    dep_pass_seen = any(s in PASSIVATION_DEPOSIT for s in prefix_steps)
    if not dep_pass_seen or not cure_seen:
        blocked.update(PAD_OPEN_STEPS)

    # RULE_LITHO_LEVEL_SKIP — litho levels must be sequential
    prefix_set = set(prefix_steps)
    highest_completed = 0
    for lvl in range(1, 7):
        if (f"ALIGN MASK LEVEL {lvl}" in prefix_set and
                f"DEVELOP PHOTORESIST" in prefix_set):
            highest_completed = max(highest_completed, lvl)
    # Block skipping: level N+2 and above are forbidden if level N isn't done
    for lvl in range(highest_completed + 2, 8):
        blocked.add(f"ALIGN MASK LEVEL {lvl}")
        blocked.add(f"EXPOSE LITHO LEVEL {lvl}")

    return vocab_steps - blocked


def build_constraint_mask(prefix_steps: List[str], vocab,
                          device=None) -> "torch.Tensor":
    """
    Returns a float tensor of shape (vocab_size,) where:
      0.0  = step is valid (not blocked)
      -inf = step is blocked by a rule

    Add this mask to logits before taking top-k:
        constrained_logits = logits + mask
        top5 = constrained_logits.topk(5)
    """
    import torch
    import math

    all_steps = set(vocab.step2id.keys())
    valid = get_valid_next_steps(prefix_steps, all_steps)

    mask = torch.full((len(vocab),), float('-inf'),
                      device=device, dtype=torch.float32)
    for step in valid:
        if step in vocab.step2id:
            mask[vocab.step2id[step]] = 0.0

    return mask
