"""
Detailed Task-1 hybrid architecture diagram — every stage from input through
RANK_1 (3-model ensemble + decanonicalization) and RANK_2-5 (variant enumeration).
Renders plots/00_architecture_detail.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "plots" / "00_architecture_detail.png"

C = dict(bg="#0F172A", card="#1E293B", card2="#172554", text="#E2E8F0",
         sub="#94A3B8", blue="#3B82F6", green="#22C55E", orange="#F97316",
         purple="#A855F7", yellow="#EAB308", red="#EF4444", line="#475569")

fig, ax = plt.subplots(figsize=(16, 15))
fig.patch.set_facecolor(C["bg"]); ax.set_facecolor(C["bg"])
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, fc, ec, lw=2, r=0.025):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.3,rounding_size={r*100}",
                                fc=fc, ec=ec, lw=lw, zorder=2))

def t(x, y, s, size=12, color=None, bold=False, ha="left", va="top", mono=True, style="normal"):
    ax.text(x, y, s, fontsize=size, color=color or C["text"], ha=ha, va=va,
            family="monospace" if mono else "sans-serif",
            fontweight="bold" if bold else "normal", style=style, zorder=4)

def arrow(x1, y1, x2, y2, color=None, lw=2.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=22,
                                 color=color or C["line"], lw=lw, zorder=1,
                                 shrinkA=0, shrinkB=0))

def label(x, y, s, color, size=11):
    t(x, y, s, size=size, color=color, bold=True, ha="center", va="center")


# ── INPUT ─────────────────────────────────────────────────────────────────
box(14, 93, 72, 6, C["card"], C["blue"], 2.5)
t(50, 97.6, "INPUT", 13, C["blue"], bold=True, ha="center")
t(50, 95.4, "partial process sequence (original step names)  +  family tag [MOSFET | IGBT | IC]",
  11, C["text"], ha="center")

# ── ENCODING (two parallel encodings) ───────────────────────────────────────
arrow(50, 93, 50, 89.5, C["blue"])
box(10, 82, 80, 7.2, C["card2"], C["line"], 1.8)
t(50, 88.4, "encode_prefix()  →  prepend [BOS] [FAMILY] token", 11.5, C["text"], bold=True, ha="center")
t(28, 86.0, "CANONICAL stream  (for the 3 models)", 10.5, C["green"], bold=True, ha="center")
t(28, 84.2, "synonyms merged → canonical token IDs", 9.5, C["sub"], ha="center")
t(28, 82.8, "vocab = 196 tokens", 9.5, C["sub"], ha="center")
ax.plot([50, 50], [82.4, 88.6], color=C["line"], lw=1, zorder=1)
t(73, 86.0, "ORIGINAL stream  (kept for decanon)", 10.5, C["orange"], bold=True, ha="center")
t(73, 84.2, "raw original step names, no merging", 9.5, C["sub"], ha="center")
t(73, 82.8, "vocab = 206 / Markov-orig 198", 9.5, C["sub"], ha="center")

# ── RANK_1 : 3-MODEL ENSEMBLE ───────────────────────────────────────────────
arrow(50, 82, 50, 78.5, C["green"])
box(5, 41.5, 90, 37, C["card"], C["green"], 2.5)
t(8, 76.8, "RANK_1   =   3-MODEL ENSEMBLE   (operates in canonical vocab, 196)", 13, C["green"], bold=True)
t(8, 74.6, "picks the single best next step", 10.5, C["sub"])

# three model rows
models = [
    ("LSTM-Attn", "canonical · 18 MB · val-loss 0.256", "log_softmax  →  la_lp[196]", C["blue"]),
    ("GPT",       "canonical · 6.4 M params · val 0.258", "log_softmax  →  gpt_lp[196]", C["purple"]),
    ("Markov-3",  "canonical · order-3 · vocab 188",      "rank → markov_lp[id] = -rank·0.5", C["yellow"]),
]
my = 70.5
for i, (name, meta, op, col) in enumerate(models):
    y = my - i * 5.6
    box(8, y - 4.4, 33, 4.6, C["card2"], col, 1.6)
    t(10, y - 0.6, name, 12, col, bold=True)
    t(10, y - 2.3, meta, 9, C["sub"])
    t(10, y - 3.7, op, 9.3, C["text"])

# soft-vote fusion
arrow(41, 65.5, 47, 60, models[0][3])
arrow(41, 60.0, 47, 59, models[1][3])
arrow(41, 54.5, 47, 58, models[2][3])
box(47.5, 52, 26, 16, C["card2"], C["green"], 2)
t(60.5, 66.2, "WEIGHTED SOFT-VOTE", 12, C["green"], bold=True, ha="center")
t(60.5, 63.6, "combined =", 10.5, C["text"], ha="center")
t(60.5, 61.9, "w_la·la_lp + w_gpt·gpt_lp", 9.8, C["text"], ha="center")
t(60.5, 60.4, "+ w_mk·markov_lp", 9.8, C["text"], ha="center")
t(60.5, 57.8, "per-family weights", 9.5, C["sub"], ha="center")
t(60.5, 56.3, "≈ 0.40 / 0.40 / 0.20", 9.8, C["orange"], bold=True, ha="center")
t(60.5, 54.0, "(tuned on val NLL)", 9, C["sub"], ha="center")

# argmax -> top canonical
arrow(73.5, 60, 79, 60, C["green"])
box(79, 54.5, 14, 11, C["card2"], C["line"], 1.6)
t(86, 63.6, "argmax", 11, C["text"], bold=True, ha="center")
t(86, 61.4, "→ top CANONICAL", 9.3, C["text"], ha="center")
t(86, 60.0, "step  (the TYPE)", 9.3, C["text"], ha="center")
t(86, 57.6, 'e.g. "STRIP', 9, C["sub"], ha="center")
t(86, 56.3, 'RESIST"', 9, C["sub"], ha="center")

# decanonicalize row
arrow(86, 54.5, 86, 51.5, C["orange"])
box(8, 43, 85, 8, C["card2"], C["orange"], 1.8)
t(10, 49.6, "DECANONICALIZE   →  pick the real original surface VARIANT", 11.5, C["orange"], bold=True)
t(10, 47.4, "markov_orig (original-name trigram, vocab 198)  +  original prefix  →  most likely variant of this canonical step",
  9.5, C["text"])
t(10, 45.4, 'canonical "STRIP RESIST"  →  RANK_1 = "STRIP PHOTORESIST"   (the actual variant in this context)',
  9.5, C["green"])

# ── RANK_1 output arrow ─────────────────────────────────────────────────────
arrow(50, 41.5, 50, 37.5, C["green"])
t(52, 39.6, "RANK_1  (original variant name)", 10.5, C["green"], bold=True, va="center")

# ── RANK_2-5 : VARIANT ENUMERATION ──────────────────────────────────────────
box(5, 16.5, 90, 20.5, C["card"], C["blue"], 2.5)
t(8, 35.2, "RANK_2-5   =   SYNONYM-VARIANT ENUMERATION   (static lookup · NO model)", 13, C["blue"], bold=True)
t(8, 32.0, "for each step the ensemble ranked:   canon = CANONICAL_STEPS[step]", 10, C["text"])
t(8, 30.0, "                                     emit VARIANTS_OF[canon] = [canonical, variant_1, variant_2, …]", 10, C["text"])
t(8, 28.0, "dedup  ·  drop RANK_1  ·  fill to 5 slots", 10, C["text"])
# synonym group example box
box(8, 18.5, 85, 7.6, C["card2"], C["line"], 1.5)
t(10, 24.6, "8 synonym groups, e.g.:", 9.8, C["sub"], bold=True)
t(10, 22.8, 'STRIP RESIST  →  {STRIP RESIST, STRIP PHOTORESIST, STRIP RESIST LEVEL 2}', 9.5, C["yellow"])
t(10, 21.2, 'VIA ETCH  →  {VIA ETCH, VIA ETCH THROUGH DIELECTRIC, DIELECTRIC ETCH VIA}', 9.5, C["yellow"])
t(10, 19.4, "steps with no synonyms contribute only themselves   →   guarantees the true variant is in Top-5  (100% Top-5)",
  9.5, C["green"])

# ── OUTPUT ──────────────────────────────────────────────────────────────────
arrow(50, 16.5, 50, 12.8, C["blue"])
box(14, 5, 72, 7.5, C["card"], C["green"], 2.5)
t(50, 11.0, "OUTPUT   →   submission row", 12, C["green"], bold=True, ha="center")
t(50, 8.4, "EXAMPLE_ID , RANK_1 , RANK_2 , RANK_3 , RANK_4 , RANK_5", 11, C["text"], ha="center")
t(50, 6.4, "71.8% Top-1   ·   100% Top-5   ·   0.857 MRR   (held-out, original-name GT)", 9.8, C["sub"], ha="center")

ax.set_title("Task 1 — Hybrid Next-Step Architecture (every stage)",
             fontsize=17, color=C["text"], fontweight="bold", pad=18, family="monospace")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=C["bg"])
print(f"→ wrote {OUT}")
