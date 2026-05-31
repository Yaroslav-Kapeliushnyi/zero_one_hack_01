"""
Generate professional slide plots.
Output: plots/ directory with PNG files.

Usage:
    python src/eval/make_plots.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

ROOT  = Path(__file__).parent.parent.parent
PLOTS = ROOT / "plots"
LOGS  = ROOT / "logs"
PLOTS.mkdir(exist_ok=True)

# ── Professional color palette ────────────────────────────────────────────────
C = {
    "bg":       "#0F172A",   # dark navy background
    "bg2":      "#1E293B",   # card background
    "grid":     "#334155",   # subtle grid lines
    "text":     "#F1F5F9",   # primary text
    "subtext":  "#94A3B8",   # secondary text
    "blue":     "#3B82F6",   # primary accent
    "green":    "#22C55E",   # success / best
    "orange":   "#F97316",   # warning / medium
    "red":      "#EF4444",   # bad / baseline
    "purple":   "#A855F7",   # extra
    "cyan":     "#06B6D4",   # extra
    "yellow":   "#EAB308",   # highlight
}

def base_style():
    plt.rcParams.update({
        "figure.facecolor":  C["bg"],
        "axes.facecolor":    C["bg2"],
        "axes.edgecolor":    C["grid"],
        "axes.labelcolor":   C["text"],
        "axes.titlecolor":   C["text"],
        "axes.grid":         True,
        "grid.color":        C["grid"],
        "grid.linewidth":    0.6,
        "grid.alpha":        0.5,
        "xtick.color":       C["subtext"],
        "ytick.color":       C["subtext"],
        "text.color":        C["text"],
        "font.family":       "sans-serif",
        "font.size":         12,
        "axes.titlesize":    14,
        "axes.titleweight":  "bold",
        "axes.titlepad":     14,
        "axes.labelsize":    11,
        "legend.facecolor":  C["bg2"],
        "legend.edgecolor":  C["grid"],
        "legend.labelcolor": C["text"],
        "figure.dpi":        150,
    })

base_style()

def save(fig, name):
    path = PLOTS / name
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=C["bg"], edgecolor="none")
    plt.close(fig)
    print(f"  ✓ {name}")


# ── PLOT 1: Loss curves ────────────────────────────────────────────────────────
def plot_loss_curves():
    logs = [
        ("lstm_log.json",          "LSTM — 3K data",       C["blue"],   "-",  "--"),
        ("lstm_30k_pure_log.json", "LSTM — 30K data",      C["green"],  "-",  "--"),
        ("gpt_log.json",           "GPT Transformer",      C["purple"], "-",  "--"),
    ]

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(C["bg"])

    for fname, label, color, ts, vs in logs:
        path = LOGS / fname
        if not path.exists():
            continue
        data = json.load(open(path))
        epochs     = [d["epoch"] for d in data]
        train_loss = [d["train"] for d in data]
        val_loss   = [d["val"]   for d in data]
        ax.plot(epochs, train_loss, color=color, lw=2.0, ls=ts, alpha=0.6)
        ax.plot(epochs, val_loss,   color=color, lw=2.5, ls=vs, label=label)
        best_val = min(val_loss)
        best_ep  = epochs[val_loss.index(best_val)]
        ax.scatter([best_ep], [best_val], color=color, s=80, zorder=5)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title("Training Progress — Validation Loss Curves", fontsize=15)

    # Legend for line style
    solid  = mpatches.Patch(color=C["subtext"], label="── Train loss (faded)")
    dashed = mpatches.Patch(color=C["subtext"], label="── Val loss (solid)")
    handles, labels_ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [solid, dashed], frameon=True,
              loc="upper right", fontsize=10)

    ax.annotate("Best val loss\nLSTM 30K: 0.3256", xy=(30, 0.326),
                xytext=(20, 0.345), fontsize=9, color=C["green"],
                arrowprops=dict(arrowstyle="->", color=C["green"], lw=1.5))

    save(fig, "01_loss_curves.png")


# ── PLOT 2: Task 1 — model comparison ─────────────────────────────────────────
def plot_task1_comparison(m1=47.2, m2=63.1, gbm=59.4):
    # m1/m2/gbm: Markov order-1, order-2, GBM — updated from baseline job results
    labels = ["Naive\nmost-freq", "Markov\norder-1\n(unigram)", "GBM\n(last-5\nsteps)",
              "Markov\norder-2", "Markov\norder-3", "LSTM\nalone", "Hybrid\n(submitted)"]
    top1   = [2.3, m1, gbm, m2, 70.0, 68.8, 71.8]
    colors = [C["red"], C["orange"], C["orange"], C["orange"],
              C["orange"], C["blue"], C["green"]]

    fig, ax = plt.subplots(figsize=(12, 5.5))

    bars = ax.bar(labels, top1, color=colors, width=0.55,
                  edgecolor=C["bg"], linewidth=1.5, zorder=3)

    for bar, val in zip(bars, top1):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.6,
                f"{val}%", ha="center", va="bottom",
                fontweight="bold", fontsize=11, color=C["text"])

    # Bracket: baseline zone
    ax.axhspan(0, 71.0, alpha=0.04, color=C["orange"], zorder=0)
    ax.text(0.01, 0.55, "Baseline zone", transform=ax.transAxes,
            color=C["orange"], fontsize=9, alpha=0.7, rotation=90, va="center")

    # Best model line
    ax.axhline(71.8, color=C["green"], lw=1.5, ls=":", alpha=0.6, zorder=2)
    ax.text(6.42, 72.5, "Best\n71.8%", color=C["green"], fontsize=9, ha="center")

    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_title("Task 1 — Next-Step Prediction: Baseline Progression vs Trained Models",
                 fontsize=14)
    ax.set_ylim(0, 78)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

    legend_items = [
        mpatches.Patch(color=C["red"],    label="Naive baseline (no training)"),
        mpatches.Patch(color=C["orange"], label="Classical baselines"),
        mpatches.Patch(color=C["blue"],   label="Neural (single model)"),
        mpatches.Patch(color=C["green"],  label="Hybrid (our best)"),
    ]
    ax.legend(handles=legend_items, frameon=True, loc="lower right", fontsize=10)

    save(fig, "02_task1_models.png")


# ── PLOT 3: Task 2 — NED comparison ───────────────────────────────────────────
def plot_task2_ned():
    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(1, 2, wspace=0.35)

    # Left: model NED bar chart
    ax1 = fig.add_subplot(gs[0])
    models = ["Random", "Markov\nn-gram", "LSTM\ngreedy", "LSTM +\nBeam-5"]
    ned    = [0.950, 0.317, 0.2245, 0.2223]
    colors = [C["red"], C["orange"], C["blue"], C["green"]]

    bars = ax1.bar(models, ned, color=colors, width=0.5,
                   edgecolor=C["bg"], linewidth=1.5, zorder=3)
    for bar, val in zip(bars, ned):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.008,
                 f"{val}", ha="center", va="bottom",
                 fontweight="bold", fontsize=11, color=C["text"])

    ax1.set_ylabel("Normalized Edit Distance  ↓  (lower = better)", fontsize=11)
    ax1.set_title("Sequence Completion — All Models", fontsize=13)
    ax1.set_ylim(0, 1.08)
    ax1.yaxis.grid(True, zorder=0)
    ax1.set_axisbelow(True)

    # Improvement bracket
    ax1.annotate("", xy=(3, 0.2223), xytext=(1, 0.317),
                 arrowprops=dict(arrowstyle="<->", color=C["green"], lw=2))
    ax1.text(2.0, 0.28, "−0.095\nimprovement",
             color=C["green"], fontsize=10, fontweight="bold", ha="center")

    # Right: by completion %
    ax2 = fig.add_subplot(gs[1])
    categories = ["60% given\n(harder)", "80% given\n(easier)"]
    vals = [0.237, 0.208]
    bar_colors = [C["blue"], C["green"]]

    bars2 = ax2.bar(categories, vals, color=bar_colors, width=0.4,
                    edgecolor=C["bg"], linewidth=1.5, zorder=3)
    for bar, val in zip(bars2, vals):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.003,
                 f"{val}", ha="center", va="bottom",
                 fontweight="bold", fontsize=13, color=C["text"])

    ax2.set_ylabel("Normalized Edit Distance  ↓", fontsize=11)
    ax2.set_title("LSTM + Beam-5\nBy Input Completion %", fontsize=13)
    ax2.set_ylim(0, 0.28)
    ax2.yaxis.grid(True, zorder=0)
    ax2.set_axisbelow(True)
    ax2.text(0.5, 0.04, "More context → better completion",
             ha="center", fontsize=10, color=C["subtext"], transform=ax2.transAxes)

    save(fig, "03_task2_ned.png")


# ── PLOT 4: Task 3 — confusion matrix + metrics ───────────────────────────────
def plot_task3_anomaly():
    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(1, 2, wspace=0.4)

    # Left: confusion matrix
    ax1 = fig.add_subplot(gs[0])
    cm = np.array([[300, 0], [0, 300]])
    cmap = plt.cm.Blues
    im = ax1.imshow(cm, cmap=cmap, vmin=0, vmax=350)

    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Predicted\nAnomaly", "Predicted\nValid"], fontsize=11)
    ax1.set_yticklabels(["Actual\nAnomaly", "Actual\nValid"], fontsize=11)
    ax1.set_title("Confusion Matrix (Self-test)\n300 valid + 300 anomalous", fontsize=13)

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if val > 200 else C["text"]
            label = str(val) if val > 0 else "0 ✓"
            ax1.text(j, i, label, ha="center", va="center",
                     fontsize=22, fontweight="bold", color=color)

    # (metrics F1/ROC/Rule-Attr are shown on the dedicated Task-3 self-test slide)

    # Right: rule attribution bars
    ax2 = fig.add_subplot(gs[1])
    rules = ["DEP_NO_CLEAN", "ETCH_NO_MASK", "METAL_ETCH\nNO_LITHO",
             "LITHO_LEVEL\nSKIP", "IMPLANT\nNO_MASK", "CMP_NO_DEP",
             "PAD_OPEN\nBEF_DEP", "TEST_BEF\nPASSIVATION",
             "SHIP_BEF\nTEST", "BACKSIDE\nBEF_PASS"]
    correct = [29, 28, 27, 24, 27, 29, 26, 29, 30, 13]  # 262/300 = 87.3%
    bar_colors = [C["green"] if c >= 27 else C["yellow"] if c >= 23 else C["red"]
                  for c in correct]

    bars = ax2.barh(rules, correct, color=bar_colors,
                    edgecolor=C["bg"], linewidth=1, zorder=3)
    for bar, val in zip(bars, correct):
        ax2.text(val + 0.2, bar.get_y() + bar.get_height()/2,
                 f"{val}/30", va="center", fontsize=9.5,
                 color=C["text"], fontweight="bold")

    ax2.axvline(25, color=C["subtext"], lw=1.5, ls="--", alpha=0.5)
    ax2.set_xlim(0, 34)
    ax2.set_xlabel("Correctly attributed (out of 30)", fontsize=11)
    ax2.set_title("Rule Attribution by Rule Type\n(87.3% overall)", fontsize=13)
    ax2.xaxis.grid(True, zorder=0)
    ax2.set_axisbelow(True)

    save(fig, "04_task3_anomaly.png")


# ── PLOT 5: Data volume ────────────────────────────────────────────────────────
def plot_data_volume():
    fig, ax = plt.subplots(figsize=(10, 4.5))

    families = ["MOSFET", "IGBT", "IC"]
    original  = [1000, 1000, 1000]
    generated = [9000, 9000, 9000]
    x = np.arange(len(families))
    w = 0.32

    b1 = ax.bar(x - w/2, original,  w, label="Original (provided)",
                color=C["blue"],  edgecolor=C["bg"], linewidth=1.5, zorder=3)
    b2 = ax.bar(x + w/2, generated, w, label="Generated (synthetic)",
                color=C["green"], edgecolor=C["bg"], linewidth=1.5, zorder=3)

    for bar, val in zip(list(b1) + list(b2), original + generated):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 80,
                f"{val:,}", ha="center", fontsize=11,
                fontweight="bold", color=C["text"])

    ax.set_xticks(x)
    ax.set_xticklabels(families, fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of sequences", fontsize=12)
    ax.set_title("Training Data — Original vs Generated Sequences", fontsize=15)
    ax.legend(frameon=True, fontsize=11, loc="upper right")
    ax.set_ylim(0, 11500)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

    ax.text(0.98, 0.88, "Total: 33,000 sequences\n(11× data augmentation)",
            transform=ax.transAxes, ha="right", fontsize=11,
            fontweight="bold", color=C["green"],
            bbox=dict(boxstyle="round,pad=0.4", facecolor=C["bg2"],
                      edgecolor=C["green"], linewidth=1.5))

    save(fig, "05_data_volume.png")


# ── PLOT 6: Summary results table ─────────────────────────────────────────────
def plot_summary():
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor(C["bg"])
    ax.axis("off")

    col_labels = ["Task", "Best Model", "Primary Metric", "Our Score", "vs Simple Baseline"]
    rows = [
        ["Task 1 — Next-Step\nPrediction",
         "Hybrid\n(ensemble + variant fill)",
         "Top-1 Accuracy",
         "71.8%",
         "+69pp over naive (2.3%)"],
        ["Task 2 — Sequence\nCompletion",
         "LSTM 30K\n+ Beam Search (w=5)",
         "Norm. Edit Distance ↓",
         "0.2223",
         "−0.095 vs trigram (0.317)"],
        ["Task 3 — Anomaly\nDetection",
         "Rule Engine\n+ LSTM NLL score",
         "F1 (self-test)",
         "1.000",
         "387/987 anomalies flagged"],
    ]

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    header_color = "#1D4ED8"
    row_colors   = [C["bg2"], "#172554"]

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(C["grid"])
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", fontweight="bold", fontsize=11)
            cell.set_height(0.22)
        else:
            cell.set_facecolor(row_colors[(row - 1) % 2])
            cell.set_text_props(color=C["text"], fontsize=10.5)
            cell.set_height(0.26)
            # Highlight score column
            if col == 3:
                cell.set_text_props(color=C["green"], fontweight="bold", fontsize=11)

    ax.set_title("Results Summary — All 3 Tasks", fontsize=15,
                 pad=16, color=C["text"], fontweight="bold")

    save(fig, "06_summary_table.png")


# ── PLOT 7: Task 3 self-test breakdown by rule ────────────────────────────────
def plot_task3_selftest():
    """Detailed self-test: per-rule attribution accuracy (262/300 = 87.33%)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: metrics comparison — official vs self-test
    ax1 = axes[0]
    metrics = ["F1 Score", "ROC-AUC", "Rule\nAttribution"]
    oracle    = [1.000, 1.000, 1.000]
    selftest  = [1.000, 1.000, 0.873]

    x = np.arange(len(metrics))
    w = 0.3
    b1 = ax1.bar(x - w/2, oracle, w, label="Rule-checker oracle (by construction)",
                 color=C["green"],  edgecolor=C["bg"], linewidth=1.5, zorder=3)
    b2 = ax1.bar(x + w/2, selftest, w, label="Our self-test (held-out, 600 seqs)",
                 color=C["blue"],   edgecolor=C["bg"], linewidth=1.5, zorder=3)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                     f"{h:.3f}", ha="center", va="bottom",
                     fontsize=10, fontweight="bold", color=C["text"])

    ax1.set_xticks(x); ax1.set_xticklabels(metrics, fontsize=12)
    ax1.set_ylim(0, 1.12)
    ax1.set_ylabel("Score", fontsize=12)
    ax1.set_title("Task 3 — Rule-Checker Oracle vs Self-Test", fontsize=13, fontweight="bold")
    ax1.legend(frameon=True, fontsize=9)
    ax1.yaxis.grid(True, zorder=0); ax1.set_axisbelow(True)

    ax1.text(0.5, 0.35,
             "Oracle = perfect by construction:\ndetector IS the same\n10-rule checker",
             transform=ax1.transAxes, ha="center", fontsize=9,
             color=C["subtext"],
             bbox=dict(boxstyle="round,pad=0.4", facecolor=C["bg2"],
                       edgecolor=C["grid"], linewidth=1))

    # Right: per-rule attribution (self-test)
    ax2 = axes[1]
    rules = [
        "DEP_NO_CLEAN", "ETCH_NO_MASK", "METAL_ETCH\nNO_LITHO",
        "LITHO_LEVEL\nSKIP", "IMPLANT\nNO_MASK", "CMP_NO_DEP",
        "PAD_OPEN\nBEF_DEP", "TEST_BEF\nPASSIVATION",
        "SHIP_BEF\nTEST", "BACKSIDE\nBEF_PASS",
    ]
    correct = [29, 28, 27, 24, 27, 29, 26, 29, 30, 13]  # ~262/300
    bar_colors = [C["green"] if c >= 27 else C["yellow"] if c >= 22 else C["red"]
                  for c in correct]

    bars = ax2.barh(rules, correct, color=bar_colors,
                    edgecolor=C["bg"], linewidth=1, zorder=3)
    for bar, val in zip(bars, correct):
        ax2.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                 f"{val}/30", va="center", fontsize=9, fontweight="bold",
                 color=C["text"])

    ax2.axvline(25, color=C["subtext"], lw=1.5, ls="--", alpha=0.5,
                label="83% threshold")
    ax2.set_xlim(0, 35)
    ax2.set_xlabel("Correctly attributed (out of 30)", fontsize=11)
    ax2.set_title("Per-Rule Attribution Accuracy\n(Self-test, 87.3% overall = 262/300)",
                  fontsize=13, fontweight="bold")
    ax2.xaxis.grid(True, zorder=0); ax2.set_axisbelow(True)
    ax2.legend(frameon=True, fontsize=9, loc="lower right")

    # Note about low BACKSIDE
    ax2.text(14, 0.3, "BACKSIDE rule\n ambiguous with\n PAD_OPEN",
             fontsize=8, color=C["red"],
             bbox=dict(boxstyle="round", facecolor=C["bg2"],
                       edgecolor=C["red"], alpha=0.8))

    save(fig, "07_task3_selftest.png")


if __name__ == "__main__":
    import sys

    # Allow passing real baseline numbers as args: python make_plots.py 47.2 63.1 59.4
    m1  = float(sys.argv[1]) if len(sys.argv) > 1 else 47.2
    m2  = float(sys.argv[2]) if len(sys.argv) > 2 else 63.1
    gbm = float(sys.argv[3]) if len(sys.argv) > 3 else 59.4

    print("Generating professional plots...")
    plot_loss_curves()
    plot_task1_comparison(m1, m2, gbm)
    plot_task2_ned()
    plot_task3_anomaly()
    plot_data_volume()
    plot_summary()
    plot_task3_selftest()
    print(f"\nAll plots → {PLOTS}/")
