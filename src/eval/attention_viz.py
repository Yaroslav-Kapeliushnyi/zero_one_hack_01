"""
Attention map visualization for LSTMWithAttention.
Shows which historical steps the model focuses on when predicting each next step.

Usage:
    python src/eval/attention_viz.py
Output:
    plots/attention_map.png
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

CKPT_DIR = ROOT / "checkpoints"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


# ── Patch attention layer to capture weights ──────────────────────────────────

_captured_weights = {}

def _make_hook(name):
    def hook(module, input, output):
        # MultiheadAttention forward returns (output, attn_weights) when need_weights=True
        pass
    return hook


def get_attention_weights(model, input_ids: torch.Tensor, device) -> torch.Tensor:
    """
    Run one forward pass and return attention weight matrix (T, T).
    Hooks into _CausalSelfAttention to re-run attn with need_weights=True.
    """
    weights_store = {}

    def attn_hook(module, input, output):
        # input[0] is the query tensor passed to _CausalSelfAttention.forward
        query = input[0]                              # (B, T, H)
        kv    = input[1] if len(input) > 1 and input[1] is not None else query
        T = query.size(1)
        attn_mask = None
        if T > 1:
            attn_mask = torch.triu(
                torch.ones(T, T, device=query.device, dtype=torch.bool), diagonal=1)
        with torch.no_grad():
            _, w = module.attn(query, kv, kv,
                               attn_mask=attn_mask,
                               need_weights=True,
                               average_attn_weights=True)
        if w is not None:
            weights_store["w"] = w.detach().cpu()   # (B, T, T) or (T, T)

    handle = model.attention.register_forward_hook(attn_hook)
    with torch.no_grad():
        model(input_ids)
    handle.remove()

    return weights_store.get("w", None)


def visualize_attention(sequence_steps: list[str], vocab, model, device,
                        out_path: Path, title: str = "Attention Map"):
    """
    Generate and save an attention heatmap for a given sequence.

    Args:
        sequence_steps: list of step names (original or canonical, already encoded)
        vocab:          canonical Vocabulary
        model:          LSTMWithAttention instance
        device:         torch device
        out_path:       where to save the PNG
        title:          plot title
    """
    from data import CANONICAL_STEPS, FAMILY_TOKEN, encode_prefix

    # Encode: use MOSFET as default family for demo (doesn't affect attention shape)
    canonical = [CANONICAL_STEPS.get(s, s) for s in sequence_steps]
    ids = encode_prefix(vocab, canonical, "mosfet", apply_canonical=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    # Capture attention weights (T, T) or (B, T, T)
    attn = get_attention_weights(model, x, device)
    if attn is None:
        print("Could not capture attention weights — model may not support hooks.")
        return

    if attn.dim() == 3:
        attn = attn[0]    # (T, T)

    T = attn.shape[0]

    # Labels: trim to actual step names (remove BOS/family token prefix)
    raw_ids = ids
    labels = []
    for i, tok_id in enumerate(raw_ids):
        name = vocab.id2step.get(tok_id, "?")
        if name.startswith("["):
            labels.append(name)
        else:
            # Abbreviate long names for readability
            short = name.replace(" ", "\n")[:20]
            labels.append(short)

    labels = labels[:T]

    # ── Plot ──────────────────────────────────────────────────────────────────
    BG = "#0F172A"
    fig, ax = plt.subplots(figsize=(min(T * 0.35, 18), min(T * 0.35, 18)))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    cmap = plt.cm.Blues
    im = ax.imshow(attn.numpy(), cmap=cmap, aspect="auto", vmin=0, vmax=attn.max())

    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xticklabels(labels, rotation=90, fontsize=5, color="white")
    ax.set_yticklabels(labels, fontsize=5, color="white")

    ax.set_xlabel("Key (attended-to steps)", color="white", fontsize=9)
    ax.set_ylabel("Query (predicting steps)", color="white", fontsize=9)
    ax.set_title(f"{title}\n(brighter = stronger attention)", color="white",
                 fontsize=11, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"Attention map saved → {out_path}")


def run():
    from data import build_vocab, load_sequences, FAMILY_FILES, CANONICAL_STEPS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load canonical vocab + LSTMWithAttention
    vocab = build_vocab()

    attn_ckpt = CKPT_DIR / "lstm_attn_canonical_best.pt"
    if not attn_ckpt.exists():
        print(f"Checkpoint not found: {attn_ckpt}")
        print("Run: python src/train.py --model lstm_attn --suffix _canonical")
        return

    ckpt = torch.load(attn_ckpt, map_location=device, weights_only=False)
    from models.lstm_attention import LSTMWithAttention
    a = ckpt["args"]
    model = LSTMWithAttention(len(vocab), a.get("embed", 128), a.get("hidden", 512),
                              a.get("layers", 2), num_heads=8,
                              dropout=a.get("dropout", 0.1),
                              pad_id=vocab.pad_id).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded LSTMWithAttention (val_loss={ckpt['val_loss']:.4f})")

    # Pick a sample sequence from training data (first MOSFET sequence, first 30 steps)
    seqs = load_sequences(FAMILY_FILES["mosfet"], apply_canonical=True)
    sample = list(seqs.values())[0][:30]
    print(f"Sample sequence ({len(sample)} steps): {sample[:5]}...")

    visualize_attention(
        sequence_steps=sample,
        vocab=vocab,
        model=model,
        device=device,
        out_path=PLOTS_DIR / "attention_map.png",
        title="LSTM+Attention — Step Attention Map\n(MOSFET process, first 30 steps)",
    )


if __name__ == "__main__":
    run()
