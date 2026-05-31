"""
Attention-Augmented LSTM
------------------------
LSTM hidden states are passed through a causal self-attention layer so the
model can look back at any earlier step directly — targeting the long-range
drift observed in Task 2 completions past step ~50.

Training:  forward(input_ids, hidden=None) → (logits, hidden)
           Full-sequence causal self-attention; drop-in for LSTMModel in train.py.

Inference: complete_sequence() maintains a KV cache (past_states) so each new
           step cross-attends to every prior LSTM hidden state.
"""

import torch
import torch.nn as nn
from data import Vocabulary


class _CausalSelfAttention(nn.Module):
    """Scaled dot-product self-attention with optional causal mask."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query: torch.Tensor,
                key_value: torch.Tensor | None = None,
                causal: bool = True) -> torch.Tensor:
        """
        query:     (B, T, H)
        key_value: (B, S, H) — if None, self-attention over query
        causal:    mask future positions (only makes sense when T == S)
        Returns:   (B, T, H)  residual + layernorm applied
        """
        kv = key_value if key_value is not None else query
        T, S = query.size(1), kv.size(1)

        attn_mask = None
        if causal and T == S:
            # Upper-triangular mask: position i cannot attend to j > i
            attn_mask = torch.triu(
                torch.ones(T, S, device=query.device, dtype=torch.bool),
                diagonal=1,
            )

        out, _ = self.attn(query, kv, kv, attn_mask=attn_mask, need_weights=False)
        return self.norm(query + out)


class LSTMWithAttention(nn.Module):
    """2-layer LSTM + causal self-attention over hidden states."""

    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 hidden_dim: int = 512, num_layers: int = 2,
                 num_heads: int = 8, dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"

        self.pad_id     = pad_id
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = _CausalSelfAttention(hidden_dim, num_heads, dropout)
        self.dropout   = nn.Dropout(dropout)
        self.lm_head   = nn.Linear(hidden_dim, vocab_size)

    # ── Standard forward (train.py compatible) ──────────────────────────────────

    def forward(self, input_ids: torch.Tensor,
                hidden=None) -> tuple[torch.Tensor, tuple]:
        """Full-sequence causal self-attention. Drop-in for LSTMModel."""
        x   = self.dropout(self.embedding(input_ids))       # (B, T, E)
        out, hidden = self.lstm(x, hidden)                  # (B, T, H)
        attended = self.attention(out, causal=True)         # (B, T, H)
        logits   = self.lm_head(self.dropout(attended))     # (B, T, V)
        return logits, hidden

    # ── Single-step forward with KV cache (for inference) ───────────────────────

    def forward_step(self, input_id: torch.Tensor, hidden,
                     past_states: torch.Tensor | None
                     ) -> tuple[torch.Tensor, tuple, torch.Tensor]:
        """
        input_id:    (B, 1) — one token
        past_states: (B, t, H) — all prior LSTM hidden states (KV cache)
        Returns:     logits (B, 1, V), new_hidden, updated past_states (B, t+1, H)
        """
        x = self.dropout(self.embedding(input_id))   # (B, 1, E)
        step_out, hidden = self.lstm(x, hidden)      # (B, 1, H)

        kv = step_out if past_states is None else torch.cat([past_states, step_out], dim=1)
        attended = self.attention(step_out, key_value=kv, causal=False)   # (B, 1, H)
        logits = self.lm_head(self.dropout(attended))                      # (B, 1, V)
        return logits, hidden, kv


# ── Inference helpers ──────────────────────────────────────────────────────────

@torch.no_grad()
def predict_next_top_k(model: LSTMWithAttention, prefix_ids: list[int],
                       vocab: Vocabulary, k: int = 5,
                       device: str = "cuda") -> list[str]:
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(dev)

    # Encode the full prefix in one pass (training-mode forward)
    x = torch.tensor([prefix_ids], dtype=torch.long, device=dev)
    logits, _ = model(x)
    valid_vocab = len(vocab.id2step)
    top_ids = logits[0, -1, :valid_vocab].topk(k).indices.tolist()
    return [vocab.id2step[i] for i in top_ids]


@torch.no_grad()
def complete_sequence(model: LSTMWithAttention, prefix_ids: list[int],
                      vocab: Vocabulary, max_new: int = 160,
                      device: str = "cuda") -> list[str]:
    """Autoregressive completion with KV cache for full-history attention."""
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(dev)

    # ── Encode prefix to warm up LSTM state + build initial KV cache ──────────
    x = torch.tensor([prefix_ids], dtype=torch.long, device=dev)
    emb = model.dropout(model.embedding(x))        # (1, T_prefix, E)
    past_states, hidden = model.lstm(emb)          # past_states: (1, T_prefix, H)

    valid_vocab = len(vocab.id2step)  # clamp for models trained on larger vocab

    # First prediction off the last prefix position
    step_logits = model.lm_head(
        model.dropout(
            model.attention(past_states[:, -1:, :], key_value=past_states, causal=False)
        )
    )
    next_id = step_logits[0, 0, :valid_vocab].argmax().item()

    generated = []
    _STOP = {"[PAD]", "[UNK]", "[BOS]", "[CLS]", "[MOSFET]", "[IGBT]", "[IC]"}

    while next_id != vocab.eos_id and len(generated) < max_new:
        step = vocab.id2step[next_id]
        if step in _STOP:
            break
        if len(generated) >= 3 and step == generated[-1] == generated[-2]:
            break
        generated.append(step)

        inp = torch.tensor([[next_id]], dtype=torch.long, device=dev)
        step_logits, hidden, past_states = model.forward_step(inp, hidden, past_states)
        next_id = step_logits[0, 0, :valid_vocab].argmax().item()

    return generated


@torch.no_grad()
def sequence_log_prob(model: LSTMWithAttention, seq_ids: list[int],
                      device: str = "cuda") -> float:
    """Mean NLL — higher = more anomalous."""
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(dev)
    x = torch.tensor([seq_ids[:-1]], dtype=torch.long, device=dev)
    y = torch.tensor([seq_ids[1:]],  dtype=torch.long, device=dev)
    logits, _ = model(x)
    loss = nn.CrossEntropyLoss(ignore_index=-100)(
        logits.view(-1, logits.size(-1)), y.view(-1)
    )
    return loss.item()


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data import build_vocab

    vocab = build_vocab()
    model = LSTMWithAttention(len(vocab), pad_id=vocab.pad_id)
    x = torch.randint(0, len(vocab), (2, 50))
    logits, _ = model(x)
    print(f"LSTMWithAttention OK — vocab={len(vocab)}, output={logits.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
