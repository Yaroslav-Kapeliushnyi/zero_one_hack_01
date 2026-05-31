"""
LSTM Baseline — Yehor
2-layer unidirectional LSTM for next-step prediction and sequence completion.
"""

import torch
import torch.nn as nn
from data import Vocabulary


class LSTMModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 hidden_dim: int = 512, num_layers: int = 2,
                 dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor,
                hidden=None) -> tuple[torch.Tensor, tuple]:
        x = self.dropout(self.embedding(input_ids))
        out, hidden = self.lstm(x, hidden)
        logits = self.lm_head(self.dropout(out))
        return logits, hidden


# ── Inference ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_next_top_k(model: LSTMModel, prefix_ids: list[int],
                       vocab: Vocabulary, k: int = 5,
                       device: str = "cuda") -> list[str]:
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(dev)
    x = torch.tensor([prefix_ids], dtype=torch.long, device=dev)
    logits, _ = model(x)
    top_ids = logits[0, -1].topk(k).indices.tolist()
    return [vocab.id2step[i] for i in top_ids]


@torch.no_grad()
def complete_sequence(model: LSTMModel, prefix_ids: list[int],
                      vocab: Vocabulary, max_new: int = 160,
                      device: str = "cuda") -> list[str]:
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(dev)

    generated = []
    x = torch.tensor([prefix_ids], dtype=torch.long, device=dev)
    logits, hidden = model(x)
    next_id = logits[0, -1].argmax().item()

    while next_id != vocab.eos_id and len(generated) < max_new:
        step = vocab.id2step[next_id]
        if step in ("[PAD]", "[UNK]", "[BOS]", "[CLS]", "[MOSFET]", "[IGBT]", "[IC]"):
            break
        # Simple repetition guard
        if len(generated) >= 3 and step == generated[-1] == generated[-2]:
            break
        generated.append(step)
        x = torch.tensor([[next_id]], dtype=torch.long, device=dev)
        logits, hidden = model(x, hidden)
        next_id = logits[0, -1].argmax().item()

    return generated


@torch.no_grad()
def sequence_log_prob(model: LSTMModel, seq_ids: list[int],
                      device: str = "cuda") -> float:
    """Returns mean NLL — higher = more anomalous."""
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(dev)
    x = torch.tensor([seq_ids[:-1]], dtype=torch.long, device=dev)
    y = torch.tensor([seq_ids[1:]],  dtype=torch.long, device=dev)
    logits, _ = model(x)
    loss = nn.CrossEntropyLoss(ignore_index=-100)(
        logits.view(-1, logits.size(-1)), y.view(-1))
    return loss.item()


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data import build_vocab
    vocab = build_vocab()
    model = LSTMModel(len(vocab), pad_id=vocab.pad_id)
    x = torch.randint(0, len(vocab), (2, 50))
    logits, _ = model(x)
    print(f"LSTM OK — vocab={len(vocab)}, output={logits.shape}")
