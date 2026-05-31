"""
GPT Transformer — Olga
Small causal transformer trained from scratch on process sequences.
Uses HuggingFace GPT2 architecture with custom vocab.
"""

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel
from data import Vocabulary


def build_gpt_model(vocab_size: int, bos_id: int, eos_id: int,
                    n_layer: int = 8, n_head: int = 8,
                    n_embd: int = 256, context_length: int = 210,
                    dropout: float = 0.1) -> GPT2LMHeadModel:
    """Build a small GPT-2 style model with custom vocab."""
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=context_length,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        resid_pdrop=dropout,
        embd_pdrop=dropout,
        attn_pdrop=dropout,
        bos_token_id=bos_id,   # correct: vocab BOS (2), not PAD (0)
        eos_token_id=eos_id,   # correct: vocab EOS (3), not PAD (0)
    )
    model = GPT2LMHeadModel(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GPT model: {n_layer}L/{n_head}H/{n_embd}D — {n_params:,} params")
    return model


# ── Inference ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_next_top_k(model: GPT2LMHeadModel, prefix_ids: list[int],
                       vocab: Vocabulary, k: int = 5,
                       device: str = "cuda") -> list[str]:
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(dev)
    x = torch.tensor([prefix_ids], dtype=torch.long, device=dev)
    last_logits = model(x).logits[0, -1]
    top_ids = last_logits.topk(k).indices.tolist()
    return [vocab.id2step[i] for i in top_ids]


@torch.no_grad()
def complete_sequence(model: GPT2LMHeadModel, prefix_ids: list[int],
                      vocab: Vocabulary, max_new: int = 160,
                      device: str = "cuda") -> list[str]:
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(dev)

    generated = []
    ids = list(prefix_ids)

    for _ in range(max_new):
        x = torch.tensor([ids], dtype=torch.long, device=dev)
        next_id = model(x).logits[0, -1].argmax().item()
        if next_id == vocab.eos_id:
            break
        step = vocab.id2step[next_id]
        if step in ("[PAD]", "[UNK]", "[BOS]", "[CLS]", "[MOSFET]", "[IGBT]", "[IC]"):
            break
        # Repetition guard
        if len(generated) >= 3 and step == generated[-1] == generated[-2]:
            break
        generated.append(step)
        ids.append(next_id)

    return generated


@torch.no_grad()
def sequence_log_prob(model: GPT2LMHeadModel, seq_ids: list[int],
                      device: str = "cuda") -> float:
    """Mean NLL — higher = more anomalous."""
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(dev)
    x = torch.tensor([seq_ids[:-1]], dtype=torch.long, device=dev)
    y = torch.tensor([seq_ids[1:]],  dtype=torch.long, device=dev)
    logits = model(x).logits
    loss = torch.nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), y.view(-1))
    return loss.item()


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data import build_vocab
    vocab = build_vocab()
    model = build_gpt_model(len(vocab), bos_id=vocab.bos_id, eos_id=vocab.eos_id)
    x = torch.randint(0, len(vocab), (2, 50))
    out = model(x)
    print(f"GPT OK — vocab={len(vocab)}, logits={out.logits.shape}")
