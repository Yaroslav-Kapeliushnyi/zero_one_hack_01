"""
Shared training loop — LSTM and GPT, with scheduled sampling.
Usage:
    python train.py --model lstm --epochs 50
    python train.py --model lstm --epochs 80 --data-size 30k --scheduled-sampling 0.3
    python train.py --model gpt  --epochs 50
    python train.py --model markov
"""

import argparse
import json
import random
import sys
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

sys.path.insert(0, str(Path(__file__).parent))
from data import SequenceDataset, build_vocab, collate_fn, train_val_split, FAMILY_FILES

CKPT_DIR = Path(__file__).parent.parent / "checkpoints"
LOG_DIR  = Path(__file__).parent.parent / "logs"
CKPT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


@torch.no_grad()
def compute_accuracy(model, loader, device, model_type):
    """Top-1 token accuracy on the val set."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)[0] if model_type in ("lstm", "lstm_attn") else model(x).logits
        mask   = y != -100
        correct += (logits.argmax(-1)[mask] == y[mask]).sum().item()
        total   += mask.sum().item()
    return correct / total if total > 0 else 0.0


def train_epoch_scheduled(model, loader, optimizer, device, model_type,
                          ss_prob: float = 0.0):
    """
    Training with optional scheduled sampling.
    ss_prob: probability of using model's own prediction instead of ground truth.
    ss_prob=0.0 → standard teacher forcing.
    """
    model.train()
    total_loss, n = 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if ss_prob == 0.0 or model_type != "lstm":
            # Standard teacher forcing for both models
            logits = model(x)[0] if model_type in ("lstm", "lstm_attn") else model(x).logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   y.view(-1), ignore_index=-100)
        else:
            # Scheduled sampling — only implemented for LSTM
            B, T = x.shape
            hidden = None
            all_logits = []

            for t in range(T):
                if t == 0 or random.random() > ss_prob:
                    inp = x[:, t:t+1]          # teacher force
                else:
                    inp = all_logits[-1].argmax(-1)  # (B, 1) — own prediction

                step_logits, hidden = model(inp, hidden)
                all_logits.append(step_logits)

            logits = torch.cat(all_logits, dim=1)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   y.view(-1), ignore_index=-100)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1

    return total_loss / n


@torch.no_grad()
def eval_epoch(model, loader, device, model_type):
    model.eval()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)[0] if model_type in ("lstm", "lstm_attn") else model(x).logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               y.view(-1), ignore_index=-100)
        total_loss += loss.item()
        n += 1
    return total_loss / n


def build_dataset(vocab, extra_files=None):
    """Build dataset from base 3K sequences + optional extra generated files."""
    base_ds = SequenceDataset(vocab)

    if not extra_files:
        return base_ds

    # Load extra generated CSV files
    from data import load_sequences, FAMILY_TOKEN
    import torch
    from torch.utils.data import Dataset

    class ExtraDataset(Dataset):
        def __init__(self, vocab, csv_paths):
            self.samples = []
            for fam, path in csv_paths:
                if not Path(path).exists():
                    print(f"  WARNING: {path} not found, skipping")
                    continue
                fam_tok = vocab.step2id[FAMILY_TOKEN[fam]]
                seqs = load_sequences(Path(path))
                for seq in seqs.values():
                    ids = [vocab.bos_id, fam_tok] + vocab.encode(seq) + [vocab.eos_id]
                    if len(ids) > 210:
                        ids = ids[:210]
                    self.samples.append(ids)
                print(f"  Loaded {len(seqs)} extra sequences from {Path(path).name}")

        def __len__(self): return len(self.samples)
        def __getitem__(self, idx):
            ids = self.samples[idx]
            return torch.tensor(ids[:-1], dtype=torch.long), \
                   torch.tensor(ids[1:],  dtype=torch.long)

    extra_ds = ExtraDataset(vocab, extra_files)
    if len(extra_ds) == 0:
        return base_ds
    combined = ConcatDataset([base_ds, extra_ds])
    print(f"Dataset: {len(base_ds)} base + {len(extra_ds)} extra = {len(combined)} total")
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    choices=["lstm", "gpt", "markov", "lstm_attn"], default="lstm")
    parser.add_argument("--epochs",   type=int,   default=50)
    parser.add_argument("--batch",    type=int,   default=64)
    parser.add_argument("--lr",       type=float, default=3e-4)
    parser.add_argument("--embed",    type=int,   default=128)
    parser.add_argument("--hidden",   type=int,   default=512)
    parser.add_argument("--layers",   type=int,   default=2)
    parser.add_argument("--n_layer",  type=int,   default=8)
    parser.add_argument("--n_head",   type=int,   default=8)
    parser.add_argument("--n_embd",   type=int,   default=256)
    parser.add_argument("--dropout",  type=float, default=0.1)
    parser.add_argument("--device",   default="cuda")
    parser.add_argument("--scheduled-sampling", type=float, default=0.0,
                        help="Max scheduled sampling probability (0=off, 0.3=recommended)")
    parser.add_argument("--extra-data", nargs="*",
                        help="Extra CSV files: mosfet:path igbt:path ic:path")
    parser.add_argument("--suffix",   default="",
                        help="Checkpoint name suffix e.g. '_30k'")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"Device: {device}")

    vocab = build_vocab()
    print(f"Vocab: {len(vocab)}")

    if args.model == "markov":
        from models.markov import MarkovModel
        from data import load_all_sequences
        seqs, _ = load_all_sequences()
        m = MarkovModel(order=3)
        m.train(seqs)
        m.save(CKPT_DIR / "markov_order3.pkl")
        print("Markov saved.")
        return

    # Parse extra data files
    extra_files = []
    if args.extra_data:
        for entry in args.extra_data:
            fam, path = entry.split(":", 1)
            extra_files.append((fam.lower(), path))

    dataset = build_dataset(vocab, extra_files)
    train_ds, val_ds = train_val_split(dataset)
    collate = partial(collate_fn, pad_id=vocab.pad_id)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=collate, num_workers=4, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              collate_fn=collate, num_workers=4, pin_memory=pin)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    if args.model == "lstm":
        from models.lstm_baseline import LSTMModel
        model = LSTMModel(len(vocab), args.embed, args.hidden,
                          args.layers, args.dropout, vocab.pad_id).to(device)
    elif args.model == "lstm_attn":
        from models.lstm_attention import LSTMWithAttention
        model = LSTMWithAttention(len(vocab), args.embed, args.hidden,
                                  args.layers, num_heads=8,
                                  dropout=args.dropout, pad_id=vocab.pad_id).to(device)
    else:
        from models.gpt_model import build_gpt_model
        model = build_gpt_model(len(vocab), vocab.bos_id, vocab.eos_id,
                                args.n_layer, args.n_head, args.n_embd,
                                dropout=args.dropout).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val, log = float("inf"), []
    ckpt_name = f"{args.model}{args.suffix}_best.pt"

    for epoch in range(1, args.epochs + 1):
        # Linearly ramp scheduled sampling probability
        ss_prob = args.scheduled_sampling * min(1.0, epoch / (args.epochs * 0.5))

        t0 = time.time()
        train_loss = train_epoch_scheduled(model, train_loader, optimizer, device,
                                           args.model, ss_prob)
        val_loss   = eval_epoch(model, val_loader, device, args.model)
        val_acc    = compute_accuracy(model, val_loader, device, args.model)
        scheduler.step()

        ss_str = f" ss={ss_prob:.2f}" if ss_prob > 0 else ""
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train={train_loss:.4f} val={val_loss:.4f} "
              f"acc={val_acc:.4f}{ss_str} | {time.time()-t0:.1f}s")
        log.append({"epoch": epoch, "train": train_loss,
                    "val": val_loss, "val_acc": val_acc})

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "val_loss": val_loss, "vocab_size": len(vocab), "args": vars(args),
            }, CKPT_DIR / ckpt_name)
            print(f"  ✓ saved → {ckpt_name}")

    with open(LOG_DIR / f"{args.model}{args.suffix}_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nDone. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
