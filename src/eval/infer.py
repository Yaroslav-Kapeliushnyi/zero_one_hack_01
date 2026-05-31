"""
Inference script — produces all 3 submission CSVs.

Self-eval mode (uses our val set):
    python infer.py --model lstm --self-eval

Official submission mode:
    python infer.py --model lstm \
        --valid-input data/eval_input_valid.csv \
        --anomaly-input data/eval_input_anomaly.csv

Output files (official submission format):
    results/nextstep_<model>.csv    — EXAMPLE_ID, RANK_1..5
    results/completion_<model>.csv  — EXAMPLE_ID, PREDICTED_SEQUENCE
    results/anomaly_<model>.csv     — EXAMPLE_ID, IS_VALID, SCORE, PREDICTED_RULE
"""

import argparse
import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data import (Vocabulary, SequenceDataset, build_vocab,
                  encode_prefix, parse_pipe_sequence, train_val_split,
                  FAMILY_TOKEN, FAMILY_FILES, CANONICAL_STEPS, VARIANTS_OF,
                  load_sequences)

CKPT_DIR = ROOT / "checkpoints"
RESULTS  = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# ── Load generate_sequences validator ONCE at module level ────────────────────
_GEN_MOD = None

def _get_validator():
    global _GEN_MOD
    if _GEN_MOD is None:
        spec = importlib.util.spec_from_file_location(
            "generate_sequences", ROOT / "data" / "generate_sequences.py")
        _GEN_MOD = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_GEN_MOD)
    return _GEN_MOD


def rule_based_anomaly(steps: list[str]) -> tuple[int, str]:
    """Returns (is_valid, predicted_rule). Uses official validator."""
    try:
        mod = _get_validator()
        violations = mod.validate_sequence(steps)
        if violations:
            return 0, violations[0].rule
        return 1, ""
    except Exception:
        return 1, ""


# ── Model loaders ─────────────────────────────────────────────────────────────

def load_lstm(vocab, device, ckpt_name=None):
    from models.lstm_baseline import LSTMModel
    if ckpt_name is None:
        ckpt_name = "lstm_30k_best.pt" if (CKPT_DIR / "lstm_30k_best.pt").exists() else "lstm_best.pt"
    ckpt_file = ckpt_name
    print(f"Loading {ckpt_file}")
    ckpt = torch.load(CKPT_DIR / ckpt_file, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = LSTMModel(len(vocab), a["embed"], a["hidden"],
                      a["layers"], a["dropout"], vocab.pad_id).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded LSTM (val_loss={ckpt['val_loss']:.4f})")
    return model, ckpt["val_loss"]


def load_gpt(vocab, device):
    from models.gpt_model import build_gpt_model
    # Prefer canonical retrain if available
    gpt_file = ("gpt_canonical_best.pt" if (CKPT_DIR / "gpt_canonical_best.pt").exists()
                else "gpt_best.pt")
    ckpt = torch.load(CKPT_DIR / gpt_file, map_location=device, weights_only=False)
    a = ckpt["args"]
    # Check vocab size matches — GPT trained before canonical vocab change won't load
    ckpt_vocab_size = ckpt["model_state"]["transformer.wte.weight"].shape[0]
    if ckpt_vocab_size != len(vocab):
        raise RuntimeError(
            f"GPT vocab size mismatch: checkpoint has {ckpt_vocab_size} tokens "
            f"but current vocab has {len(vocab)}. "
            f"GPT needs to be retrained on canonical vocab.")
    model = build_gpt_model(len(vocab), vocab.bos_id, vocab.eos_id,
                            a["n_layer"], a["n_head"], a["n_embd"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded GPT (val_loss={ckpt['val_loss']:.4f})")
    return model, ckpt["val_loss"]


def load_markov(ckpt_name="markov_order3.pkl"):
    from models.markov import MarkovModel
    m = MarkovModel.load(CKPT_DIR / ckpt_name)
    print(f"Loaded Markov {ckpt_name} (order={m.order}, vocab={len(m.vocab)})")
    return m


def load_tcn(vocab, device, ckpt_name="tcn_best.pt"):
    from models.tcn import CausalTCNLM
    ckpt = torch.load(CKPT_DIR / ckpt_name, map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    model = CausalTCNLM(
        vocab_size=len(vocab),
        d_model=a.get("d_model", 256),
        num_channels=a.get("num_channels", [256, 256, 256, 256]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded TCN (val_loss={ckpt['val_loss']:.4f})")
    return model, ckpt["val_loss"]


def load_lstm_attn(vocab, device, ckpt_name=None):
    from models.lstm_attention import LSTMWithAttention
    if ckpt_name is None:
        for candidate in ("lstm_attn_canonical_30k_best.pt",
                          "lstm_attn_canonical_best.pt",
                          "lstm_attn_30k_best.pt",
                          "lstm_attn_best.pt"):
            if (CKPT_DIR / candidate).exists():
                ckpt_name = candidate
                break
        if ckpt_name is None:
            raise FileNotFoundError("No lstm_attn checkpoint found in checkpoints/")
    print(f"Loading {ckpt_name}")
    ckpt = torch.load(CKPT_DIR / ckpt_name, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = LSTMWithAttention(
        len(vocab), a["embed"], a["hidden"], a["layers"],
        a.get("heads", 8), a["dropout"], vocab.pad_id,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded LSTMWithAttention (val_loss={ckpt['val_loss']:.4f})")
    return model, ckpt["val_loss"]


# ── Task 1: Next-step prediction ──────────────────────────────────────────────

_SPECIAL = {"[PAD]","[UNK]","[BOS]","[EOS]","[CLS]","[MOSFET]","[IGBT]","[IC]"}

# Per-family ensemble weights (empirically derived from 300 held-out val sequences).
# lstm_attn and gpt share 80% using inverse-NLL weights; Markov gets fixed 20%.
#   mosfet: lstm_attn NLL=0.2384, gpt NLL=0.2403 → lstm_attn fraction = 0.502
#   igbt:   lstm_attn NLL=0.2424, gpt NLL=0.2453 → lstm_attn fraction = 0.503
#   ic:     lstm_attn NLL=0.2913, gpt NLL=0.2950 → lstm_attn fraction = 0.503
FAMILY_ENSEMBLE_WEIGHTS = {
    "mosfet": {"w_lstm_attn": 0.402, "w_gpt": 0.398, "w_markov": 0.20},
    "igbt":   {"w_lstm_attn": 0.402, "w_gpt": 0.398, "w_markov": 0.20},
    "ic":     {"w_lstm_attn": 0.403, "w_gpt": 0.397, "w_markov": 0.20},
}


def decanonicalize(step: str, original_prefix: list[str], markov_orig) -> str:
    """
    After the canonical LSTM predicts a step, pick the best ORIGINAL-vocabulary
    variant using the ORIGINAL Markov (markov_order3.pkl) as a tie-breaker.

    Key design decision:
        We use markov_orig = MarkovModel trained on ORIGINAL step names (pre-canonical).
        It knows the trigram distribution over the full original vocabulary, including
        both "STRIP RESIST" and "STRIP PHOTORESIST" as distinct tokens.
        Querying it with the ORIGINAL prefix (non-canonicalized) gives us exactly the
        local trigram preference for which surface form is most likely here.

    This is the correct approach because:
        - markov_orig was NOT retrained; it still has original-vocabulary n-gram counts
        - original_prefix comes from parse_pipe_sequence() — original step names
        - Together they give a genuine context-sensitive preference per variant

    Args:
        step:            canonical step name predicted by LSTM
        original_prefix: raw step names from parse_pipe_sequence() — NOT canonicalized
        markov_orig:     MarkovModel loaded from markov_order3.pkl (original names)

    Returns:
        best original-vocabulary variant, or step itself if no disambiguation needed
    """
    from data import VARIANTS_OF

    variants = VARIANTS_OF.get(step)
    if not variants:
        return step          # non-canonical step — no disambiguation needed

    if len(variants) == 1:
        return variants[0]   # only one original form exists

    # Query ORIGINAL Markov with ORIGINAL prefix — correct vocabulary alignment
    ranked = markov_orig.predict_next_top_k(original_prefix, k=len(markov_orig.vocab) + 1)

    # Find which variant Markov ranks highest given this specific trigram context
    rank_of: dict[str, int] = {v: len(ranked) for v in variants}  # default: not found
    for rank, s in enumerate(ranked):
        if s in rank_of:
            rank_of[s] = rank

    return min(rank_of, key=rank_of.get)

def _get_logits_last(model, model_type, x):
    """Unified logit extraction: returns (vocab_size,) at last position."""
    if model_type in ("lstm", "lstm_attn"):
        return model(x)[0][0, -1]       # both return (logits, hidden)
    elif model_type == "tcn":
        return model(x)[0, -1]          # TCN returns (B, T, V) directly
    else:
        return model(x).logits[0, -1]   # GPT


@torch.no_grad()
def predict_next_top5(model, model_type, vocab, prefix_ids, device,
                      markov=None, use_constraints=True,
                      family="mosfet",
                      original_prefix: list[str] = None,
                      markov_orig=None) -> list[str]:
    # canonical steps decoded from prefix_ids (for constraint decoder)
    steps = [vocab.id2step[i] for i in prefix_ids if vocab.id2step[i] not in _SPECIAL]

    if model_type == "markov":
        return markov.predict_next_top_k(steps, k=5)

    x = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    logits = _get_logits_last(model, model_type, x)

    if use_constraints:
        from eval.constraint_decoder import build_constraint_mask
        mask = build_constraint_mask(steps, vocab, device=device)
        logits = logits + mask

    top_ids = logits.topk(10).indices.tolist()
    canonical = [vocab.id2step[i] for i in top_ids if vocab.id2step[i] not in _SPECIAL][:5]

    # De-canonicalize: original Markov + original prefix → best surface variant
    if markov_orig is not None and original_prefix is not None:
        return [decanonicalize(s, original_prefix, markov_orig) for s in canonical]
    return canonical


# ── Task 1: Ensemble next-step (LSTM + GPT + Markov soft-vote) ───────────────

@torch.no_grad()
def predict_next_top5_ensemble(lstm_model, markov_model, vocab,
                               prefix_ids, device,
                               gpt_model=None, tcn_model=None,
                               w_lstm=0.45, w_gpt=0.35, w_markov=0.20,
                               family="mosfet",
                               original_prefix: list[str] = None,
                               markov_orig=None) -> list[str]:
    """
    Soft-vote ensemble: LSTM (recurrent) + GPT (self-attention) + Markov (frequency prior).
    LSTM and GPT have different architectural biases → complementary errors.
    """
    x = torch.tensor([prefix_ids], dtype=torch.long, device=device)

    # LSTM log-probs
    lstm_lp = torch.log_softmax(
        _get_logits_last(lstm_model, "lstm", x), dim=-1).cpu()

    # GPT log-probs (if available)
    if gpt_model is not None:
        gpt_lp = torch.log_softmax(
            _get_logits_last(gpt_model, "gpt", x), dim=-1).cpu()
    else:
        gpt_lp = lstm_lp  # fallback: double-weight LSTM

    # Markov log-probs over full vocab (rank-based approximation)
    steps = [vocab.id2step[i] for i in prefix_ids if vocab.id2step[i] not in _SPECIAL]
    markov_lp = torch.full((len(vocab),), -20.0)
    top_markov = markov_model.predict_next_top_k(steps, k=len(vocab))
    for rank, step in enumerate(top_markov):
        if step in vocab.step2id:
            markov_lp[vocab.step2id[step]] = -rank * 0.5

    if gpt_model is None:
        combined = (w_lstm + w_gpt) * lstm_lp + w_markov * markov_lp
    else:
        combined = w_lstm * lstm_lp + w_gpt * gpt_lp + w_markov * markov_lp

    top_ids  = combined.topk(10).indices.tolist()
    canonical = [vocab.id2step[i] for i in top_ids if vocab.id2step[i] not in _SPECIAL][:5]

    # De-canonicalize: original Markov + original prefix → best surface variant
    if markov_orig is not None and original_prefix is not None:
        return [decanonicalize(s, original_prefix, markov_orig) for s in canonical]
    return canonical


# ── Task 1: Context-Aware Ensemble (LSTM-Attn + GPT + Markov, per-family weights) ──

@torch.no_grad()
def predict_next_top5_context_aware(lstm_attn_model, gpt_model, markov_model, vocab,
                                    prefix_ids, device,
                                    family="mosfet",
                                    original_prefix: list[str] = None,
                                    markov_orig=None) -> list[str]:
    """
    Context-aware weighted ensemble with per-family calibrated weights.
    LSTM-Attn excels at long-range structure; GPT at local transitions; Markov as prior.
    Weights are derived from held-out val NLL per family (see FAMILY_ENSEMBLE_WEIGHTS).
    """
    weights  = FAMILY_ENSEMBLE_WEIGHTS.get(family, FAMILY_ENSEMBLE_WEIGHTS["mosfet"])
    w_la     = weights["w_lstm_attn"]
    w_gpt    = weights["w_gpt"]
    w_markov = weights["w_markov"]

    x = torch.tensor([prefix_ids], dtype=torch.long, device=device)

    la_lp = torch.log_softmax(
        _get_logits_last(lstm_attn_model, "lstm_attn", x), dim=-1).cpu()

    if gpt_model is not None:
        gpt_lp = torch.log_softmax(
            _get_logits_last(gpt_model, "gpt", x), dim=-1).cpu()
    else:
        gpt_lp  = la_lp
        w_la   += w_gpt
        w_gpt   = 0.0

    steps = [vocab.id2step[i] for i in prefix_ids if vocab.id2step[i] not in _SPECIAL]
    markov_lp = torch.full((len(vocab),), -20.0)
    for rank, step in enumerate(markov_model.predict_next_top_k(steps, k=len(vocab))):
        if step in vocab.step2id:
            markov_lp[vocab.step2id[step]] = -rank * 0.5

    combined  = w_la * la_lp + w_gpt * gpt_lp + w_markov * markov_lp
    top_ids   = combined.topk(10).indices.tolist()
    canonical = [vocab.id2step[i] for i in top_ids if vocab.id2step[i] not in _SPECIAL][:5]

    if markov_orig is not None and original_prefix is not None:
        return [decanonicalize(s, original_prefix, markov_orig) for s in canonical]
    return canonical


# ── Task 4: Perplexity routing for OOD / unknown family ──────────────────────

@torch.no_grad()
def route_family(model, model_type, vocab, steps: list[str], device) -> str:
    """
    For an unknown 4th product family: run the prefix with each of the 3 known
    family conditioning tokens, pick the one that gives the lowest NLL.
    Returns the best family name ('mosfet', 'igbt', or 'ic').
    """
    best_family, best_nll = None, float('inf')
    for fam in ("mosfet", "igbt", "ic"):
        prefix_ids = encode_prefix(vocab, steps, fam)
        if len(prefix_ids) < 2:
            continue
        x = torch.tensor([prefix_ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([prefix_ids[1:]],  dtype=torch.long, device=device)
        if model_type == "lstm":
            logits, _ = model(x)
        elif model_type == "gpt":
            logits = model(x).logits
        else:
            logits, _ = model(x)
        nll = nn.CrossEntropyLoss()(
            logits.view(-1, logits.size(-1)), y.view(-1)).item()
        if nll < best_nll:
            best_nll, best_family = nll, fam
    return best_family or "mosfet"


# ── Task 2: Sequence completion ───────────────────────────────────────────────

@torch.no_grad()
def complete(model, model_type, vocab, prefix_ids, device,
             max_new=160, markov=None) -> list[str]:
    if model_type == "markov":
        steps = [vocab.id2step[i] for i in prefix_ids if vocab.id2step[i] not in _SPECIAL]
        return markov.complete_sequence(steps, max_new=max_new)

    if model_type == "lstm_attn":
        from models.lstm_attention import complete_sequence as attn_complete
        return attn_complete(model, prefix_ids, vocab, max_new=max_new,
                             device=str(device))

    generated, ids, hidden = [], list(prefix_ids), None

    for _ in range(max_new):
        if model_type == "lstm":
            if hidden is None:
                x = torch.tensor([ids], dtype=torch.long, device=device)
                logits, hidden = model(x)
            else:
                x = torch.tensor([[ids[-1]]], dtype=torch.long, device=device)
                logits, hidden = model(x, hidden)
            next_id = logits[0, -1].argmax().item()
        else:
            x = torch.tensor([ids], dtype=torch.long, device=device)
            # TCN returns (B,T,V) directly; GPT returns object with .logits
            raw = model(x)
            logits_last = raw[0, -1] if isinstance(raw, torch.Tensor) else raw.logits[0, -1]
            next_id = logits_last.argmax().item()

        if next_id == vocab.eos_id:
            break
        step = vocab.id2step[next_id]
        if step in _SPECIAL:
            break
        if len(generated) >= 3 and step == generated[-1] == generated[-2]:
            break
        generated.append(step)
        ids.append(next_id)

    return generated


# ── Task 2: Beam search completion ───────────────────────────────────────────

@torch.no_grad()
def complete_beam(model, model_type, vocab, prefix_ids, device,
                  beam_width=5, max_new=160, markov=None) -> list[str]:
    """
    Beam search sequence completion.
    Returns the highest-scoring complete sequence (suffix only).
    """
    if model_type == "markov":
        # Markov doesn't benefit from beam search — use greedy directly
        steps = [vocab.id2step[i] for i in prefix_ids if vocab.id2step[i] not in _SPECIAL]
        return markov.complete_sequence(steps, max_new=max_new)

    # Each beam: (log_prob, token_ids, hidden_state)
    # Seed beams from prefix
    x = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    if model_type in ("lstm", "lstm_attn"):
        logits, hidden = model(x)
        log_probs = torch.log_softmax(logits[0, -1], dim=-1)
    else:
        log_probs = torch.log_softmax(model(x).logits[0, -1], dim=-1)
        hidden = None

    top_lp, top_ids = log_probs.topk(beam_width)

    # Initialize beams
    beams = []
    for i in range(beam_width):
        nid = top_ids[i].item()
        step = vocab.id2step[nid]
        if step in _SPECIAL:
            continue
        h = (hidden[0][:, 0:1, :].clone(), hidden[1][:, 0:1, :].clone()) \
            if model_type in ("lstm", "lstm_attn") and hidden is not None else None
        beams.append({
            "lp":     top_lp[i].item(),
            "tokens": [nid],
            "hidden": h,
            "done":   nid == vocab.eos_id,
        })

    completed = []
    for _ in range(max_new):
        if not beams:
            break

        candidates = []
        for beam in beams:
            if beam["done"] or len(beam["tokens"]) >= max_new:
                completed.append(beam)
                continue

            last_id = beam["tokens"][-1]
            inp = torch.tensor([[last_id]], dtype=torch.long, device=device)

            if model_type in ("lstm", "lstm_attn"):
                logits, new_h = model(inp, beam["hidden"])
                raw_lp = logits[0, -1]
            else:
                ids_so_far = prefix_ids + beam["tokens"]
                x2 = torch.tensor([ids_so_far], dtype=torch.long, device=device)
                raw_lp = model(x2).logits[0, -1]
                new_h = None

            # Apply constraint mask during beam search
            current_steps = [vocab.id2step[i] for i in prefix_ids + beam["tokens"]
                             if vocab.id2step[i] not in _SPECIAL]
            from eval.constraint_decoder import build_constraint_mask
            c_mask = build_constraint_mask(current_steps, vocab, device=device)
            lp = torch.log_softmax(raw_lp + c_mask, dim=-1)

            top_lp2, top_ids2 = lp.topk(beam_width)
            for j in range(beam_width):
                nid2 = top_ids2[j].item()
                step2 = vocab.id2step[nid2]
                new_lp = beam["lp"] + top_lp2[j].item()

                # Length normalization
                new_len = len(beam["tokens"]) + 1
                normed_lp = new_lp / new_len

                h2 = (new_h[0][:, 0:1, :].clone(), new_h[1][:, 0:1, :].clone()) \
                     if model_type in ("lstm", "lstm_attn") and new_h is not None else None

                candidates.append({
                    "lp":     new_lp,
                    "norm_lp": normed_lp,
                    "tokens": beam["tokens"] + [nid2],
                    "hidden": h2,
                    "done":   nid2 == vocab.eos_id or step2 in _SPECIAL,
                })

        # Keep top beam_width by normalized log-prob
        candidates.sort(key=lambda b: b["norm_lp"], reverse=True)
        beams = []
        for c in candidates[:beam_width]:
            if c["done"]:
                completed.append(c)
            else:
                # Repetition guard
                tokens = c["tokens"]
                if (len(tokens) >= 3 and tokens[-1] == tokens[-2] == tokens[-3]):
                    continue
                beams.append(c)

        if len(completed) >= beam_width:
            break

    # Return the best completed beam
    if not completed:
        completed = beams
    if not completed:
        return []

    best = max(completed, key=lambda b: b["lp"] / max(len(b["tokens"]), 1))
    result = []
    for tid in best["tokens"]:
        step = vocab.id2step[tid]
        if step in _SPECIAL or step == vocab.id2step[vocab.eos_id]:
            break
        result.append(step)
    return result


# ── Task 3: Anomaly detection ─────────────────────────────────────────────────

@torch.no_grad()
def get_anomaly(model, model_type, vocab, seq_ids, device,
                valid_nll: float, markov=None,
                rule_override=None) -> tuple[int, float, str]:
    """Returns (is_valid, score 0-1, predicted_rule).
    rule_override: pre-computed (is_valid_rule, rule) using original step names.
    """
    steps = [vocab.id2step[i] for i in seq_ids if vocab.id2step[i] not in _SPECIAL]

    # Rule-based check — use pre-computed result if provided (uses original names)
    # Otherwise fall back to canonical (may miss name-sensitive violations)
    if rule_override is not None:
        is_valid_rule, rule = rule_override
    else:
        is_valid_rule, rule = rule_based_anomaly(steps)

    # Model NLL score — calibrated against known valid NLL
    if model_type == "markov":
        import math
        lp = markov.sequence_log_prob(steps)
        raw = max(0.0, min(1.0, 1.0 / (1.0 + math.exp(5 * (lp + 2.0)))))
    else:
        x = torch.tensor([seq_ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([seq_ids[1:]],  dtype=torch.long, device=device)
        if model_type in ("lstm", "lstm_attn", "ensemble", "dual_ensemble"):
            logits, _ = model(x)
        elif model_type == "tcn":
            logits = model(x)          # TCN: (B, T, V) directly
        else:
            logits = model(x).logits
        nll = nn.CrossEntropyLoss()(
            logits.view(-1, logits.size(-1)), y.view(-1)).item()
        # Calibrated: valid sequences cluster around valid_nll
        # anomalous sequences have significantly higher NLL
        raw = max(0.0, min(1.0, (nll - valid_nll) / (valid_nll * 1.5) + 0.1))

    # If rule validator says invalid, force score high
    if is_valid_rule == 0:
        score = max(raw, 0.85)
        return 0, round(score, 4), rule

    is_valid = 0 if raw >= 0.5 else 1
    return is_valid, round(raw, 4), ""


# ── Dual-model ensemble helpers ───────────────────────────────────────────────

def build_orig_vocab():
    """Vocabulary built from original (non-canonical) step names (206 tokens)."""
    from collections import defaultdict
    all_seqs = []
    for fam, path in FAMILY_FILES.items():
        seqs = load_sequences(path, apply_canonical=False)
        for seq in seqs.values():
            all_seqs.append(seq)
    vocab = Vocabulary()
    vocab.build(all_seqs)
    return vocab


def build_family_variant_priors():
    """Per-family most-frequent original variant for each canonical step."""
    from collections import Counter
    priors: dict[str, dict[str, str]] = {}
    for fam, path in FAMILY_FILES.items():
        seqs = load_sequences(path, apply_canonical=False)
        cnt: dict[str, Counter] = {}
        for seq in seqs.values():
            for step in seq:
                canon = CANONICAL_STEPS.get(step, step)
                cnt.setdefault(canon, Counter())[step] += 1
        priors[fam] = {canon: counter.most_common(1)[0][0]
                       for canon, counter in cnt.items()}
    return priors


@torch.no_grad()
def predict_next_top5_dual(canonical_model, original_model,
                            vocab_canon, vocab_orig,
                            prefix_ids_canon, prefix_ids_orig,
                            device, family, family_priors=None,
                            w_canon=0.6, w_orig=0.4) -> list[str]:
    """
    Dual-model ensemble for Task 1.
    canonical_model picks the step *type*; original_model picks the *variant*.
    Combined score = w_canon × log_p_canon + w_orig × log_p_orig.
    Returns top-5 original step names.
    """
    xc = torch.tensor([prefix_ids_canon], dtype=torch.long, device=device)
    canon_lp = torch.log_softmax(canonical_model(xc)[0][0, -1], dim=-1)

    xo = torch.tensor([prefix_ids_orig],  dtype=torch.long, device=device)
    orig_lp  = torch.log_softmax(original_model(xo)[0][0, -1],  dim=-1)

    canon_top20 = canon_lp.topk(20)
    scored: list[tuple[float, str]] = []

    for tok_idx, canon_tok_id in enumerate(canon_top20.indices.tolist()):
        canon_step = vocab_canon.id2step.get(canon_tok_id, "")
        if not canon_step or canon_step in _SPECIAL:
            continue
        c_score = canon_top20.values[tok_idx].item()

        variants = VARIANTS_OF.get(canon_step, [canon_step])
        for var in variants:
            var_id = vocab_orig.step2id.get(var)
            if var_id is None:
                continue
            combined = w_canon * c_score + w_orig * orig_lp[var_id].item()
            scored.append((combined, var))

    scored.sort(key=lambda x: x[0], reverse=True)
    seen, top5 = set(), []
    for _, var in scored:
        if var not in seen:
            seen.add(var)
            top5.append(var)
            if len(top5) == 5:
                break

    if family_priors and len(top5) < 5:
        for _, pref_var in family_priors.get(family, {}).items():
            if pref_var not in seen:
                seen.add(pref_var)
                top5.append(pref_var)
                if len(top5) == 5:
                    break

    return top5 if top5 else [""]


# ── Self-eval: build mock eval set from our val data ──────────────────────────

def build_self_eval(vocab, use_original_names=False):
    """
    Build the 300-sequence self-eval set.

    use_original_names=False (default): decode step names from the canonical
        vocabulary → GT uses canonical names (e.g. "STRIP RESIST").
        Correct for canonical-output models.

    use_original_names=True: load sequences directly from CSV files without
        canonicalization → GT uses original names (e.g. "STRIP PHOTORESIST").
        Required for models that output original variant names (dual_ensemble).
        Uses the same val-split indices as the canonical path so the 300
        held-out sequences are identical; only the name strings differ.
    """
    from eval.anomaly_generator import generate_anomaly_set

    # Always build dataset with canonical vocab to get stable val_ds.indices.
    dataset = SequenceDataset(vocab)
    _, val_ds = train_val_split(dataset)

    # Build a parallel list of (family, original_steps) if needed.
    # load_sequences preserves insertion order (same CSV row order), so
    # all_orig_seqs[k] corresponds to dataset.samples[k] for all k.
    if use_original_names:
        all_orig_seqs: list[tuple[str, list[str]]] = []
        for fam in ("mosfet", "igbt", "ic"):
            orig_seqs = load_sequences(FAMILY_FILES[fam], apply_canonical=False)
            for seq in orig_seqs.values():
                all_orig_seqs.append((fam, seq))

    def _get_steps(idx: int):
        if use_original_names:
            return all_orig_seqs[idx]          # (family, original_steps)
        raw_ids = dataset.samples[idx]
        steps = [vocab.id2step[j] for j in raw_ids if vocab.id2step[j] not in _SPECIAL]
        family = vocab.id2step[raw_ids[1]].strip("[]").lower()
        return family, steps

    valid_rows, valid_seqs_for_anomaly = [], []

    for i, idx in enumerate(val_ds.indices[:300]):
        family, steps = _get_steps(idx)

        for frac in (0.6, 0.8):
            cut = max(1, int(len(steps) * frac))
            valid_rows.append({
                "EXAMPLE_ID":          f"val_{i:04d}_f{int(frac*100)}",
                "FAMILY":              family,
                "COMPLETION_FRACTION": frac,
                "PARTIAL_SEQUENCE":    "|".join(steps[:cut]),
                "_ACTUAL_NEXT_STEP":   steps[cut] if cut < len(steps) else "",
                "_REMAINING_STEPS":    "|".join(steps[cut:]),
            })
        valid_seqs_for_anomaly.append((f"src_{i:04d}", family, steps))

    # --- Task 3: balanced valid + anomalous sequences ---
    anomaly_rows = []
    for i, idx in enumerate(val_ds.indices[:300]):
        family, steps = _get_steps(idx)
        anomaly_rows.append({
            "EXAMPLE_ID": f"valid_{i:04d}",
            "FAMILY":     family,
            "SEQUENCE":   "|".join(steps),
            "_IS_VALID":  1,
        })

    # Generate anomalies using all 10 rule injectors + validator verification
    print("Generating anomalies across all 10 rule types...")
    validator = _get_validator()
    bad_rows = generate_anomaly_set(
        valid_seqs_for_anomaly,
        validator_fn=validator.validate_sequence,
        n_per_rule=30,
        seed=42,
    )
    for r in bad_rows:
        anomaly_rows.append({
            "EXAMPLE_ID": r["EXAMPLE_ID"],
            "FAMILY":     r["FAMILY"],
            "SEQUENCE":   r["SEQUENCE"],
            "_IS_VALID":  0,
            "_RULE":      r["VIOLATED_RULE"],
        })

    print(f"Anomaly set: {sum(1 for r in anomaly_rows if r['_IS_VALID']==1)} valid, "
          f"{sum(1 for r in anomaly_rows if r['_IS_VALID']==0)} anomalous")
    return valid_rows, anomaly_rows


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    global FAMILY_ENSEMBLE_WEIGHTS
    if (hasattr(args, "w_lstm") and args.w_lstm is not None) or \
       (hasattr(args, "w_gpt") and args.w_gpt is not None) or \
       (hasattr(args, "w_markov") and args.w_markov is not None):
        w_lstm = args.w_lstm if args.w_lstm is not None else 0.45
        w_gpt = args.w_gpt if args.w_gpt is not None else 0.35
        w_markov = args.w_markov if args.w_markov is not None else 0.20
        print(f"Dynamically overriding FAMILY_ENSEMBLE_WEIGHTS with: lstm={w_lstm}, gpt={w_gpt}, markov={w_markov}")
        for fam in FAMILY_ENSEMBLE_WEIGHTS:
            FAMILY_ENSEMBLE_WEIGHTS[fam] = {
                "w_lstm_attn": w_lstm,
                "w_gpt": w_gpt,
                "w_markov": w_markov
            }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vocab = build_vocab()

    model, valid_nll, markov, tcn_model = None, 0.33, None, None
    markov_orig = None  # original-vocab Markov for decanonicalization tie-breaker
    beam_width = args.beam_width if hasattr(args, 'beam_width') else 1
    use_constraints = not args.no_constraints if hasattr(args, 'no_constraints') else True

    # Checkpoint selection
    if hasattr(args, 'ckpt') and args.ckpt:
        lstm_ckpt = args.ckpt
    elif (CKPT_DIR / "lstm_canonical_best.pt").exists():
        lstm_ckpt = "lstm_canonical_best.pt"   # canonical retrain takes priority
    elif (CKPT_DIR / "lstm_30k_best.pt").exists():
        lstm_ckpt = "lstm_30k_best.pt"
    else:
        lstm_ckpt = "lstm_best.pt"

    gpt_model = None
    lstm_attn_model = None   # attention LSTM for context-aware ensemble
    orig_model = None
    vocab_orig = None
    family_priors = None
    if args.model == "lstm":
        model, valid_nll = load_lstm(vocab, device, ckpt_name=lstm_ckpt)
        model_type = "lstm"
    if args.model == "lstm_attn":
        model, valid_nll = load_lstm_attn(vocab, device)
        model_type = "lstm_attn"
    if args.model == "dual_ensemble":
        model, valid_nll = load_lstm(vocab, device, ckpt_name=lstm_ckpt)
        model_type = "dual_ensemble"
        vocab_orig = build_orig_vocab()
        orig_ckpt = ("lstm_30k_best.pt" if (CKPT_DIR / "lstm_30k_best.pt").exists()
                     else "lstm_best.pt")
        orig_model, _ = load_lstm(vocab_orig, device, ckpt_name=orig_ckpt)
        print(f"Dual ensemble: canonical({len(vocab)} tokens) + original({len(vocab_orig)} tokens)")
        family_priors = build_family_variant_priors()
        print(f"  ✓ Priors built for {len(family_priors)} families")
    if args.model == "gpt":
        model, valid_nll = load_gpt(vocab, device)
        model_type = "gpt"
    if args.model == "tcn":
        model, valid_nll = load_tcn(vocab, device)
        model_type = "tcn"
    if args.model in ("markov", "ensemble"):
        # Use canonical Markov for ensemble if available (matches canonical LSTM vocab)
        markov_ckpt = ("markov_canonical.pkl"
                       if (CKPT_DIR / "markov_canonical.pkl").exists()
                       else "markov_order3.pkl")
        markov = load_markov(markov_ckpt)
        if args.model == "markov":
            model_type = "markov"
    if args.model == "ensemble":
        # Context-aware ensemble: LSTM-Attn (long-range) + GPT (local) + Markov (prior)
        lstm_attn_model, valid_nll = load_lstm_attn(vocab, device)
        model = lstm_attn_model   # also used for Task 2/3
        try:
            gpt_model, _ = load_gpt(vocab, device)
        except RuntimeError as e:
            print(f"⚠ GPT skipped: {e}")
            gpt_model = None
        model_type = "lstm_attn"   # Task 2/3 use lstm_attn (KV-cache completion)
        print(f"Running context-aware ensemble: LSTM-Attn + GPT + Markov")
        print(f"  Family weights: {FAMILY_ENSEMBLE_WEIGHTS}")
        print(f"  GPT: {'loaded' if gpt_model else 'skipped'}")

    # Load original-vocabulary Markov for decanonicalization tie-breaker.
    # markov_orig_names.pkl was trained with apply_canonical=False → vocab=198 (original names).
    # Fall back to markov_order3.pkl only if orig_names version not available.
    orig_ckpt = ("markov_orig_names.pkl" if (CKPT_DIR / "markov_orig_names.pkl").exists()
                 else "markov_order3.pkl")
    markov_orig = load_markov(orig_ckpt)

    # Eval inputs
    if args.self_eval:
        print("Building self-eval set...")
        # dual_ensemble outputs original step names → GT must also use original names
        use_orig_gt = (model_type == "dual_ensemble")
        if use_orig_gt:
            print("  → using original step names for GT (dual_ensemble outputs variants)")
        valid_rows, anomaly_rows = build_self_eval(vocab, use_original_names=use_orig_gt)
    else:
        with open(args.valid_input, newline="") as f:
            valid_rows = list(csv.DictReader(f))
        with open(args.anomaly_input, newline="") as f:
            anomaly_rows = list(csv.DictReader(f))

    tag = f"_{args.model}" + ("_selfeval" if args.self_eval else "")

    # ── Tasks 1 & 2 ───────────────────────────────────────────────────────────
    nextstep_rows, completion_rows = [], []
    print(f"\nTasks 1 & 2: {len(valid_rows)} sequences...")

    for row in valid_rows:
        eid    = row["EXAMPLE_ID"]
        family = row["FAMILY"].lower()
        steps  = parse_pipe_sequence(row["PARTIAL_SEQUENCE"])
        prefix = encode_prefix(vocab, steps, family)

        # Perplexity routing for unknown/4th family: pick best conditioning token
        if family not in ("mosfet", "igbt", "ic") and model is not None:
            family = route_family(model, model_type, vocab, steps, device)
            prefix = encode_prefix(vocab, steps, family)

        # Task 1: dual ensemble, context-aware ensemble, or single model
        if model_type == "dual_ensemble":
            prefix_orig = encode_prefix(vocab_orig, steps, family, apply_canonical=False)
            top5 = predict_next_top5_dual(
                model, orig_model, vocab, vocab_orig,
                prefix, prefix_orig, device, family,
                family_priors=family_priors)
        elif model_type == "lstm_attn" and lstm_attn_model is not None and markov is not None:
            # Context-aware ensemble: per-family weights derived from val-set NLL
            top5 = predict_next_top5_context_aware(
                lstm_attn_model, gpt_model, markov, vocab, prefix, device,
                family=family, original_prefix=steps, markov_orig=markov_orig)
        else:
            top5 = predict_next_top5(model, model_type, vocab, prefix, device,
                                     markov, use_constraints=use_constraints,
                                     family=family, original_prefix=steps,
                                     markov_orig=markov_orig)
        while len(top5) < 5:
            top5.append("")

        nextstep_rows.append({
            "EXAMPLE_ID": eid,
            "RANK_1": top5[0], "RANK_2": top5[1], "RANK_3": top5[2],
            "RANK_4": top5[3], "RANK_5": top5[4],
        })

        # Task 2: beam search or greedy
        # lstm_attn uses KV-cache completion (full-history attention per step)
        eff_model_type = "lstm" if model_type == "dual_ensemble" else model_type
        if beam_width > 1:
            sfx = complete_beam(model, eff_model_type, vocab, prefix, device,
                                beam_width=beam_width)
        else:
            sfx = complete(model, eff_model_type, vocab, prefix, device, markov=markov)

        # Decanonicalize completion output — model generates canonical names but
        # official GT uses original names (e.g. "STRIP PHOTORESIST" not "STRIP RESIST").
        # Build running prefix (original steps + decanonicalised suffix so far) for Markov.
        if markov_orig is not None:
            decan_sfx, running_prefix = [], list(steps)
            for s in sfx:
                orig_s = decanonicalize(s, running_prefix, markov_orig)
                decan_sfx.append(orig_s)
                running_prefix.append(orig_s)
            sfx = decan_sfx

        completion_rows.append({
            "EXAMPLE_ID":        eid,
            "PREDICTED_SEQUENCE": "|".join(sfx),
        })

    # ── Task 3 ────────────────────────────────────────────────────────────────
    anomaly_out = []
    print(f"Task 3: {len(anomaly_rows)} sequences...")

    for row in anomaly_rows:
        eid    = row["EXAMPLE_ID"]
        family = row["FAMILY"].lower()
        steps  = parse_pipe_sequence(row["SEQUENCE"])  # original names — for rule validator
        seq_ids = encode_prefix(vocab, steps, family) + [vocab.eos_id]

        # Rule check uses ORIGINAL step names (validator has hardcoded step name checks)
        is_valid_rule, rule = rule_based_anomaly(steps)

        is_valid, score, rule = get_anomaly(
            model, model_type, vocab, seq_ids, device, valid_nll, markov,
            rule_override=(is_valid_rule, rule))
        anomaly_out.append({
            "EXAMPLE_ID":    eid,
            "IS_VALID":      is_valid,
            "SCORE":         score,
            "PREDICTED_RULE": rule,
        })

    # ── Write submission CSVs ─────────────────────────────────────────────────
    ns_path = RESULTS / f"nextstep{tag}.csv"
    with open(ns_path, "w", newline="") as f:
        w = csv.DictWriter(f, ["EXAMPLE_ID","RANK_1","RANK_2","RANK_3","RANK_4","RANK_5"])
        w.writeheader(); w.writerows(nextstep_rows)

    cp_path = RESULTS / f"completion{tag}.csv"
    with open(cp_path, "w", newline="") as f:
        w = csv.DictWriter(f, ["EXAMPLE_ID","PREDICTED_SEQUENCE"])
        w.writeheader(); w.writerows(completion_rows)

    an_path = RESULTS / f"anomaly{tag}.csv"
    with open(an_path, "w", newline="") as f:
        w = csv.DictWriter(f, ["EXAMPLE_ID","IS_VALID","SCORE","PREDICTED_RULE"])
        w.writeheader(); w.writerows(anomaly_out)

    print(f"\n✓ {ns_path.name}")
    print(f"✓ {cp_path.name}")
    print(f"✓ {an_path.name}")

    # ── Self-eval via Yaroslav's script ───────────────────────────────────────
    if args.self_eval:
        print("\n=== Self-Evaluation (Yaroslav's eval_metrics.py) ===")
        eval_script = ROOT / "src" / "eval" / "eval_metrics.py"

        # Ground truth CSVs (matching eval_metrics.py column names)
        gt_ns = RESULTS / "gt_nextstep.csv"
        with open(gt_ns, "w", newline="") as f:
            w = csv.DictWriter(f, ["SEQUENCE_ID", "ACTUAL_NEXT_STEP",
                                   "FAMILY", "COMPLETION_FRACTION"])
            w.writeheader()
            for r in valid_rows:
                w.writerow({"SEQUENCE_ID": r["EXAMPLE_ID"],
                            "ACTUAL_NEXT_STEP": r.get("_ACTUAL_NEXT_STEP", ""),
                            "FAMILY": r.get("FAMILY", ""),
                            "COMPLETION_FRACTION": r.get("COMPLETION_FRACTION", "")})

        gt_cp = RESULTS / "gt_completion.csv"
        with open(gt_cp, "w", newline="") as f:
            w = csv.DictWriter(f, ["SEQUENCE_ID", "REMAINING_STEPS",
                                   "FAMILY", "COMPLETION_FRACTION"])
            w.writeheader()
            for r in valid_rows:
                w.writerow({"SEQUENCE_ID": r["EXAMPLE_ID"],
                            "REMAINING_STEPS": r.get("_REMAINING_STEPS", ""),
                            "FAMILY": r.get("FAMILY", ""),
                            "COMPLETION_FRACTION": r.get("COMPLETION_FRACTION", "")})

        gt_an = RESULTS / "gt_anomaly.csv"
        with open(gt_an, "w", newline="") as f:
            w = csv.DictWriter(f, ["SEQUENCE_ID", "IS_ANOMALY",
                                   "VIOLATED_RULE", "FAMILY"])
            w.writeheader()
            for r in anomaly_rows:
                is_anomaly = 1 - int(r.get("_IS_VALID", 1))
                w.writerow({"SEQUENCE_ID":  r["EXAMPLE_ID"],
                            "IS_ANOMALY":   is_anomaly,
                            "VIOLATED_RULE": r.get("_RULE", ""),
                            "FAMILY":       r.get("FAMILY", "")})

        # Prediction CSVs renamed for eval_metrics.py
        pred_ns = RESULTS / f"pred_nextstep{tag}.csv"
        with open(ns_path) as fi, open(pred_ns, "w", newline="") as fo:
            rows = list(csv.DictReader(fi))
            w = csv.DictWriter(fo, ["SEQUENCE_ID","PRED_1","PRED_2","PRED_3","PRED_4","PRED_5"])
            w.writeheader()
            for r in rows:
                w.writerow({"SEQUENCE_ID": r["EXAMPLE_ID"], "PRED_1": r["RANK_1"],
                            "PRED_2": r["RANK_2"], "PRED_3": r["RANK_3"],
                            "PRED_4": r["RANK_4"], "PRED_5": r["RANK_5"]})

        pred_cp = RESULTS / f"pred_completion{tag}.csv"
        with open(cp_path) as fi, open(pred_cp, "w", newline="") as fo:
            rows = list(csv.DictReader(fi))
            w = csv.DictWriter(fo, ["SEQUENCE_ID","PREDICTED_STEPS"])
            w.writeheader()
            for r in rows:
                w.writerow({"SEQUENCE_ID": r["EXAMPLE_ID"],
                            "PREDICTED_STEPS": r["PREDICTED_SEQUENCE"]})

        pred_an = RESULTS / f"pred_anomaly{tag}.csv"
        with open(an_path) as fi, open(pred_an, "w", newline="") as fo:
            rows = list(csv.DictReader(fi))
            w = csv.DictWriter(fo, ["SEQUENCE_ID","ANOMALY_SCORE","PREDICTED_RULE"])
            w.writeheader()
            for r in rows:
                w.writerow({"SEQUENCE_ID": r["EXAMPLE_ID"],
                            "ANOMALY_SCORE": r["SCORE"],
                            "PREDICTED_RULE": r["PREDICTED_RULE"]})

        for task, gt, pred in [
            ("next-step",  gt_ns, pred_ns),
            ("completion", gt_cp, pred_cp),
            ("anomaly",    gt_an, pred_an),
        ]:
            res = subprocess.run(
                ["python", str(eval_script),
                 "--task", task, "--ground-truth", str(gt), "--predictions", str(pred)],
                capture_output=True, text=True)
            print(res.stdout)
            if res.returncode != 0:
                print("ERR:", res.stderr[:300])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm","lstm_attn","gpt","markov","tcn","ensemble","dual_ensemble"], default="lstm")
    parser.add_argument("--self-eval", action="store_true")
    parser.add_argument("--beam-width", type=int, default=1,
                        help="Beam width for Task 2 completion (1=greedy, 5=recommended)")
    parser.add_argument("--no-constraints", action="store_true",
                        help="Disable constraint masking for Task 1 next-step prediction")
    parser.add_argument("--no-synonyms", action="store_true",
                        help="Disable synonym probability merging")
    parser.add_argument("--ckpt", default="",
                        help="Explicit checkpoint filename e.g. lstm_30k_best.pt")
    parser.add_argument("--ckpt-suffix", default="",
                        help="Load checkpoint with suffix e.g. '_30k'")
    parser.add_argument("--valid-input",   default="data/eval_input_valid.csv")
    parser.add_argument("--anomaly-input", default="data/eval_input_anomaly.csv")
    parser.add_argument("--w-lstm", type=float, default=None, help="LSTM weight in ensemble")
    parser.add_argument("--w-gpt", type=float, default=None, help="GPT weight in ensemble")
    parser.add_argument("--w-markov", type=float, default=None, help="Markov weight in ensemble")
    run(parser.parse_args())
