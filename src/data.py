"""
Shared data utilities: vocabulary, dataset loader, tokenizer.
All three models (LSTM, Markov, GPT) import from here.
"""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

# ── Special tokens ──────────────────────────────────────────────────────────
PAD_TOKEN   = "[PAD]"
UNK_TOKEN   = "[UNK]"
BOS_TOKEN   = "[BOS]"
EOS_TOKEN   = "[EOS]"
CLS_TOKEN   = "[CLS]"
MOSFET_TOK  = "[MOSFET]"
IGBT_TOK    = "[IGBT]"
IC_TOK      = "[IC]"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN,
                  CLS_TOKEN, MOSFET_TOK, IGBT_TOK, IC_TOK]

FAMILY_TOKEN = {"mosfet": MOSFET_TOK, "igbt": IGBT_TOK, "ic": IC_TOK}

# ── Vocabulary consolidation ──────────────────────────────────────────────────
# Derived from error analysis: 100% of prediction failures were caused by the
# model splitting probability mass between these synonym pairs.
# Maps each variant → its canonical (preferred) name.
# Applied at load time so the model trains on a single consistent signal.
CANONICAL_STEPS: Dict[str, str] = {
    # STRIP (34 confusions — top confusion pair)
    "STRIP PHOTORESIST":              "STRIP RESIST",
    "STRIP RESIST LEVEL 2":           "STRIP RESIST",
    # PASSIVATION ETCH (22 confusions)
    "PASSIVATION ETCH PAD OPENING":   "PASSIVATION ETCH",
    # MEASURE PLANARITY (20 confusions)
    "MEASURE SURFACE PLANARITY":      "MEASURE PLANARITY",
    # PAD WINDOW (15 confusions)
    "OPEN BOND PAD WINDOW":           "OPEN PAD WINDOW",
    # CMP (12 confusions)
    "CMP INTERLAYER DIELECTRIC":      "CMP DIELECTRIC",
    # LITHO WINDOW (12 confusions)
    "OPEN PAD WINDOW LITHO":          "PAD WINDOW LITHO",
    # VIA ETCH (10 confusions — includes 3 surface forms)
    "VIA ETCH THROUGH DIELECTRIC":    "VIA ETCH",
    "DIELECTRIC ETCH VIA":            "VIA ETCH",
    # MEASURE THICKNESS (5 confusions)
    "MEASURE FILM THICKNESS":         "MEASURE DIELECTRIC THICKNESS",
}

# Reverse map: canonical → [canonical, variant1, variant2, ...]
# Used at inference to enumerate which original names to ask Markov about.
VARIANTS_OF: Dict[str, List[str]] = {}
for _variant, _canonical in CANONICAL_STEPS.items():
    if _canonical not in VARIANTS_OF:
        VARIANTS_OF[_canonical] = [_canonical]   # canonical is always a valid output
    VARIANTS_OF[_canonical].append(_variant)


def canonicalize(step: str) -> str:
    """Return the canonical form of a step name, or the step itself if not a variant."""
    return CANONICAL_STEPS.get(step, step)


DATA_DIR = Path(__file__).parent.parent / "data"

FAMILY_FILES = {
    "mosfet": DATA_DIR / "MOSFET_variants.csv",
    "igbt":   DATA_DIR / "IGBT_variants.csv",
    "ic":     DATA_DIR / "IC_variants.csv",
}


# ── Vocabulary ───────────────────────────────────────────────────────────────
class Vocabulary:
    def __init__(self):
        self.step2id: Dict[str, int] = {}
        self.id2step: Dict[int, str] = {}

    def build(self, sequences: List[List[str]]) -> None:
        steps = sorted({step for seq in sequences for step in seq})
        for tok in SPECIAL_TOKENS:
            idx = len(self.step2id)
            self.step2id[tok] = idx
            self.id2step[idx] = tok
        for step in steps:
            if step not in self.step2id:
                idx = len(self.step2id)
                self.step2id[step] = idx
                self.id2step[idx] = step

    def encode(self, steps: List[str]) -> List[int]:
        unk = self.step2id[UNK_TOKEN]
        return [self.step2id.get(s, unk) for s in steps]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.id2step.get(i, UNK_TOKEN) for i in ids]

    def __len__(self) -> int:
        return len(self.step2id)

    @property
    def pad_id(self): return self.step2id[PAD_TOKEN]
    @property
    def bos_id(self): return self.step2id[BOS_TOKEN]
    @property
    def eos_id(self): return self.step2id[EOS_TOKEN]
    @property
    def unk_id(self): return self.step2id[UNK_TOKEN]
    @property
    def cls_id(self): return self.step2id[CLS_TOKEN]


# ── CSV loading ───────────────────────────────────────────────────────────────
def load_sequences(csv_path: Path, apply_canonical: bool = True) -> Dict[str, List[str]]:
    """
    Returns {sequence_id: [step, step, ...]}
    With apply_canonical=True (default), variant names are merged into their
    canonical forms so the model trains on a single consistent signal per step.
    """
    seqs: Dict[str, List[str]] = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = row["STEP"]
            if apply_canonical:
                step = CANONICAL_STEPS.get(step, step)
            seqs[row["SEQUENCE_ID"]].append(step)
    return dict(seqs)


def load_all_sequences(families=("mosfet", "igbt", "ic")) -> Tuple[List[List[str]], List[str]]:
    """Returns (sequences, family_labels)"""
    all_seqs, labels = [], []
    for fam in families:
        seqs = load_sequences(FAMILY_FILES[fam])
        for seq in seqs.values():
            all_seqs.append(seq)
            labels.append(fam)
    return all_seqs, labels


def build_vocab(families=("mosfet", "igbt", "ic")) -> Vocabulary:
    seqs, _ = load_all_sequences(families)
    vocab = Vocabulary()
    vocab.build(seqs)
    return vocab


# ── PyTorch Dataset ────────────────────────────────────────────────────────────
class SequenceDataset(Dataset):
    """
    Each item: full tokenized sequence with [BOS] [FAMILY] steps... [EOS]
    Returns (input_ids, target_ids) shifted by 1 for next-token prediction.
    """
    def __init__(self, vocab: Vocabulary, families=("mosfet", "igbt", "ic"),
                 max_len: int = 210):
        self.vocab = vocab
        self.max_len = max_len
        self.samples: List[List[int]] = []

        for fam in families:
            fam_tok = vocab.step2id[FAMILY_TOKEN[fam]]
            seqs = load_sequences(FAMILY_FILES[fam])
            for seq in seqs.values():
                ids = [vocab.bos_id, fam_tok] + vocab.encode(seq) + [vocab.eos_id]
                if len(ids) > max_len:
                    ids = ids[:max_len]
                self.samples.append(ids)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        ids = self.samples[idx]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:],  dtype=torch.long)
        return x, y


def collate_fn(batch, pad_id: int):
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    x_pad = torch.stack([torch.nn.functional.pad(x, (0, max_len - x.size(0)), value=pad_id) for x in xs])
    y_pad = torch.stack([torch.nn.functional.pad(y, (0, max_len - y.size(0)), value=-100) for y in ys])
    return x_pad, y_pad


def train_val_split(dataset: SequenceDataset, val_ratio=0.1, seed=42):
    n = len(dataset)
    n_val = int(n * val_ratio)
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(dataset, [n - n_val, n_val], generator=g)


# ── Inference helpers ──────────────────────────────────────────────────────────
def encode_prefix(vocab: Vocabulary, steps: List[str], family: str,
                  apply_canonical: bool = True) -> List[int]:
    """
    Encode a partial sequence for inference.
    apply_canonical=True canonicalises step names before encoding so test-time
    prefixes match the vocabulary the model was trained on.
    """
    fam_tok = vocab.step2id[FAMILY_TOKEN[family]]
    if apply_canonical:
        steps = [CANONICAL_STEPS.get(s, s) for s in steps]
    return [vocab.bos_id, fam_tok] + vocab.encode(steps)


def parse_pipe_sequence(pipe_str: str) -> List[str]:
    """Parse 'STEP_A|STEP_B|...' format from eval CSV."""
    return [s.strip() for s in pipe_str.split("|") if s.strip()]
