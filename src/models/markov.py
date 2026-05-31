"""
Markov Chain baseline — Yaroslav
N-gram model for next-step prediction, completion, and anomaly scoring.
No GPU needed — trains in seconds.
"""

import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


class MarkovModel:
    def __init__(self, order: int = 3):
        """
        order=1 → bigram (current step predicts next)
        order=2 → trigram
        order=3 → 4-gram (recommended)
        """
        self.order = order
        self.counts: Dict[tuple, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.vocab: set = set()

    def train(self, sequences: List[List[str]]) -> None:
        """Train on list of step sequences."""
        for seq in sequences:
            self.vocab.update(seq)
            padded = ["[BOS]"] * self.order + seq + ["[EOS]"]
            for i in range(len(padded) - self.order):
                context = tuple(padded[i:i + self.order])
                next_step = padded[i + self.order]
                self.counts[context][next_step] += 1
        print(f"Markov trained: order={self.order}, "
              f"vocab={len(self.vocab)}, contexts={len(self.counts)}")

    def _get_context(self, prefix: List[str]) -> tuple:
        padded = ["[BOS]"] * self.order + prefix
        return tuple(padded[-self.order:])

    def predict_next_top_k(self, prefix: List[str], k: int = 5) -> List[str]:
        """Return top-k most likely next steps given a prefix."""
        context = self._get_context(prefix)

        for length in range(self.order, 0, -1):
            ctx = context[-length:]
            if ctx in self.counts and self.counts[ctx]:
                ranked = sorted(self.counts[ctx].items(),
                                key=lambda x: x[1], reverse=True)
                return [step for step, _ in ranked[:k]
                        if step not in ("[EOS]", "[BOS]")]

        # Unigram fallback
        totals: Dict[str, int] = defaultdict(int)
        for dist in self.counts.values():
            for step, cnt in dist.items():
                totals[step] += cnt
        ranked = sorted(totals.items(), key=lambda x: x[1], reverse=True)
        return [step for step, _ in ranked[:k] if step not in ("[EOS]", "[BOS]")]

    def complete_sequence(self, prefix: List[str],
                          max_new: int = 160) -> List[str]:
        """Greedily complete a sequence from a prefix."""
        generated = list(prefix)
        for _ in range(max_new):
            top = self.predict_next_top_k(generated, k=1)
            if not top or top[0] == "[EOS]":
                break
            next_step = top[0]
            # Repetition guard — break if last 3 steps are identical
            if len(generated) >= 3 and next_step == generated[-1] == generated[-2]:
                break
            generated.append(next_step)
        return generated[len(prefix):]

    def sequence_log_prob(self, sequence: List[str]) -> float:
        """
        Mean log-probability per step.
        Lower (more negative) = less likely = more anomalous.
        Uses Laplace (+1) smoothing to avoid zero probabilities.
        """
        padded = ["[BOS]"] * self.order + sequence + ["[EOS]"]
        log_prob = 0.0
        n = 0
        vocab_size = len(self.vocab) + 1  # +1 for [EOS]

        for i in range(len(padded) - self.order):
            context = tuple(padded[i:i + self.order])
            next_step = padded[i + self.order]

            prob = 0.0
            for length in range(self.order, 0, -1):
                ctx = context[-length:]
                if ctx in self.counts and self.counts[ctx]:
                    total = sum(self.counts[ctx].values())
                    # Laplace smoothing
                    prob = (self.counts[ctx].get(next_step, 0) + 1) / (total + vocab_size)
                    break

            if prob == 0.0:
                prob = 1.0 / vocab_size  # uniform fallback

            log_prob += math.log(prob)
            n += 1

        return log_prob / max(n, 1)

    def anomaly_score(self, sequence: List[str]) -> float:
        """Score in [0, 1] — higher = more anomalous."""
        lp = self.sequence_log_prob(sequence)
        score = 1.0 / (1.0 + math.exp(5 * (lp + 2.0)))
        return float(score)

    def save(self, path: Path) -> None:
        data = {
            "order": self.order,
            "vocab": self.vocab,
            "counts": {k: dict(v) for k, v in self.counts.items()},
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load(path: Path) -> "MarkovModel":
        with open(path, "rb") as f:
            data = pickle.load(f)
        m = MarkovModel(order=data["order"])
        m.vocab = data["vocab"]
        m.counts = defaultdict(lambda: defaultdict(int))
        for k, v in data["counts"].items():
            m.counts[k] = defaultdict(int, v)
        return m


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data import load_all_sequences

    print("Loading sequences...")
    seqs, labels = load_all_sequences()
    print(f"Loaded {len(seqs)} sequences")

    model = MarkovModel(order=3)
    model.train(seqs)

    prefix = ["RECEIVE WAFER LOT", "LOT IDENTIFICATION", "INITIAL WAFER INSPECTION"]
    top5 = model.predict_next_top_k(prefix, k=5)
    print(f"\nPrefix: {prefix[-1]} → ?")
    print(f"Top-5: {top5}")

    completion = model.complete_sequence(prefix, max_new=10)
    print(f"\nCompletion (first 10 steps): {completion[:10]}")

    score = model.anomaly_score(seqs[0])
    print(f"\nAnomaly score (valid seq): {score:.4f}")

    model.save(Path("checkpoints/markov_order3.pkl"))
    print("\nMarkov model saved!")
