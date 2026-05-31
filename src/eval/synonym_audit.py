"""
Definitive synonym audit — resolve the 70.7% vs ~78% math gap.

Key question: is our CANONICAL form the MINORITY variant? If so, outputting the
canonical name systematically picks the less-frequent string on skewed groups,
costing free points. Also decomposes dual-ensemble accuracy and computes the
always-majority oracle ceiling.

Run on cluster:
  ~/.pixi/bin/pixi run python src/eval/synonym_audit.py
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
from data import (build_vocab, encode_prefix, parse_pipe_sequence,
                  CANONICAL_STEPS, VARIANTS_OF, FAMILY_FILES, load_sequences)
import eval.infer as I


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = build_vocab()

    gc = defaultdict(Counter)
    for fam, path in FAMILY_FILES.items():
        for seq in load_sequences(path, apply_canonical=False).values():
            for s in seq:
                gc[CANONICAL_STEPS.get(s, s)][s] += 1

    print("=== SYNONYM GROUP AUDIT ===")
    print(f"{'canonical form':<32}{'maj variant':<32}{'maj%':>6}  canon==maj?")
    majority = {}
    for canon, variants in VARIANTS_OF.items():
        cnt = gc.get(canon, Counter())
        tot = sum(cnt.values())
        if not tot:
            continue
        maj_var, maj_n = cnt.most_common(1)[0]
        majority[canon] = maj_var
        flag = "YES" if maj_var == canon else "NO  <-- canonical is MINORITY"
        print(f"{canon:<32}{maj_var:<32}{100*maj_n/tot:>5.0f}%  {flag}")

    model, _ = I.load_lstm(vocab, device, ckpt_name="lstm_canonical_best.pt")
    vocab_orig = I.build_orig_vocab()
    orig_model, _ = I.load_lstm(vocab_orig, device, ckpt_name="lstm_30k_best.pt")
    fam_priors = I.build_family_variant_priors()
    valid_rows, _ = I.build_self_eval(vocab, use_original_names=True)

    non_tot = non_hit = 0
    syn_tot = syn_hit = syn_maj_hit = 0
    for r in valid_rows:
        truth = r.get("_ACTUAL_NEXT_STEP", "")
        if not truth:
            continue
        family = r["FAMILY"].lower()
        steps = parse_pipe_sequence(r["PARTIAL_SEQUENCE"])
        prefix = encode_prefix(vocab, steps, family)
        prefix_orig = encode_prefix(vocab_orig, steps, family, apply_canonical=False)
        top1 = I.predict_next_top5_dual(model, orig_model, vocab, vocab_orig,
                                        prefix, prefix_orig, device, family,
                                        family_priors=fam_priors)[0]
        canon = CANONICAL_STEPS.get(truth, truth)
        if canon in VARIANTS_OF:
            syn_tot += 1
            if top1 == truth:
                syn_hit += 1
            if majority.get(canon) == truth:
                syn_maj_hit += 1
        else:
            non_tot += 1
            if top1 == truth:
                non_hit += 1

    n = non_tot + syn_tot
    print("\n=== DUAL-ENSEMBLE DECOMPOSITION (600 eval, original GT) ===")
    print(f"non-synonym : {non_hit}/{non_tot} = {100*non_hit/non_tot:.1f}%")
    print(f"synonym now : {syn_hit}/{syn_tot} = {100*syn_hit/syn_tot:.1f}%")
    print(f"synonym best: {syn_maj_hit}/{syn_tot} = {100*syn_maj_hit/syn_tot:.1f}%  (always-majority oracle)")
    print(f"\nOverall now          : {100*(non_hit+syn_hit)/n:.1f}%")
    print(f"Overall majority-fix : {100*(non_hit+syn_maj_hit)/n:.1f}%  <-- achievable ceiling")
    print(f"synonym fraction      : {100*syn_tot/n:.1f}% of eval steps")


if __name__ == "__main__":
    main()
