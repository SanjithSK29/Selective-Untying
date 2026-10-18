"""Analyze the conflict signal produced by calibrate.py.

Turns the calibration artifact into findings, without needing a behavioral win:

  - distribution of conflict scores (is the signal real / how spread out?),
  - conflict vs. frequency (rank correlation) — is "conflict" just frequency?,
  - conflict vs. gradient norm,
  - selection overlap (Jaccard) between conflict- and frequency-selected rows,
  - the top tokens by conflict (decoded) — what is being selected?,
  - mean conflict across frequency bins.

Run:
    python analyze.py
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from common import load_tokenizer


def spearman(a, b):
    """Rank correlation via Pearson on ranks."""
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return np.corrcoef(ra, rb)[0, 1]


def main():
    p = argparse.ArgumentParser(description="Analyze the conflict signal")
    p.add_argument("--calibration", default="calibration.json")
    p.add_argument("--tokenizer_path", default="tokenizer.json")
    p.add_argument("--top_k", type=int, default=20)
    args = p.parse_args()

    calib = json.load(open(args.calibration))
    tok = load_tokenizer(args.tokenizer_path)

    eligible = np.array(calib["eligible"], dtype=bool)
    conflict = np.array([np.nan if c is None else c for c in calib["conflict"]])
    freq = np.array(calib["freq"], dtype=np.float64)
    grad_norm = np.array(calib["grad_norm"], dtype=np.float64)
    g_in_norm = np.array(calib["g_in_norm"], dtype=np.float64)
    g_out_norm = np.array(calib["g_out_norm"], dtype=np.float64)

    mask = eligible & ~np.isnan(conflict)
    c, f, gn = conflict[mask], freq[mask], grad_norm[mask]
    print(f"eligible tokens with a conflict score: {int(mask.sum()):,}")
    print(f"conflict score: mean={np.nanmean(conflict[mask]):.4f} "
          f"median={np.median(conflict[mask]):.4f} "
          f"min={np.nanmin(conflict[mask]):.4f} max={np.nanmax(conflict[mask]):.4f}")
    print(f"  fraction with conflict > 0.3: {(conflict[mask] > 0.3).mean():.2%}")
    print(f"  fraction with conflict < 0.0: {(conflict[mask] < 0.0).mean():.2%}")

    print("\n=== conflict vs. frequency (rank correlation) ===")
    print(f"  Pearson : {np.corrcoef(c, f)[0, 1]:+.4f}")
    print(f"  Spearman: {spearman(c, f):+.4f}")
    print("  (near 0 => conflict is NOT just re-discovering frequent tokens)")

    print("\n=== conflict vs. gradient norm ===")
    print(f"  Spearman (total norm): {spearman(c, gn):+.4f}")
    print(f"  Spearman (input norm) : {spearman(c, g_in_norm[mask]):+.4f}")
    print(f"  Spearman (output norm): {spearman(c, g_out_norm[mask]):+.4f}")

    # mean conflict per frequency decile (is conflict concentrated on rare/common?)
    print("\n=== mean conflict by frequency decile (low->high freq) ===")
    q = np.quantile(f, np.linspace(0, 1, 11))
    for i in range(10):
        sel = (f >= q[i]) & (f <= q[i + 1])
        print(f"  decile {i + 1:2d}: n={int(sel.sum()):6d}  mean conflict={c[sel].mean():+.4f}")

    # selection overlap
    selected = set(calib["selected_token_ids"])
    max_select = calib["max_select"]
    # frequency-selected = top max_select by freq among eligible
    f2 = freq.copy()
    f2[~eligible] = -1
    freq_ids = set(np.argsort(-f2)[:max_select].tolist())
    # gradient-norm-selected
    gn2 = grad_norm.copy()
    gn2[~eligible] = -1
    gnorm_ids = set(np.argsort(-gn2)[:max_select].tolist())

    def jaccard(a, b):
        return len(a & b) / max(len(a | b), 1)

    print(f"\n=== selection overlap (Jaccard, {max_select} rows each) ===")
    print(f"  conflict vs frequency : {jaccard(selected, freq_ids):.3f}")
    print(f"  conflict vs grad-norm : {jaccard(selected, gnorm_ids):.3f}")

    # top tokens by conflict (decoded)
    order = np.argsort(-conflict)
    order = [i for i in order if eligible[i]][:args.top_k]
    print(f"\n=== top {args.top_k} tokens by conflict ===")
    for i in order:
        token = tok.id_to_token(int(i))
        print(f"  {token!r:24s} conflict={conflict[i]:+.3f} freq={freq[i]:.0f}")

    # top tokens by frequency (for contrast)
    order_f = np.argsort(-freq)
    order_f = [i for i in order_f if eligible[i]][:args.top_k]
    print(f"\n=== top {args.top_k} tokens by frequency ===")
    for i in order_f:
        token = tok.id_to_token(int(i))
        print(f"  {token!r:24s} conflict={conflict[i]:+.3f} freq={freq[i]:.0f}")


if __name__ == "__main__":
    main()
