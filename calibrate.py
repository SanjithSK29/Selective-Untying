"""Calibration: measure per-token input/output gradient conflict.

Loads the tied pretrained checkpoint, runs a two-pass gradient decomposition
over (a subset of) the TinyStoriesInstruct training stories, and ranks each
vocabulary row by its input-vs-output gradient conflict:

    conflict[i] = -cosine_similarity(g_in[i], g_out[i])

where ``g_in`` is the gradient contribution to the shared embedding from the
input path (token i used as input) and ``g_out`` is the contribution from the
output path (token i used as the next-token target). Higher conflict => the two
paths pull the embedding in opposite directions.

The output JSON stores, per token: the conflict score, corpus frequency,
gradient norms, an eligibility mask, and the final selected token ids. This one
artifact is enough for every selection baseline in sft.py (conflict, frequency,
frequency-matched random, gradient-norm), so all selectors share the same pool.

Run:
    python calibrate.py                     # defaults
    python calibrate.py --max_tokens 2000000 --min_freq 10   # quick
"""

from __future__ import annotations

import argparse
import json
import math
from array import array

import numpy as np
import torch
import torch.nn.functional as F

from common import iter_instruct_records, load_pretrained, load_tokenizer, parse_record
from transformer_architecture import SelectiveTransformer


def compute_grads(model: SelectiveTransformer, x: torch.Tensor, y: torch.Tensor):
    """Return (g_in, g_out) — the two gradient contributions to the embedding.

    g_total = g_in + g_out (autograd sums them on the shared weight). We get
    g_out directly by detaching the input path, then g_in = g_total - g_out.
    """
    vocab = model.config.vocab_size

    # Pass 1: total gradient (both paths).
    model.zero_grad(set_to_none=True)
    logits = model(x)
    F.cross_entropy(logits.view(-1, vocab).float(), y.view(-1)).backward()
    g_total = model.token_embedding.weight.grad.detach().clone()

    # Pass 2: output-only (input embedding output detached).
    def detach_hook(module, inp, out):
        return out.detach()

    handle = model.token_embedding.register_forward_hook(detach_hook)
    model.zero_grad(set_to_none=True)
    logits = model(x)
    F.cross_entropy(logits.view(-1, vocab).float(), y.view(-1)).backward()
    handle.remove()
    g_out = model.token_embedding.weight.grad.detach().clone()
    model.zero_grad(set_to_none=True)

    return g_total - g_out, g_out


def main():
    p = argparse.ArgumentParser(description="Gradient-conflict calibration")
    p.add_argument("--tokenizer_path", default="tokenizer.json")
    p.add_argument("--ckpt_path", default="checkpoints/latest.pt")
    p.add_argument("--sft_train", default="dataset/TinyStories-Instruct-train.txt")
    p.add_argument("--out", default="calibration.json")
    p.add_argument("--vocab_size", type=int, default=8000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_tokens", type=int, default=10_000_000,
                   help="cap on calibration tokens (subset is enough for stable estimates)")
    p.add_argument("--max_select", type=int, default=400)
    p.add_argument("--min_freq", type=int, default=20,
                   help="exclude tokens seen fewer than this many times as input")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    V = args.vocab_size
    ctx = args.context_length

    tok = load_tokenizer(args.tokenizer_path)
    eos = tok.token_to_id("<eos>")
    print(f"device: {device} | vocab: {V}")

    # ---- model (tied) ------------------------------------------------------ #
    model = SelectiveTransformer().to(device)
    load_pretrained(model, args.ckpt_path, device)
    model.eval()
    print(f"loaded tied checkpoint ({sum(p.numel() for p in model.parameters()):,} params)")

    # ---- collect calibration tokens + frequencies -------------------------- #
    ids_buf = array("H")
    freq = np.zeros(V, dtype=np.int64)
    n_stories = 0
    n_kept = 0
    n_no_words = 0
    n_too_long = 0

    for rec in iter_instruct_records(args.sft_train):
        n_stories += 1
        words, story = parse_record(rec)
        if words is None or not story:
            n_no_words += 1
            continue
        ids = tok.encode(story).ids
        if len(ids) + 1 > ctx:  # +1 for <eos>; match SFT's length filter
            n_too_long += 1
            continue
        for t in ids:
            freq[t] += 1
        ids_buf.extend(ids)
        ids_buf.append(eos)
        n_kept += 1
        if len(ids_buf) >= args.max_tokens:
            break

    data = np.array(ids_buf, dtype=np.uint16).astype(np.int64)
    print(f"scanned {n_stories:,} records | kept {n_kept:,} "
          f"(no-words {n_no_words:,}, too-long {n_too_long:,}) | "
          f"tokens {len(data):,}")

    # ---- two-pass accumulation -------------------------------------------- #
    num_blocks = (len(data) - 1) // ctx
    g_in_sum = torch.zeros(V, model.config.d_model, dtype=torch.float32, device=device)
    g_out_sum = torch.zeros(V, model.config.d_model, dtype=torch.float32, device=device)

    for b in range(0, num_blocks, args.batch_size):
        ends = min(b + args.batch_size, num_blocks)
        starts = (np.arange(b, ends, dtype=np.int64) * ctx)
        offs = np.arange(ctx, dtype=np.int64)
        x_idx = starts[:, None] + offs[None, :]
        x = torch.from_numpy(data[x_idx]).to(device)
        y = torch.from_numpy(data[x_idx + 1]).to(device)
        g_in, g_out = compute_grads(model, x, y)
        g_in_sum += g_in
        g_out_sum += g_out
        if (b // args.batch_size) % 20 == 0:
            print(f"  batch {b // args.batch_size + 1}/{math.ceil(num_blocks / args.batch_size)}")

    # ---- conflict scores --------------------------------------------------- #
    g_in_norm = g_in_sum.norm(dim=1).cpu().numpy()
    g_out_norm = g_out_sum.norm(dim=1).cpu().numpy()
    grad_norm = (g_in_sum + g_out_sum).norm(dim=1).cpu().numpy()
    dot = (g_in_sum * g_out_sum).sum(dim=1).cpu().numpy()
    denom = g_in_norm * g_out_norm
    conflict = np.full(V, np.nan, dtype=np.float64)
    valid = denom > 0
    conflict[valid] = -dot[valid] / denom[valid]
    conflict = np.clip(conflict, -1.0, 1.0)  # guard against fp noise

    # ---- eligibility + selection ------------------------------------------- #
    special = {tok.token_to_id("<unk>"), eos}
    eligible = np.zeros(V, dtype=bool)
    for i in range(V):
        eligible[i] = (i not in special
                       and freq[i] >= args.min_freq
                       and g_in_norm[i] > 0
                       and g_out_norm[i] > 0)
    eligible_count = int(eligible.sum())

    max_select = min(args.max_select, eligible_count // 10)
    rank = np.full(V, -np.inf, dtype=np.float64)
    rank[eligible] = conflict[eligible]
    order = np.argsort(-rank)  # descending conflict
    selected = [int(i) for i in order[:max_select]]

    print(f"eligible rows: {eligible_count:,} | max_select: {max_select} "
          f"(min(400, 10% of eligible))")
    print(f"selected {len(selected)} rows; top conflict scores: "
          f"{[round(conflict[i], 3) for i in selected[:8]]}")

    # ---- save -------------------------------------------------------------- #
    out = {
        "vocab_size": V,
        "min_freq": args.min_freq,
        "max_select": max_select,
        "eligible_count": eligible_count,
        "selected_token_ids": selected,
        "conflict": [None if math.isnan(c) else float(c) for c in conflict],
        "freq": freq.tolist(),
        "g_in_norm": g_in_norm.tolist(),
        "g_out_norm": g_out_norm.tolist(),
        "grad_norm": grad_norm.tolist(),
        "eligible": eligible.tolist(),
    }
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
