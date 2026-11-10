"""Conflict-weighted input-gradient scaling (NEW, separate experiment).

Generalizes the global input-gradient scaling (alpha, see sft_gradscale.py) to a
PER-TOKEN input-path reweighting driven by the gradient-conflict calibration:

    g_eff,i = (1 + beta * w_i) * g_in,i + g_out,i

where ``g_in,i`` / ``g_out,i`` are the input-path / output-path contributions to
the shared embedding row ``i`` (the two terms autograd sums on the tied weight),
and ``w_i`` is a per-token weight derived from calibration.json.

Weighting schemes (``--configs "scheme:beta" ...``):
    conflict      w_i = max(0, c_i) / max_j max(0, c_j)   (continuous; the new idea)
    conflict_topk w_i = 1 if i in top-400 conflict else 0  (sparse ablation)
    freq_topk     w_i = 1 if i in top-400 frequency else 0 (frequency control)
    uniform       w_i = 1 for all i  (reproduces the global alpha baseline; sanity)

``beta`` is the reweighting strength (beta=0 -> tied; uniform + beta = global
alpha = 1+beta). This script is standalone: it imports shared helpers but does
not modify any existing experiment code.

Run:
    python sft_conflictgrad.py                                   # full schedule, 12 seeds
    python sft_conflictgrad.py --configs uniform:10 --seeds 0 --max_steps 20 --eval_examples 20
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from reproducibility import (
    atomic_json,
    provenance,
    prepare_results,
    prepare_artifacts,
    save_model,
)
from common import load_pretrained, load_tokenizer
from sft import collate, configure_optimizer, evaluate, get_lr, load_pairs
from transformer_architecture import SelectiveTransformer

DEFAULT_CONFIGS = [
    "conflict:1", "conflict:2", "conflict:5", "conflict:10",
    "conflict_topk:5", "freq_topk:5",
]


# --------------------------------------------------------------------------- #
# Per-token weights
# --------------------------------------------------------------------------- #

def build_weights(calib: dict, scheme: str) -> np.ndarray:
    """Return a float32 [V] weight vector for the requested scheme."""
    V = calib["vocab_size"]
    eligible = np.array(calib["eligible"], dtype=bool)
    conflict = np.array([0.0 if c is None else float(c) for c in calib["conflict"]],
                        dtype=np.float64)
    freq = np.array(calib["freq"], dtype=np.float64)
    max_select = int(calib["max_select"])
    w = np.zeros(V, dtype=np.float32)

    if scheme == "uniform":
        w[:] = 1.0
    elif scheme == "conflict":
        pos = np.clip(conflict, 0.0, None)
        cmax = float(pos.max())
        if cmax > 0:
            w = (pos / cmax).astype(np.float32)
    elif scheme == "conflict_topk":
        s = conflict.copy()
        s[~eligible] = -np.inf
        w[np.argsort(-s)[:max_select]] = 1.0
    elif scheme == "freq_topk":
        s = freq.copy()
        s[~eligible] = -np.inf
        w[np.argsort(-s)[:max_select]] = 1.0
    else:
        raise ValueError(f"unknown weighting scheme {scheme!r}")
    return w


# --------------------------------------------------------------------------- #
# One (scheme, beta, seed) run
# --------------------------------------------------------------------------- #

def run_one(args, scheme, beta, weight_t, seed, train_pairs, val_pairs, tok, ctx,
            device, manifest=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SelectiveTransformer().to(device)  # tied; rebalancing is train-time
    load_pretrained(model, args.ckpt_path, device)
    model.train()
    optimizer = configure_optimizer(model, args.weight_decay, args.lr)

    # Forward hook: capture gradient w.r.t. the embedding output each step.
    captured = {}

    def fwd_hook(module, inp, out):
        out.register_hook(lambda g: captured.__setitem__("dx", g))
        return out

    handle = model.token_embedding.register_forward_hook(fwd_hook)

    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) \
        if device.type == "cuda" else nullcontext()

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_pairs)).tolist()
    bs = args.batch_size

    for step in range(args.max_steps):
        batch = [train_pairs[order[(step * bs + j) % len(train_pairs)]]
                 for j in range(bs)]
        input_ids, targets = collate(batch, tok, device)
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr
        with autocast:
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                                   targets.view(-1), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Per-token reweighting: W.grad = g_in + g_out -> (1 + beta*w_i)*g_in + g_out
        if beta != 0.0 and "dx" in captured:
            dx = captured["dx"]                       # [B, L, D]
            g_in = torch.zeros_like(model.token_embedding.weight)
            g_in.index_add_(0, input_ids.reshape(-1),
                            dx.reshape(-1, dx.size(-1)))
            model.token_embedding.weight.grad += beta * weight_t.unsqueeze(1) * g_in

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % 200 == 0:
            print(f"  {scheme}:{beta:g}_seed{seed} step {step}/{args.max_steps} "
                  f"loss {loss.item():.3f}")

    handle.remove()
    model.eval()
    key = f"{scheme}_b{beta:g}_seed{seed}"
    artifact = Path(args.artifact_dir) / key
    sat, nll, details = evaluate(model, val_pairs, tok, ctx, device,
                                 artifact.with_suffix(".examples.json"),
                                 return_details=True)
    save_model(artifact.with_suffix(".pt"), model, manifest,
               scheme=scheme, beta=beta, seed=seed)
    atomic_json(artifact.with_suffix(".metadata.json"),
                {"manifest": manifest, "scheme": scheme, "beta": beta,
                 "seed": seed, "steps": args.max_steps})
    return sat, nll, details["keyword_recall"]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description="Conflict-weighted input-gradient scaling")
    p.add_argument("--tokenizer_path", default="tokenizer.json")
    p.add_argument("--ckpt_path", default="checkpoints/latest.pt")
    p.add_argument("--calibration", default="calibration.json")
    p.add_argument("--sft_train", default="dataset/TinyStories-Instruct-train.txt")
    p.add_argument("--sft_valid", default="dataset/TinyStories-Instruct-valid.txt")
    p.add_argument("--out", default="runs/conflictgrad/results.json")
    p.add_argument("--artifact_dir", default="runs/conflictgrad/artifacts")
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--sft_examples", type=int, default=20000)
    p.add_argument("--eval_examples", type=int, default=500)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS,
                   help="'scheme:beta' pairs, e.g. conflict:5 conflict_topk:5")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5,
                                                            6, 7, 8, 9, 10, 11])
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer(args.tokenizer_path)
    ctx = args.context_length

    # parse configs -> list[(scheme, beta)]
    configs = []
    for c in args.configs:
        scheme, beta = c.split(":")
        configs.append((scheme, float(beta)))
    schemes = sorted({s for s, _ in configs})

    with open(args.calibration) as f:
        calib = json.load(f)

    print("loading SFT pairs...")
    train_pairs = load_pairs(args.sft_train, tok, ctx, args.sft_examples)
    val_pairs = load_pairs(args.sft_valid, tok, ctx, args.eval_examples)
    print(f"train pairs: {len(train_pairs):,} | eval pairs: {len(val_pairs):,}")

    if not train_pairs or not val_pairs:
        raise ValueError("Training and evaluation data must contain qualifying pairs")
    if ctx != 256:
        raise ValueError("This experiment requires context_length=256")

    manifest = provenance(args, [args.tokenizer_path, args.ckpt_path,
                                 args.sft_train, args.sft_valid, args.calibration])
    results = prepare_results(args.out, manifest)
    prepare_artifacts(args.artifact_dir, manifest)

    weights = {s: torch.from_numpy(build_weights(calib, s)).to(device)
               for s in schemes}
    for s, w in weights.items():
        print(f"weight[{s}]: {int((w > 0).sum()):,} tokens have w>0, "
              f"max={float(w.max()):.4f}")

    for scheme, beta in configs:
        weight_t = weights[scheme]
        for seed in args.seeds:
            key = f"{scheme}_b{beta:g}_seed{seed}"
            if key in results:
                print(f"[{key}] already done, skipping")
                continue
            t0 = time.time()
            sat, nll, recall = run_one(args, scheme, beta, weight_t, seed,
                                       train_pairs, val_pairs, tok, ctx, device,
                                       manifest)
            results[key] = {"scheme": scheme, "beta": beta, "seed": seed,
                            "keyword_satisfaction": sat,
                            "response_nll": nll,
                            "keyword_recall": recall}
            atomic_json(args.out, results)
            print(f"[{key}] satisfaction={sat:.4f} nll={nll:.4f} "
                  f"recall={recall:.4f} ({time.time() - t0:.0f}s)")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
