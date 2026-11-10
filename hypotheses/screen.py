"""Screening harness for five exploratory hypotheses (see each H*/ folder).

Each hypothesis is a small, mechanistically distinct intervention on the tied
embedding's SFT gradient, screened with a few seeds (NOT the full 12-seed sweep)
to judge whether it is worth a full confirmatory run.

All interventions operate on the shared embedding weight's gradient after
``loss.backward()``, when ``weight.grad == g_in + g_out`` (autograd sums the
input-path and output-path contributions). ``g_in`` is recovered from a forward
hook on the embedding output; ``g_out = weight.grad - g_in``.

Interventions (``--configs "name:dose[@extra]"``):
  input_scaling      g_eff = alpha * g_in + g_out            (alpha = dose)
  inverse_conflict   g_eff = (1 + beta*w_i) * g_in + g_out    (w = max(0,-c)/max)
  norm_scaling       g_eff = (1 + beta*w_i) * g_in + g_out    (w = g_out_norm/max)
  output_damping     g_eff = g_in + gamma * g_out             (gamma = dose)
  orthogonalize      g_eff = g_in + g_out - lam * (anti-aligned g_out component)
  early_rebalance    g_eff = alpha*g_in + g_out for step < T  (dose@T)

This script is exploratory: it records (satisfaction, nll, recall) only and does
not save model checkpoints.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make repo-root modules (common, sft, reproducibility, ...) importable when
# this script is invoked as `python hypotheses/screen.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from contextlib import nullcontext
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from reproducibility import atomic_json
from common import load_pretrained, load_tokenizer
from sft import collate, configure_optimizer, evaluate, get_lr, load_pairs
from transformer_architecture import SelectiveTransformer


# --------------------------------------------------------------------------- #
# Per-token weight vectors (from calibration.json)
# --------------------------------------------------------------------------- #

def build_weights(calib: dict, scheme: str) -> np.ndarray:
    V = calib["vocab_size"]
    conflict = np.array([0.0 if c is None else float(c) for c in calib["conflict"]],
                        dtype=np.float64)
    if scheme == "inverse_conflict":
        # aligned tokens: input/output gradients point the SAME way (negative c)
        pos = np.clip(-conflict, 0.0, None)
    elif scheme == "norm_scaling":
        pos = np.array(calib["g_out_norm"], dtype=np.float64)
    else:
        raise ValueError(f"unknown weight scheme {scheme!r}")
    mx = float(pos.max())
    w = np.zeros(V, dtype=np.float32)
    if mx > 0:
        w = (pos / mx).astype(np.float32)
    return w


# --------------------------------------------------------------------------- #
# One (intervention, dose, seed) run
# --------------------------------------------------------------------------- #

def run_one(args, name, dose, extra, weight_t, seed, train_pairs, val_pairs,
            tok, ctx, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SelectiveTransformer().to(device)
    load_pretrained(model, args.ckpt_path, device)
    model.train()
    optimizer = configure_optimizer(model, args.weight_decay, args.lr)

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
    V = model.config.vocab_size

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

        if "dx" in captured:
            dx = captured["dx"]                       # [B, L, D]
            g_in = torch.zeros_like(model.token_embedding.weight)
            g_in.index_add_(0, input_ids.reshape(-1),
                            dx.reshape(-1, dx.size(-1)))
            grad = model.token_embedding.weight.grad  # == g_in + g_out
            g_out = grad - g_in

            if name == "input_scaling":
                grad.copy_(dose * g_in + g_out)
            elif name == "inverse_conflict" or name == "norm_scaling":
                grad.copy_(g_in + g_out + dose * weight_t.unsqueeze(1) * g_in)
            elif name == "output_damping":
                grad.copy_(g_in + dose * g_out)
            elif name == "orthogonalize":
                g_in_n = torch.norm(g_in, dim=1, keepdim=True).clamp(min=1e-8)
                g_in_hat = g_in / g_in_n
                proj = (g_out * g_in_hat).sum(dim=1)          # [V] signed proj
                anti = torch.clamp(proj, max=0.0)             # [V] anti-aligned
                g_out_conflict = anti.unsqueeze(1) * g_in_hat  # [V,D]
                grad.copy_(g_in + g_out - dose * g_out_conflict)
            elif name == "early_rebalance":
                if step < extra:
                    grad.copy_(dose * g_in + g_out)
                # else: leave tied (g_in + g_out)

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % 250 == 0:
            print(f"  {name}_{dose:g}_seed{seed} step {step} loss {loss.item():.3f}")

    handle.remove()
    model.eval()
    sat, nll, details = evaluate(model, val_pairs, tok, ctx, device,
                                 return_details=True)
    return sat, nll, details["keyword_recall"]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description="Screening harness for 5 hypotheses")
    p.add_argument("--tokenizer_path", default="tokenizer.json")
    p.add_argument("--ckpt_path", default="checkpoints/latest.pt")
    p.add_argument("--calibration", default="calibration.json")
    p.add_argument("--sft_train", default="dataset/TinyStories-Instruct-train.txt")
    p.add_argument("--sft_valid", default="dataset/TinyStories-Instruct-valid.txt")
    p.add_argument("--out", required=True)
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
    p.add_argument("--configs", nargs="+", required=True,
                   help="'name:dose[@extra]' intervention configs")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer(args.tokenizer_path)
    ctx = args.context_length

    # parse configs -> list[(name, dose, extra)]
    configs = []
    for c in args.configs:
        name, rest = c.split(":")
        if "@" in rest:
            d, e = rest.split("@")
            configs.append((name, float(d), float(e)))
        else:
            configs.append((name, float(rest), None))

    with open(args.calibration) as f:
        calib = json.load(f)

    # weight vectors for the per-token schemes used
    weights = {}
    for name, _, _ in configs:
        if name in ("inverse_conflict", "norm_scaling"):
            weights[name] = torch.from_numpy(
                build_weights(calib, name)).to(device)

    print("loading SFT pairs...")
    train_pairs = load_pairs(args.sft_train, tok, ctx, args.sft_examples)
    val_pairs = load_pairs(args.sft_valid, tok, ctx, args.eval_examples)
    print(f"train pairs: {len(train_pairs):,} | eval pairs: {len(val_pairs):,}")

    results = {}
    if __import__("os").path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)

    for name, dose, extra in configs:
        weight_t = weights.get(name)
        for seed in args.seeds:
            key = f"{name}_{dose:g}_seed{seed}" if extra is None \
                else f"{name}_{dose:g}@{extra:g}_seed{seed}"
            if key in results:
                print(f"[{key}] already done, skipping")
                continue
            t0 = time.time()
            sat, nll, recall = run_one(args, name, dose, extra, weight_t, seed,
                                       train_pairs, val_pairs, tok, ctx, device)
            results[key] = {"intervention": name, "dose": dose,
                            "extra": extra, "seed": seed,
                            "keyword_satisfaction": sat,
                            "response_nll": nll,
                            "keyword_recall": recall}
            atomic_json(args.out, results)
            print(f"[{key}] satisfaction={sat:.4f} nll={nll:.4f} "
                  f"recall={recall:.4f} ({time.time() - t0:.0f}s)")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
