"""SFT with input-gradient scaling — ref [3]'s causal fix, as a comparison arm.

Ref [3] showed the tied embedding is "stolen" by the output gradient and fixed
it by *scaling the input-path gradient* by a factor alpha during training:

    dL/dW_effective = alpha * g_in + g_out     (g_in = input path, g_out = output path)

alpha = 1 is exactly the tied baseline (no scaling). This script sweeps alpha > 1
and compares against the tied model already stored in results.json.

The scaling is applied with a forward hook on ``token_embedding`` that captures
the gradient w.r.t. the embedding *output* (dx). The input-path contribution to
the shared weight is then g_in = scatter(dx, input_ids), so one forward/backward
per step is enough — no model changes, no extra passes.

Run:
    python sft_gradscale.py                          # alphas 2,5,10 x 3 seeds
    python sft_gradscale.py --alphas 5 --seeds 0     # quick smoke
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import load_pretrained, load_tokenizer
from sft import collate, configure_optimizer, evaluate, get_lr, load_pairs
from transformer_architecture import SelectiveTransformer


def run_one(args, alpha, seed, train_pairs, val_pairs, tok, ctx, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SelectiveTransformer().to(device)  # tied model; scaling is a train-time change
    load_pretrained(model, args.ckpt_path, device)
    model.train()
    optimizer = configure_optimizer(model, args.weight_decay, args.lr)

    # Forward hook: capture the gradient w.r.t. the embedding output each step.
    captured = {}

    def fwd_hook(module, inp, out):
        out.register_hook(lambda g: captured.__setitem__("dx", g))
        return out

    handle = model.token_embedding.register_forward_hook(fwd_hook)

    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) \
        if device.type == "cuda" else torch.no_grad()

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

        # Reweight the input-path contribution: W.grad = g_in + g_out -> alpha*g_in + g_out
        if alpha != 1.0 and "dx" in captured:
            dx = captured["dx"]                       # [B, L, D]
            g_in = torch.zeros_like(model.token_embedding.weight)
            g_in.index_add_(0, input_ids.reshape(-1), dx.reshape(-1, dx.size(-1)))
            model.token_embedding.weight.grad += (alpha - 1.0) * g_in

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

    handle.remove()
    model.eval()
    return evaluate(model, val_pairs, tok, ctx, device)


def main():
    p = argparse.ArgumentParser(description="SFT with input-gradient scaling")
    p.add_argument("--tokenizer_path", default="tokenizer.json")
    p.add_argument("--ckpt_path", default="checkpoints/latest.pt")
    p.add_argument("--sft_train", default="dataset/TinyStories-Instruct-train.txt")
    p.add_argument("--sft_valid", default="dataset/TinyStories-Instruct-valid.txt")
    p.add_argument("--out", default="results_gradscale.json")
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
    p.add_argument("--alphas", nargs="+", type=float, default=[2.0, 5.0, 10.0])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer(args.tokenizer_path)
    ctx = args.context_length

    print("loading SFT pairs...")
    train_pairs = load_pairs(args.sft_train, tok, ctx, args.sft_examples)
    val_pairs = load_pairs(args.sft_valid, tok, ctx, args.eval_examples)
    print(f"train pairs: {len(train_pairs):,} | eval pairs: {len(val_pairs):,}")

    results = {}
    if os.path.exists(args.out):  # resume
        with open(args.out) as f:
            results = json.load(f)

    for alpha in args.alphas:
        for seed in args.seeds:
            key = f"gradscale_a{alpha:g}_seed{seed}"
            if key in results:
                print(f"[{key}] already done, skipping")
                continue
            t0 = time.time()
            sat, nll = run_one(args, alpha, seed, train_pairs, val_pairs, tok, ctx, device)
            results[key] = {"alpha": alpha, "seed": seed,
                            "keyword_satisfaction": round(sat, 4),
                            "response_nll": round(nll, 4)}
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[{key}] satisfaction={sat:.4f} nll={nll:.4f} ({time.time()-t0:.0f}s)")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
