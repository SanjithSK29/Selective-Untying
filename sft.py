"""Supervised fine-tuning (SFT) and evaluation of the selective-untying methods.

Loads the single pretrained checkpoint, then for each method in
{tied, full, conflict, frequency, freq_random, grad_norm} and each seed,
fine-tunes on TinyStoriesInstruct (response-only cross-entropy, teacher
forcing) and evaluates:

  - all-keyword satisfaction (case-insensitive whole-word match on generated
    stories), and
  - response negative log-likelihood on held-out validation pairs.

Coherence/keyword-integration are blinded human ratings and are intentionally
left out of this automated script (see the proposal's evaluation criteria).

The conflict / frequency / freq_random / grad_norm methods all add the same
number of output offsets (<=400 rows, capped at 10% of the eligible pool) and
differ only in *which* rows are selected. ``full`` uses a separate output head,
and ``tied`` adds nothing.

Run:
    python sft.py                                   # all 6 methods x 3 seeds
    python sft.py --methods tied conflict --seeds 0 # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import (
    iter_instruct_records,
    load_pretrained,
    load_tokenizer,
    parse_record,
    story_contains_words,
)
from transformer_architecture import SelectiveTransformer

METHODS = ["tied", "full", "conflict", "frequency", "freq_random", "grad_norm"]

INSTRUCTION_TEMPLATE = "Write a short story using these words: {words}\n\n"

_WORD_RE = re.compile(r"\b[a-z]+\b")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_pairs(path: str, tok, ctx: int, max_examples: int | None = None):
    """Parse SFT data into (instr_ids, resp_ids, words) tuples, filtered.

    Keeps records that (a) have a Words field, (b) whose story contains all the
    requested words, and (c) whose instruction+response fits in ``ctx`` tokens.
    """
    eos = tok.token_to_id("<eos>")
    pairs = []
    for rec in iter_instruct_records(path):
        words, story = parse_record(rec)
        if words is None or not story:
            continue
        if not story_contains_words(story, words):
            continue
        instr = INSTRUCTION_TEMPLATE.format(words=", ".join(words))
        instr_ids = tok.encode(instr).ids
        resp_ids = tok.encode(story).ids + [eos]
        if len(instr_ids) + len(resp_ids) > ctx:
            continue
        pairs.append((instr_ids, resp_ids, words))
        if max_examples is not None and len(pairs) >= max_examples:
            break
    return pairs


def collate(batch, tok, device):
    """Right-pad a batch of (instr_ids, resp_ids, words) into tensors.

    Returns (input_ids, targets). Targets are -100 for instruction and padding
    positions so cross-entropy only supervises the response (response-only).
    """
    pad = tok.token_to_id("<eos>")
    max_len = max(len(i) + len(r) for i, r, _ in batch)
    B = len(batch)
    input_ids = torch.full((B, max_len), pad, dtype=torch.long)
    targets = torch.full((B, max_len), -100, dtype=torch.long)
    for b, (instr, resp, _) in enumerate(batch):
        s = instr + resp
        L = len(s)
        input_ids[b, : L - 1] = torch.tensor(s[:-1], dtype=torch.long)
        for p in range(len(instr) - 1, L - 1):  # response target positions
            targets[b, p] = s[p + 1]
    return input_ids.to(device), targets.to(device)


# --------------------------------------------------------------------------- #
# Selection builders (from the calibration artifact)
# --------------------------------------------------------------------------- #

def build_selections(calib, rng, n_bins: int = 10):
    eligible = np.array(calib["eligible"], dtype=bool)
    freq = np.array(calib["freq"], dtype=np.int64)
    conflict = np.array([-np.inf if c is None else c for c in calib["conflict"]])
    grad_norm = np.array(calib["grad_norm"], dtype=np.float64)
    max_select = calib["max_select"]

    def top_by(score):
        s = score.astype(np.float64).copy()
        s[~eligible] = -np.inf
        return np.argsort(-s)[:max_select].tolist()

    conflict_ids = top_by(conflict)
    freq_ids = top_by(freq)
    gradnorm_ids = top_by(grad_norm)

    # frequency-matched random: sample random eligible tokens with the same
    # frequency profile as the conflict selection (binned by frequency rank).
    eligible_idx = np.where(eligible)[0]
    order = np.argsort(freq[eligible_idx])
    bins = np.array_split(order, n_bins)
    token_bin = np.full(len(conflict), -1, dtype=int)
    for b, idxs in enumerate(bins):
        token_bin[eligible_idx[idxs]] = b
    ref_bins = token_bin[np.array(conflict_ids)]
    freq_random_ids = set()
    for b in range(n_bins):
        count = int((ref_bins == b).sum())
        pool = eligible_idx[bins[b]]
        if count > 0 and len(pool) > 0:
            freq_random_ids.update(
                rng.choice(pool, size=min(count, len(pool)), replace=False).tolist())
    if len(freq_random_ids) < max_select:
        remaining = [int(i) for i in eligible_idx if int(i) not in freq_random_ids]
        rng.shuffle(remaining)
        freq_random_ids.update(remaining[: max_select - len(freq_random_ids)])
    freq_random_ids = list(freq_random_ids)[:max_select]

    return {
        "conflict": conflict_ids,
        "frequency": freq_ids,
        "grad_norm": gradnorm_ids,
        "freq_random": freq_random_ids,
    }


# --------------------------------------------------------------------------- #
# Model / optimizer / schedule
# --------------------------------------------------------------------------- #

def build_model(method, selection_ids, eligible_count, device):
    if method == "full":
        model = SelectiveTransformer(output_mode="full")
    elif method == "tied":
        model = SelectiveTransformer()
    else:
        model = SelectiveTransformer(selected_token_ids=selection_ids,
                                     eligible_token_count=eligible_count)
    return model.to(device)


def configure_optimizer(model, weight_decay, lr):
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95),
    )


def get_lr(step, warmup, max_steps, lr, min_lr):
    if step < warmup:
        return lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (lr - min_lr)


# --------------------------------------------------------------------------- #
# Generation / evaluation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def generate(model, instr_ids, ctx, eos_id, device, max_new=256):
    ids = torch.tensor([instr_ids], dtype=torch.long, device=device)
    for _ in range(max_new):
        logits = model(ids[:, -ctx:])
        next_tok = logits[:, -1].argmax(dim=-1)
        ids = torch.cat([ids, next_tok[:, None]], dim=1)
        if int(next_tok.item()) == eos_id:
            break
    return ids[0].tolist()


def evaluate(model, eval_pairs, tok, ctx, device):
    eos = tok.token_to_id("<eos>")
    hits = 0
    total_nll = 0.0
    total_tok = 0

    # response NLL (teacher forcing), batched
    bs = 8
    for b in range(0, len(eval_pairs), bs):
        batch = eval_pairs[b: b + bs]
        input_ids, targets = collate(batch, tok, device)
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                               targets.view(-1), ignore_index=-100)
        n_tok = int((targets != -100).sum())
        total_nll += loss.item() * n_tok
        total_tok += n_tok

    # all-keyword satisfaction (greedy generation)
    for instr, _, words in eval_pairs:
        gen_ids = generate(model, instr, ctx, eos, device)
        gen_text = tok.decode(gen_ids[len(instr):])
        present = set(_WORD_RE.findall(gen_text.lower()))
        if all(w in present for w in words):
            hits += 1

    return hits / len(eval_pairs), total_nll / max(total_tok, 1)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description="SFT + evaluation of selective untying")
    p.add_argument("--tokenizer_path", default="tokenizer.json")
    p.add_argument("--ckpt_path", default="checkpoints/latest.pt")
    p.add_argument("--calibration", default="calibration.json")
    p.add_argument("--sft_train", default="dataset/TinyStories-Instruct-train.txt")
    p.add_argument("--sft_valid", default="dataset/TinyStories-Instruct-valid.txt")
    p.add_argument("--out", default="results.json")
    p.add_argument("--vocab_size", type=int, default=8000)
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
    p.add_argument("--methods", nargs="+", default=METHODS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer(args.tokenizer_path)
    ctx = args.context_length

    # calibration artifact (required for the selective methods)
    calib = None
    eligible_count = 0
    selections = None
    if any(m in {"conflict", "frequency", "freq_random", "grad_norm"} for m in args.methods):
        with open(args.calibration) as f:
            calib = json.load(f)
        eligible_count = calib["eligible_count"]
        selections = build_selections(calib, np.random.default_rng(0))
        print(f"loaded calibration (eligible={eligible_count:,})")

    # data
    print("loading SFT pairs...")
    train_pairs = load_pairs(args.sft_train, tok, ctx, args.sft_examples)
    val_pairs = load_pairs(args.sft_valid, tok, ctx, args.eval_examples)
    print(f"train pairs: {len(train_pairs):,} | eval pairs: {len(val_pairs):,}")

    results = {}
    if os.path.exists(args.out):  # resume: keep already-completed runs
        with open(args.out) as f:
            results = json.load(f)

    for method in args.methods:
        if method not in METHODS:
            raise ValueError(f"unknown method {method}")

        selection_ids = selections[method] if (selections is not None
                                               and method in selections) else None
        if selection_ids is not None:
            print(f"\n[{method}] selected {len(selection_ids)} rows")

        for seed in args.seeds:
            key = f"{method}_seed{seed}"
            if key in results:
                print(f"[{key}] already completed, skipping")
                continue
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = build_model(method, selection_ids, eligible_count, device)
            load_pretrained(model, args.ckpt_path, device)
            model.train()
            optimizer = configure_optimizer(model, args.weight_decay, args.lr)

            autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) \
                if device.type == "cuda" else torch.no_grad()

            rng = np.random.default_rng(seed)
            order = rng.permutation(len(train_pairs)).tolist()
            bs = args.batch_size

            t0 = time.time()
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                if step % 200 == 0:
                    print(f"  {key} step {step}/{args.max_steps} loss {loss.item():.3f}")

            model.eval()
            sat, nll = evaluate(model, val_pairs, tok, ctx, device)
            results[key] = {"method": method, "seed": seed,
                            "keyword_satisfaction": round(sat, 4),
                            "response_nll": round(nll, 4)}
            with open(args.out, "w") as f:  # save incrementally (crash-safe)
                json.dump(results, f, indent=2)
            print(f"[{key}] keyword_satisfaction={sat:.4f} response_nll={nll:.4f} "
                  f"({time.time() - t0:.0f}s)")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
