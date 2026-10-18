"""Pretrain the tied SelectiveTransformer on TinyStories.

This file does NOT touch transformer_architecture.py. It is a standalone
training script with four stages, all run by `python train.py`:

  1. Build (or load) a BPE tokenizer trained on the training text only.
  2. Tokenize train/validation stories into compact uint16 ``.bin`` files
     (cached; regenerated only if missing).
  3. Instantiate the tied model (SelectiveTransformer with no selected ids).
  4. Run the next-token-prediction training loop with AdamW, a warmup+cosine
     schedule, gradient clipping, bf16 mixed precision, periodic evaluation
     (perplexity) and checkpointing.

The produced artifact is a ``.pt`` checkpoint of tied weights, which is the
single pretrained checkpoint the downstream SFT / calibration stages reuse.

Run:
    python train.py                       # defaults (one full pass over train)
    python train.py --max_steps 2000      # quick smoke test
    python train.py --help                # all knobs
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from array import array

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from tqdm import tqdm

from transformer_architecture import SelectiveTransformer


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #

def iter_stories(path: str):
    """Yield one story string per CSV row (streams; never loads the whole file)."""
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if row and row[0].strip():
                yield row[0]


def build_tokenizer(train_csv: str, vocab_size: int, save_path: str) -> ByteLevelBPETokenizer:
    """Train a byte-level BPE tokenizer on the training text only, then save it."""
    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(
        iter_stories(train_csv),
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<unk>", "<eos>"],
        show_progress=True,
    )
    tok.save(save_path)
    return tok


def load_tokenizer(path: str, vocab_size: int):
    tok = Tokenizer.from_file(path)
    assert tok.get_vocab_size() == vocab_size, (
        f"Tokenizer vocab size {tok.get_vocab_size()} != model vocab_size {vocab_size}. "
        f"Delete {path} and any *.bin files, then re-run."
    )
    return tok


def tokenize_to_bin(tokenizer: ByteLevelBPETokenizer, csv_path: str, out_path: str,
                    eos_id: int, desc: str) -> int:
    """Tokenize every story (append <eos>), stream to a uint16 .bin file."""
    buf = array("H")  # unsigned short: vocab 8000 << 65536
    total = 0
    with open(out_path, "wb") as f:
        for story in tqdm(iter_stories(csv_path), desc=desc, unit="story"):
            ids = tokenizer.encode(story).ids
            buf.extend(ids)
            buf.append(eos_id)
            total += len(ids) + 1
            if len(buf) >= 1_000_000:  # flush ~2 MB chunks
                buf.tofile(f)
                buf = array("H")
        if buf:
            buf.tofile(f)
    return total


def get_batch(data: np.memmap, starts: np.ndarray, ctx: int, device: torch.device):
    """Build x/y batches from absolute start positions (advanced indexing)."""
    offs = np.arange(ctx, dtype=np.int64)
    x_idx = starts[:, None] + offs[None, :]        # [B, ctx]
    y_idx = x_idx + 1                              # targets = next token
    x = torch.from_numpy(np.asarray(data[x_idx], dtype=np.int64))
    y = torch.from_numpy(np.asarray(data[y_idx], dtype=np.int64))
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


# --------------------------------------------------------------------------- #
# Optimizer / schedule helpers
# --------------------------------------------------------------------------- #

def configure_optimizer(model, weight_decay, lr, betas):
    """AdamW with weight decay applied only to 2-D weight matrices."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


def get_lr(step: int, warmup_steps: int, max_steps: int, lr: float, min_lr: float) -> float:
    """Linear warmup then cosine decay to min_lr."""
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (lr - min_lr)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Pretrain tied SelectiveTransformer")
    # data / io
    parser.add_argument("--train_csv", default="dataset/train.csv")
    parser.add_argument("--val_csv", default="dataset/validation.csv")
    parser.add_argument("--train_bin", default="dataset/train.bin")
    parser.add_argument("--val_bin", default="dataset/val.bin")
    parser.add_argument("--tokenizer_path", default="tokenizer.json")
    parser.add_argument("--ckpt_dir", default="checkpoints")
    # model / data shape (must match transformer_architecture.py defaults)
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--context_length", type=int, default=256)
    # training
    parser.add_argument("--batch_size", type=int, default=32,
                        help="sequences per micro-batch")
    parser.add_argument("--grad_accum_steps", type=int, default=2,
                        help="effective batch = batch_size * grad_accum_steps")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="0 => one full pass over the tokenized train data")
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--eval_iters", type=int, default=40)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--ckpt_interval", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compile", action="store_true",
                        help="use torch.compile (faster, slower first call)")
    args = parser.parse_args()

    # ---- device / seeds --------------------------------------------------- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"  gpu: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---- 1. tokenizer ----------------------------------------------------- #
    if os.path.exists(args.tokenizer_path):
        tok = load_tokenizer(args.tokenizer_path, args.vocab_size)
        print(f"Loaded tokenizer from {args.tokenizer_path} "
              f"(vocab={tok.get_vocab_size()})")
    else:
        tok = build_tokenizer(args.train_csv, args.vocab_size, args.tokenizer_path)
        print(f"Trained tokenizer -> {args.tokenizer_path} "
              f"(vocab={tok.get_vocab_size()})")
    eos_id = tok.token_to_id("<eos>")

    # ---- 2. tokenize data (cached) --------------------------------------- #
    if not os.path.exists(args.train_bin):
        n = tokenize_to_bin(tok, args.train_csv, args.train_bin, eos_id, "tokenize train")
        print(f"Wrote {args.train_bin} ({n:,} tokens)")
    if not os.path.exists(args.val_bin):
        n = tokenize_to_bin(tok, args.val_csv, args.val_bin, eos_id, "tokenize val")
        print(f"Wrote {args.val_bin} ({n:,} tokens)")

    # ---- 3. model ---------------------------------------------------------- #
    model = SelectiveTransformer().to(device)
    if args.compile:
        model = torch.compile(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    # ---- optimizer + resume ------------------------------------------------ #
    optimizer = configure_optimizer(model, args.weight_decay, args.lr, (0.9, 0.95))
    start_step = 0
    os.makedirs(args.ckpt_dir, exist_ok=True)
    latest = os.path.join(args.ckpt_dir, "latest.pt")
    if args.resume and os.path.exists(latest):
        ckpt = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"Resumed from step {start_step}")

    # ---- data memmap + block indices --------------------------------------- #
    ctx = args.context_length
    train_data = np.memmap(args.train_bin, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_bin, dtype=np.uint16, mode="r")
    n_train = len(train_data)
    n_val = len(val_data)

    num_blocks = (n_train - 1) // ctx
    block_starts = (np.arange(num_blocks, dtype=np.int64) * ctx)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(block_starts)
    steps_per_epoch = num_blocks // (args.batch_size * args.grad_accum_steps)

    if args.max_steps <= 0:
        args.max_steps = steps_per_epoch
    print(f"train tokens: {n_train:,} | blocks: {num_blocks:,} | "
          f"steps/epoch: {steps_per_epoch:,} | max_steps: {args.max_steps:,}")

    # val blocks for evaluation (fixed random set)
    val_starts = rng.integers(0, n_val - ctx - 1, size=(args.eval_iters,), dtype=np.int64)

    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if device.type == "cuda" else torch.no_grad())

    @torch.no_grad()
    def evaluate():
        model.eval()
        total = 0.0
        for i in range(args.eval_iters):
            x, y = get_batch(val_data, val_starts[i:i + 1], ctx, device)
            with autocast_ctx:
                logits = model(x)
            total += F.cross_entropy(logits.float().view(-1, args.vocab_size), y.view(-1)).item()
        model.train()
        return total / args.eval_iters

    # ---- 4. training loop --------------------------------------------------- #
    model.train()
    data_ptr = 0
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    val_loss = None

    t0 = time.time()
    step = start_step
    running_loss = 0.0

    while step < args.max_steps:
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # micro-batches for gradient accumulation
        for _ in range(args.grad_accum_steps):
            if data_ptr + args.batch_size > num_blocks:
                data_ptr = 0
                rng.shuffle(block_starts)  # reshuffle each epoch
            starts = block_starts[data_ptr: data_ptr + args.batch_size]
            data_ptr += args.batch_size

            x, y = get_batch(train_data, starts, ctx, device)
            with autocast_ctx:
                logits = model(x)
                loss = F.cross_entropy(logits.float().view(-1, args.vocab_size), y.view(-1))
                loss = loss / args.grad_accum_steps
            loss.backward()
            running_loss += loss.item() * args.grad_accum_steps

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        # logging
        if step % args.log_interval == 0:
            dt = time.time() - t0
            tok_s = (args.batch_size * args.grad_accum_steps * ctx
                     * args.log_interval) / dt
            print(f"step {step:>6,}/{args.max_steps:,} | "
                  f"loss {running_loss / args.log_interval:.4f} | "
                  f"lr {lr:.2e} | {tok_s:,.0f} tok/s")
            running_loss = 0.0
            t0 = time.time()

        # evaluation
        if step % args.eval_interval == 0 or step == args.max_steps:
            val_loss = evaluate()
            print(f"  [eval] step {step:>6,} | val loss {val_loss:.4f} | "
                  f"ppl {math.exp(val_loss):.2f}")

        # checkpoint
        if step % args.ckpt_interval == 0 or step == args.max_steps:
            ckpt = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "val_loss": val_loss,
                "vocab_size": args.vocab_size,
                "context_length": ctx,
            }
            torch.save(ckpt, latest)
            torch.save(ckpt, os.path.join(args.ckpt_dir, f"ckpt_{step}.pt"))
            print(f"  [ckpt] saved step {step}")

    print(f"\nDone. Final checkpoint: {latest}")


if __name__ == "__main__":
    main()
