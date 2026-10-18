"""Shared helpers for the calibration and SFT stages.

Kept separate from transformer_architecture.py (the model) and train.py (the
pretraining loop) so the three concerns stay independent.
"""

from __future__ import annotations

import re
from typing import Iterator, Optional, Sequence, Tuple

import torch
from tokenizers import Tokenizer

from transformer_architecture import SelectiveTransformer


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #

def load_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)


# --------------------------------------------------------------------------- #
# TinyStoriesInstruct record parsing
# --------------------------------------------------------------------------- #

def iter_instruct_records(path: str) -> Iterator[str]:
    """Yield one raw record at a time (records are <|endoftext|>-delimited)."""
    buf: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "<|endoftext|>" in line:
                # flush everything up to (and including) the delimiter
                buf.append(line.split("<|endoftext|>")[0])
                rec = "".join(buf).strip()
                if rec:
                    yield rec
                buf = []
            else:
                buf.append(line)


def parse_record(rec: str) -> Tuple[Optional[list[str]], Optional[str]]:
    """Extract the ``Words`` and ``Story`` fields from a raw record.

    Returns (words, story). ``words`` is None if the record has no ``Words``
    field; ``story`` is None if there is no ``Story`` field.
    """
    words: Optional[list[str]] = None
    story: Optional[str] = None
    lines = rec.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("Words:"):
            raw = line[len("Words:"):].strip()
            words = [w.strip().lower() for w in raw.split(",") if w.strip()]
        elif line.startswith("Story:"):
            body = [l for l in lines[i + 1:] if l.strip()]
            story = "\n".join(body).strip()
    return words, story


_WORD_RE = re.compile(r"\b[a-z]+\b")


def story_contains_words(story: str, words: Sequence[str]) -> bool:
    """Case-insensitive whole-word check that every requested word is present."""
    present = set(_WORD_RE.findall(story.lower()))
    return all(w in present for w in words)


# --------------------------------------------------------------------------- #
# Checkpoint loading (mode-aware)
# --------------------------------------------------------------------------- #

def load_pretrained(model: SelectiveTransformer, ckpt_path: str,
                    device: torch.device) -> SelectiveTransformer:
    """Load the tied pretrained checkpoint into ``model`` (any output mode).

    - tied:     loads everything directly.
    - full:     skips the missing output head, then copies the tied embedding
                into it so it starts identical to the tied model.
    - selective: skips the (empty) offsets in the checkpoint, leaving the
                zero-initialized offsets untouched (the LoRA-style zero start).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = ckpt["model"]
    model_sd = model.state_dict()
    keep = {k: v for k, v in sd.items()
            if k in model_sd and model_sd[k].shape == v.shape}
    model.load_state_dict(keep, strict=False)
    if model.output_mode == "full":
        model.init_output_from_input()
    return model
