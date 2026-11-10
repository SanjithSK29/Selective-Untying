"""Download the datasets used in this project into ./dataset/.

  1. TinyStoriesInstruct (SFT + calibration)  -> dataset/TinyStories-Instruct-{train,valid}.txt
       https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct
  2. TinyStories (pretraining)                -> dataset/train.csv, dataset/validation.csv
       https://huggingface.co/datasets/roneneldan/TinyStories

Usage:
    python download_data.py                 # both
    python download_data.py --only instruct # SFT/calibration data only
    python download_data.py --only pretrain # pretraining CSVs only

Note: the TinyStoriesInstruct train file is ~2.7 GB; the pretraining CSV is ~2 GB.
"""

import argparse
import csv
import os
import shutil

OUT_DIR = "dataset"


def download_instruct():
    from huggingface_hub import hf_hub_download

    for name in ("TinyStories-Instruct-train.txt", "TinyStories-Instruct-valid.txt"):
        dst = os.path.join(OUT_DIR, name)
        if os.path.exists(dst):
            print(f"[skip] {dst} exists")
            continue
        print(f"[download] {name}")
        cached = hf_hub_download(
            repo_id="roneneldan/TinyStoriesInstruct", filename=name, repo_type="dataset"
        )
        shutil.copyfile(cached, dst)
        print(f"  -> {dst}")


def download_pretrain():
    from datasets import load_dataset

    # train.py reads column 0 of each CSV row and skips a header row.
    for split, fname in (("train", "train.csv"), ("validation", "validation.csv")):
        dst = os.path.join(OUT_DIR, fname)
        if os.path.exists(dst):
            print(f"[skip] {dst} exists")
            continue
        print(f"[download] TinyStories split={split}")
        ds = load_dataset("roneneldan/TinyStories", split=split)
        with open(dst, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["text"])
            for row in ds:
                w.writerow([row["text"]])
        print(f"  -> {dst}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["instruct", "pretrain"], default=None)
    args = p.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.only in (None, "instruct"):
        download_instruct()
    if args.only in (None, "pretrain"):
        download_pretrain()
    print("done")
