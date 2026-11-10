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
REVISIONS = {}
from reproducibility import atomic_json, sha256


def record_source(path, repo, revision):
    manifest_path = os.path.join(OUT_DIR, "sources.json")
    import json
    manifest = json.load(open(manifest_path, encoding="utf-8")) if os.path.exists(manifest_path) else {}
    manifest[os.path.basename(path)] = {"repo": repo, "revision": revision, "sha256": sha256(path)}
    atomic_json(manifest_path, manifest)



def download_instruct():
    from huggingface_hub import hf_hub_download

    for name in ("TinyStories-Instruct-train.txt", "TinyStories-Instruct-valid.txt"):
        dst = os.path.join(OUT_DIR, name)
        if os.path.exists(dst):
            print(f"[skip] {dst} exists")
            continue
        print(f"[download] {name}")
        cached = hf_hub_download(
            repo_id="roneneldan/TinyStoriesInstruct", filename=name, repo_type="dataset",
            revision=REVISIONS["instruct"], cache_dir=os.path.join(OUT_DIR, ".hf-cache")
        )
        shutil.copyfile(cached, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
        record_source(dst, "roneneldan/TinyStoriesInstruct", REVISIONS["instruct"])
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
        ds = load_dataset("roneneldan/TinyStories", split=split, revision=REVISIONS["pretrain"],
                          cache_dir=os.path.join(OUT_DIR, ".hf-cache"))
        with open(dst + ".tmp", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["text"])
            for row in ds:
                w.writerow([row["text"]])
        os.replace(dst + ".tmp", dst)
        record_source(dst, "roneneldan/TinyStories", REVISIONS["pretrain"])
        print(f"  -> {dst}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["instruct", "pretrain"], default=None)
    p.add_argument("--output", default="dataset")
    p.add_argument("--stories_revision", default="main")
    p.add_argument("--instruct_revision", default="main")
    args = p.parse_args()
    global OUT_DIR
    OUT_DIR = args.output
    from huggingface_hub import HfApi
    if args.only in (None, "instruct"):
        REVISIONS["instruct"] = HfApi().repo_info("roneneldan/TinyStoriesInstruct", repo_type="dataset", revision=args.instruct_revision).sha
    if args.only in (None, "pretrain"):
        REVISIONS["pretrain"] = HfApi().repo_info("roneneldan/TinyStories", repo_type="dataset", revision=args.stories_revision).sha
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.only in (None, "instruct"):
        download_instruct()
    if args.only in (None, "pretrain"):
        download_pretrain()
    print("done")



if __name__ == "__main__":
    main()

