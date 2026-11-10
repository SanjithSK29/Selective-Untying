"""Download official datasets at immutable revisions and prepare CSV inputs."""
import argparse
import csv
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from reproducibility import atomic_json, sha256


def stories(path):
    buffer = ''
    with open(path, encoding='utf-8') as f:
        for line in f:
            buffer += line
            while '<|endoftext|>' in buffer:
                story, buffer = buffer.split('<|endoftext|>', 1)
                if story.strip():
                    yield story.strip()
    if buffer.strip():
        yield buffer.strip()


def to_csv(source, target):
    temporary = target.with_suffix('.csv.tmp')
    count = 0
    with temporary.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['text'])
        for story in stories(source):
            writer.writerow([story])
            count += 1
    temporary.replace(target)
    return count


def main():
    # Use the same official dataset split and CSV conversion as the experiments.
    from download_data import main as download_main
    download_main()


if __name__ == '__main__':
    main()
