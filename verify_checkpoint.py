"""Validate an original tied checkpoint and write its release metadata."""
import argparse
from dataclasses import asdict
import torch
from common import load_pretrained
from transformer_architecture import SelectiveTransformer
from reproducibility import atomic_json, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint')
    p.add_argument('--tokenizer', default='tokenizer.json')
    p.add_argument('--out', default='checkpoint-manifest.json')
    args = p.parse_args()
    model = SelectiveTransformer()
    load_pretrained(model, args.checkpoint, torch.device('cpu'))
    atomic_json(args.out, {'checkpoint_sha256': sha256(args.checkpoint),
                          'tokenizer_sha256': sha256(args.tokenizer),
                          'config': asdict(model.config), 'output_mode': 'tied',
                          'parameters': model.parameter_count()})
    print(f'Validated checkpoint; release metadata: {args.out}')


if __name__ == '__main__':
    main()
