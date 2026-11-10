"""Audit normalized exact evaluation-story overlap using pinned official sources.

Streams only the needed instruction-data prefixes, then scans the full official
pretraining split. Does not change the experiment or remove records retroactively.
"""
import argparse
import hashlib
import io
import json
import os
import re
import urllib.request
from pathlib import Path

from common import parse_record, story_contains_words, load_tokenizer
from sft import INSTRUCTION_TEMPLATE
from reproducibility import atomic_json, sha256


def normalized(text):
    return re.sub(r'\s+', ' ', text).strip().casefold()


def digest(text):
    return hashlib.sha256(normalized(text).encode('utf-8')).hexdigest()


def records(lines):
    buffer = []
    for line in lines:
        if '<|endoftext|>' in line:
            buffer.append(line.split('<|endoftext|>')[0])
            record = ''.join(buffer).strip()
            if record:
                yield record
            buffer = []
        else:
            buffer.append(line)


def instruction_scan(url, tok, limit, calibration=False):
    kept, scanned, tokens = [], 0, 0
    with urllib.request.urlopen(url, timeout=120) as response:
        with io.TextIOWrapper(response, encoding='utf-8') as text:
            for record in records(text):
                scanned += 1
                words, story = parse_record(record)
                if not words or not story:
                    continue
                ids = tok.encode(story).ids
                if calibration:
                    if len(ids) + 1 > 256:
                        continue
                    tokens += len(ids) + 1
                else:
                    if not story_contains_words(story, words):
                        continue
                    prompt = INSTRUCTION_TEMPLATE.format(words=', '.join(words))
                    if len(tok.encode(prompt).ids) + len(ids) + 1 > 256:
                        continue
                kept.append({'words': words, 'story_sha256_normalized': digest(story)})
                if (calibration and tokens >= limit) or (not calibration and len(kept) >= limit):
                    break
    return kept, scanned, tokens


def match_report(eval_rows, train_hashes):
    matched = [i for i, row in enumerate(eval_rows)
               if row['story_sha256_normalized'] in train_hashes]
    return {'matched_eval_examples': len(matched), 'eval_examples': len(eval_rows),
            'fraction': len(matched) / len(eval_rows), 'eval_indices_zero_based': matched}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='reports/overlap-audit.json')
    p.add_argument('--results', default='results_v2.json')
    p.add_argument('--tokenizer', default='tokenizer.json')
    p.add_argument('--stories_revision', default='main')
    p.add_argument('--instruct_revision', default='main')
    p.add_argument('--skip_pretrain', action='store_true')
    args = p.parse_args()
    os.environ.setdefault('HF_HOME', str(Path('dataset/.hf-cache').resolve()))
    from huggingface_hub import HfApi, hf_hub_url
    api = HfApi()
    instruct_sha = api.repo_info('roneneldan/TinyStoriesInstruct', repo_type='dataset',
                               revision=args.instruct_revision).sha
    stories_sha = api.repo_info('roneneldan/TinyStories', repo_type='dataset',
                              revision=args.stories_revision).sha
    tok = load_tokenizer(args.tokenizer)
    def url(split):
        return hf_hub_url('roneneldan/TinyStoriesInstruct', f'TinyStories-Instruct-{split}.txt',
                          repo_type='dataset', revision=instruct_sha)
    evaluation, n_eval, _ = instruction_scan(url('valid'), tok, 500)
    training, n_train, _ = instruction_scan(url('train'), tok, 20000)
    calibration, n_calibration, n_tokens = instruction_scan(url('train'), tok, 10000000, True)
    if len(evaluation) != 500 or len(training) != 20000:
        raise ValueError('Official prefixes did not produce the expected example counts')
    report = {'normalization': 'Unicode casefold; collapse whitespace; strip edges; preserve punctuation',
              'scope': 'Normalized exact story matches only. Near duplicates and semantic overlap are not tested.',
              'sources': {'instruct_revision': instruct_sha, 'stories_revision': stories_sha,
                          'tokenizer_sha256': sha256(args.tokenizer)},
              'scanned': {'validation_records': n_eval, 'sft_train_records': n_train,
                          'calibration_records': n_calibration, 'calibration_tokens': n_tokens},
              'sft_train': match_report(evaluation, {r['story_sha256_normalized'] for r in training}),
              'calibration_train': match_report(evaluation, {r['story_sha256_normalized'] for r in calibration}),
              'evaluation_unique_stories': len({r['story_sha256_normalized'] for r in evaluation}),
              'pretraining_train': {'status': 'not_scanned'}}
    results = json.loads(Path(args.results).read_text(encoding='utf-8'))
    mismatch, clean = [], {}
    contaminated = set(report['sft_train']['eval_indices_zero_based']) | set(report['calibration_train']['eval_indices_zero_based'])
    for key, run in results.items():
        missing = run['missing_words']
        if len(missing) != 500:
            raise ValueError(f'{key}: expected 500 prompt records')
        if any(not set(m).issubset(set(evaluation[i]['words'])) for i, m in enumerate(missing)):
            mismatch.append(key)
        recall = sum((len(evaluation[i]['words'])-len(m))/len(evaluation[i]['words']) for i,m in enumerate(missing))/500
        if abs(recall-run['keyword_recall']) > .000051:
            mismatch.append(key + ':recall')
    report['results_alignment_checks'] = {'mismatches': mismatch,
        'note': 'Checks missing words against reconstructed prompts and aggregate recall; not a full historical provenance proof.'}
    atomic_json(args.output, report)
    if not args.skip_pretrain:
        from datasets import load_dataset
        target = {r['story_sha256_normalized'] for r in evaluation}
        matches, count = set(), 0
        dataset = load_dataset('roneneldan/TinyStories', revision=stories_sha, split='train', streaming=True)
        for row in dataset:
            count += 1
            h = digest(row['text'])
            if h in target:
                matches.add(h)
            if count % 100000 == 0:
                print(f'Scanned {count:,} pretraining records', flush=True)
        report['pretraining_train'] = dict(status='complete', scanned_records=count,
                                          **match_report(evaluation, matches))
        contaminated |= set(report['pretraining_train']['eval_indices_zero_based'])
    clean_indices = [i for i in range(500) if i not in contaminated]
    for key, run in results.items():
        if clean_indices:
            clean[key] = {'examples':len(clean_indices),
                'keyword_satisfaction':sum(not run['missing_words'][i] for i in clean_indices)/len(clean_indices),
                'keyword_recall':sum((len(evaluation[i]['words'])-len(run['missing_words'][i]))/len(evaluation[i]['words']) for i in clean_indices)/len(clean_indices)}
    report['nonoverlap_subset_lexical_metrics'] = clean
    report['nonoverlap_subset_scope'] = 'Excludes exact overlaps in scanned sources only; no NLL reconstruction from missing-word records.'
    atomic_json(args.output, report)
    print(json.dumps({k:v for k,v in report.items() if k != 'nonoverlap_subset_lexical_metrics'},indent=2))


if __name__ == '__main__':
    main()
