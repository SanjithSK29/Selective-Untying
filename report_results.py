"""Rebuild descriptive tables, paired uncertainty and figures from saved results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t
from reproducibility import atomic_json, sha256


def describe(values):
    x = np.asarray(values, dtype=float)
    n = len(x)
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else None
    half = float(t.ppf(.975, n - 1) * sd / np.sqrt(n)) if n > 1 else None
    return {'n': n, 'mean': mean, 'sample_sd': sd,
            'ci95': [mean - half, mean + half] if half is not None else None}


def summarize(main, scaling):
    groups = {}
    for record in main.values():
        seeds = groups.setdefault(record['method'], {})
        if record['seed'] in seeds:
            raise ValueError('Duplicate method/seed')
        seeds[record['seed']] = record
    for record in scaling.values():
        groups.setdefault(f"alpha_{record['alpha']:g}", {})[record['seed']] = record
    output = {'methods': {}, 'paired_comparisons': {}}
    for name, records in groups.items():
        output['methods'][name] = {
            'satisfaction_percent': describe([100 * r['keyword_satisfaction'] for r in records.values()]),
            'response_nll': describe([r['response_nll'] for r in records.values()]),
            'seeds': sorted(records)}
        if all('keyword_recall' in r for r in records.values()):
            output['methods'][name]['recall_percent'] = describe([100*r['keyword_recall'] for r in records.values()])
    for left, right in [('conflict', 'frequency'), ('conflict', 'freq_random'),
                        ('conflict', 'tied'), ('full', 'tied'), ('alpha_10', 'tied')]:
        if left not in groups or right not in groups:
            continue
        seeds = sorted(set(groups[left]) & set(groups[right]))
        if not seeds:
            continue
        output['paired_comparisons'][f'{left}_minus_{right}'] = {
            'seeds': seeds,
            'satisfaction_pp': describe([100 * (groups[left][s]['keyword_satisfaction'] -
                                                groups[right][s]['keyword_satisfaction']) for s in seeds]),
            'response_nll': describe([groups[left][s]['response_nll'] -
                                     groups[right][s]['response_nll'] for s in seeds])}
    for left, right in [('conflict','frequency'), ('conflict','freq_random'), ('conflict','tied')]:
        key = f'{left}_minus_{right}'
        if key in output['paired_comparisons']:
            seeds = output['paired_comparisons'][key]['seeds']
            if all('keyword_recall' in groups[m][seed] for m in (left,right) for seed in seeds):
                output['paired_comparisons'][key]['recall_pp'] = describe([100*(groups[left][seed]['keyword_recall']-groups[right][seed]['keyword_recall']) for seed in seeds])
    if all(name in groups for name in ('conflict', 'frequency', 'freq_random', 'tied')):
        m = output['methods']
        diffs = {b: m['conflict']['satisfaction_percent']['mean'] -
                    m[b]['satisfaction_percent']['mean'] for b in ('frequency', 'freq_random')}
        nll = m['conflict']['response_nll']['mean'] - m['tied']['response_nll']['mean']
        output['success_criterion'] = {'threshold_pp': 2, 'differences_pp': diffs,
                                       'nll_difference_vs_tied': nll,
                                       'met': all(d >= 2 for d in diffs.values()) and nll <= 0}
    return output, groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', default='results_v2.json')
    p.add_argument('--scaling', default=None)
    p.add_argument('--output', default='reports')
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result, groups = summarize(json.loads(Path(args.results).read_text()),
                               json.loads(Path(args.scaling).read_text()) if args.scaling else {})
    result['sources'] = {}
    for source in [args.results] + ([args.scaling] if args.scaling else []):
        semantic = json.dumps(json.loads(Path(source).read_text(encoding='utf-8')), sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
        result['sources'][source] = {'file_sha256': sha256(source),
                                   'canonical_json_sha256': hashlib.sha256(semantic).hexdigest()}
    result['uncertainty_scope'] = ('Descriptive paired t intervals over SFT seeds on one fixed '
                                   'evaluation sample and one pretrained checkpoint. Historical inputs '
                                   'are rounded aggregates. No prompt-level or pretraining uncertainty.')
    atomic_json(output / 'summary.json', result)
    lines = ['# Results from committed aggregate records', '',
             'These calculations do not independently reproduce training or evaluation.', '',
             '| Method | Seeds | Satisfaction (%) | Sample SD (pp) | Keyword recall (%) | Response NLL |',
             '|---|---:|---:|---:|---:|---:|']
    with (output / 'methods.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'seeds', 'satisfaction_percent', 'sd_pp', 'keyword_recall_percent', 'response_nll'])
        for name, row in result['methods'].items():
            sat, nll = row['satisfaction_percent'], row['response_nll']
            recall = row.get('recall_percent', {}).get('mean')
            writer.writerow([name, sat['n'], sat['mean'], sat['sample_sd'], recall, nll['mean']])
            sd = f"{sat['sample_sd']:.2f}" if sat['sample_sd'] is not None else 'NA'
            recall_text = f'{recall:.2f}' if recall is not None else 'not recorded'
            lines.append(f"| {name} | {sat['n']} | {sat['mean']:.2f} | {sd} | {recall_text} | {nll['mean']:.5f} |")
    lines += ['', '## Paired differences', '',
              '| Comparison | Satisfaction difference (pp) | Approx. 95% t interval (pp) |',
              '|---|---:|---:|']
    for name, row in result['paired_comparisons'].items():
        stat = row['satisfaction_pp']
        interval = ', '.join(f'{v:+.2f}' for v in stat['ci95']) if stat['ci95'] else 'NA'
        lines.append(f"| {name} | {stat['mean']:+.2f} | [{interval}] |")
    lines += ['', result['uncertainty_scope'], '',
              'The criterion is not met for the historical records. Failure to meet it does not '
              'establish equivalence or absence of effects in other settings.', '',
              'NLL measures teacher-forced reference prediction. Keyword presence measures lexical '
              'compliance. Human fluency/coherence has not been established by these files.']
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = [n for n in ('tied', 'full', 'conflict', 'frequency', 'freq_random', 'grad_norm') if n in groups]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for seed in sorted({s for name in names for s in groups[name]}):
        ax.plot(range(len(names)), [100 * groups[name][seed]['keyword_satisfaction']
                                    if seed in groups[name] else np.nan for name in names],
                marker='o', alpha=.65, label=f'Seed {seed}')
    ax.set(xticks=range(len(names)), xticklabels=names, ylabel='All-keyword satisfaction (%)',
           title='Paired SFT seeds on the fixed evaluation sample')
    ax.legend(frameon=False, ncol=4, fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / 'paired_satisfaction.png', dpi=180)
    fig.savefig(output / 'paired_satisfaction.svg')
    plt.close(fig)
    alphas = [(1, 'tied')] + [(a, f'alpha_{a}') for a in (2, 5, 10)]
    if all(name in groups for _, name in alphas):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, metric, label, factor in [(axes[0], 'response_nll', 'Response NLL', 1),
                                          (axes[1], 'keyword_satisfaction', 'Satisfaction (%)', 100)]:
            for seed in sorted(set.intersection(*(set(groups[name]) for _, name in alphas))):
                ax.plot([a for a, _ in alphas], [factor * groups[name][seed][metric] for _, name in alphas],
                        marker='o', alpha=.6, label=f'Seed {seed}')
            ax.set(xlabel='Input-gradient scaling alpha', ylabel=label, xticks=[1, 2, 5, 10])
            ax.spines[['top', 'right']].set_visible(False)
        axes[0].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output / 'gradient_scaling.png', dpi=180)
        fig.savefig(output / 'gradient_scaling.svg')
        plt.close(fig)
    print(f'Wrote {output}/RESULTS.md, summary.json, methods.csv and figures')


if __name__ == '__main__':
    main()
