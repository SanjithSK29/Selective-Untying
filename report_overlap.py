"""Summarize the post-hoc non-overlap lexical sensitivity analysis."""
import json
from pathlib import Path
from report_results import describe
from reproducibility import atomic_json, sha256


def main():
    path = Path('reports/overlap-audit.json')
    audit = json.loads(path.read_text(encoding='utf-8'))
    if audit['pretraining_train'].get('status') != 'complete':
        raise ValueError('Full pretraining scan must complete first')
    source = json.loads(Path('results_v2.json').read_text(encoding='utf-8'))
    subset = audit['nonoverlap_subset_lexical_metrics']
    groups = {}
    for key, row in subset.items():
        groups.setdefault(source[key]['method'], {})[source[key]['seed']] = row
    result = {'scope': 'Post-hoc sensitivity analysis on 371 stories with no normalized exact match in scanned training sources. Not a new independent test set; near-duplicate overlap not checked. NLL unavailable.',
              'audit_sha256': sha256(path), 'methods':{}, 'paired_comparisons': {}}
    lines = ['# Post-hoc non-overlap sensitivity analysis', '',
             'The full scan found normalized exact pretraining matches for **129/500 (25.8%)**',
             'evaluation stories. No matches occurred in the reconstructed SFT/calibration',
             'training prefixes. The following analysis uses the remaining **371 prompts**.', '',
             '| Method | Seeds | Satisfaction (%) | Macro recall (%) |',
             '|---|---:|---:|---:|']
    for method, rows in groups.items():
        result['methods'][method] = {metric: describe([100*r[metric] for r in rows.values()])
                                     for metric in ('keyword_satisfaction','keyword_recall')}
        stat = result['methods'][method]
        lines.append(f"| {method} | {len(rows)} | {stat['keyword_satisfaction']['mean']:.2f} | {stat['keyword_recall']['mean']:.2f} |")
    lines += ['', '| Conflict minus comparator | Satisfaction difference (pp) | Paired 95% interval (pp) |',
              '|---|---:|---:|']
    for baseline in ('frequency','freq_random','tied'):
        seeds = sorted(set(groups['conflict']) & set(groups[baseline]))
        stat = {metric: describe([100*(groups['conflict'][s][metric]-groups[baseline][s][metric]) for s in seeds])
                for metric in ('keyword_satisfaction','keyword_recall')}
        result['paired_comparisons'][f'conflict_minus_{baseline}'] = stat
        satisfaction = stat['keyword_satisfaction']
        interval = satisfaction['ci95']
        lines.append(f"| {baseline} | {satisfaction['mean']:+.2f} | [{interval[0]:+.2f}, {interval[1]:+.2f}] |")
    lines += ['', result['scope'], '',
              'This subset analysis does not erase contamination in the full-set evaluation.',
              'It was defined after the main runs. Intervals concern SFT seeds on this fixed subset.',
              'The original models and generations were not rerun or changed. Response NLL on',
              'this subset cannot be reconstructed from aggregate NLL and missing-word lists.',
              'No subset NLL or human-coherence score is invented.']
    atomic_json('reports/nonoverlap-summary.json', result)
    Path('reports/NONOVERLAP.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
