"""Checks for published result coverage and uncertainty calculations."""
import json
from pathlib import Path
import unittest
import numpy as np
from scipy.stats import t

from audit_overlap import digest, match_report, records
from report_results import describe, summarize


class Analysis(unittest.TestCase):
    def test_paired_interval_uses_sample_sd_and_correct_n(self):
        d = describe([1, 2, 3, 4])
        half = t.ppf(.975, 3) * np.std([1, 2, 3, 4], ddof=1) / 2
        self.assertEqual(d['n'], 4)
        self.assertAlmostEqual(d['mean'], 2.5)
        np.testing.assert_allclose(d['ci95'], [2.5-half, 2.5+half])

    def test_main_coverage_reproduction_and_missing_word_integrity(self):
        root = Path(__file__).parent
        data = json.loads((root/'results_v2.json').read_text(encoding='utf-8'))
        old = json.loads((root/'results.json').read_text(encoding='utf-8'))
        methods = ('tied', 'conflict', 'frequency', 'freq_random')
        self.assertEqual(set(data), {f'{m}_seed{s}' for m in methods for s in range(12)})
        for m in methods:
            for seed in range(12):
                key = f'{m}_seed{seed}'
                row = data[key]
                self.assertEqual((row['method'], row['seed']), (m, seed))
                self.assertEqual(len(row['missing_words']), 500)
                self.assertAlmostEqual(sum(not words for words in row['missing_words'])/500,
                                       row['keyword_satisfaction'])
                if seed < 3:
                    for metric in ('keyword_satisfaction', 'response_nll'):
                        self.assertEqual(row[metric], old[key][metric])
        report, _ = summarize(data, {})
        self.assertFalse(report['success_criterion']['met'])
        for baseline in ('frequency', 'freq_random'):
            stat = report['paired_comparisons'][f'conflict_minus_{baseline}']['satisfaction_pp']
            self.assertEqual(stat['n'], 12)
            self.assertLess(stat['ci95'][1], 2.)

    def test_completed_overlap_audit_and_subset_integrity(self):
        root = Path(__file__).parent
        audit = json.loads((root/'reports/overlap-audit.json').read_text(encoding='utf-8'))
        data = json.loads((root/'results_v2.json').read_text(encoding='utf-8'))
        self.assertEqual(audit['pretraining_train']['status'], 'complete')
        self.assertEqual(audit['pretraining_train']['scanned_records'], 2119719)
        excluded = set(audit['pretraining_train']['eval_indices_zero_based'])
        self.assertEqual(len(excluded), 129)
        self.assertEqual(audit['sft_train']['matched_eval_examples'], 0)
        self.assertEqual(audit['calibration_train']['matched_eval_examples'], 0)
        self.assertEqual(audit['results_alignment_checks']['mismatches'], [])
        for key, row in audit['nonoverlap_subset_lexical_metrics'].items():
            self.assertEqual(row['examples'], 371)
            actual = sum(not data[key]['missing_words'][i] for i in range(500) if i not in excluded)/371
            self.assertAlmostEqual(row['keyword_satisfaction'], actual)

    def test_pairing_uses_shared_seeds_only(self):
        def row(method, seed, sat):
            return dict(method=method, seed=seed, keyword_satisfaction=sat, response_nll=1.)
        data = {'a':row('conflict', 0, .4), 'b':row('conflict', 1, .8),
                'c':row('frequency', 0, .3)}
        report, _ = summarize(data, {})
        comparison = report['paired_comparisons']['conflict_minus_frequency']
        self.assertEqual(comparison['seeds'], [0])
        self.assertAlmostEqual(comparison['satisfaction_pp']['mean'], 10.)

    def test_overlap_normalization_and_record_boundaries(self):
        self.assertEqual(digest('A story.\n Next line.'), digest('a STORY. next  line.'))
        self.assertNotEqual(digest('A story.'), digest('A story!'))
        rows = [{'story_sha256_normalized':digest(s)} for s in ('a', 'b', 'a')]
        self.assertEqual(match_report(rows, {digest('A')})['eval_indices_zero_based'], [0,2])
        self.assertEqual(list(records(['one\n', '<|endoftext|>\n', 'two\n', '<|endoftext|>'])), ['one','two'])


if __name__ == '__main__':
    unittest.main()
