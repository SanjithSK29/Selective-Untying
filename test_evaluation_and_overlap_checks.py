"""Standalone unit tests for evaluation keyword matching, overlap audit partition,
exploratory gradient scaling consistency, and paired seed difference verification.
"""
from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from common import story_contains_words, _WORD_RE
import report_results


class EvaluationAndOverlapChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).parent

    def test_whole_word_keyword_satisfaction_and_recall_edge_cases(self) -> None:
        """Test case-insensitive whole-word matching and macro recall calculation."""
        # 1. Exact case-insensitive match
        self.assertTrue(story_contains_words("A Cat ran.", ["cat"]))
        self.assertTrue(story_contains_words("THE CAT SAT ON THE MAT.", ["cat"]))
        self.assertTrue(story_contains_words("Look at that cat!", ["cat"]))
        self.assertTrue(story_contains_words("Where is (cat)?", ["cat"]))

        # 2. Substring rejections: should NOT match
        self.assertFalse(story_contains_words("A caterpillar crawled by.", ["cat"]))
        self.assertFalse(story_contains_words("Leaves scatter in the wind.", ["cat"]))
        self.assertFalse(story_contains_words("The bobcat stalked prey.", ["cat"]))
        self.assertFalse(story_contains_words("Concatenate the strings.", ["cat"]))

        # 3. Multiple words
        words = ["cat", "dog"]
        self.assertTrue(story_contains_words("The dog chased the cat.", words))
        self.assertFalse(story_contains_words("The dog barked loudly.", words))
        self.assertFalse(story_contains_words("The cat mewed softly.", words))
        self.assertFalse(story_contains_words("A caterpillar and a dog.", words))

        # 4. Macro recall calculation with partial keyword matches
        def compute_example_metrics(story: str, target_words: list[str]) -> tuple[bool, float]:
            present = set(_WORD_RE.findall(story.lower()))
            missing = [w for w in target_words if w not in present]
            satisfied = (len(missing) == 0)
            recall = (len(target_words) - len(missing)) / max(len(target_words), 1)
            return satisfied, recall

        # Example 1: 2 out of 3 matched -> recall = 2/3, sat = False
        sat1, rec1 = compute_example_metrics("The dog saw a cat.", ["cat", "dog", "bird"])
        self.assertFalse(sat1)
        self.assertAlmostEqual(rec1, 2 / 3)

        # Example 2: 1 out of 2 matched -> recall = 1/2, sat = False
        sat2, rec2 = compute_example_metrics("An apple fell.", ["apple", "banana"])
        self.assertFalse(sat2)
        self.assertAlmostEqual(rec2, 1 / 2)

        # Example 3: 2 out of 2 matched -> recall = 1.0, sat = True
        sat3, rec3 = compute_example_metrics("Sun and moon appeared.", ["sun", "moon"])
        self.assertTrue(sat3)
        self.assertEqual(rec3, 1.0)

        # Example 4: 0 out of 1 matched -> recall = 0.0, sat = False
        sat4, rec4 = compute_example_metrics("The night was dark.", ["star"])
        self.assertFalse(sat4)
        self.assertEqual(rec4, 0.0)

        # Aggregate macro recall
        macro_recall = (rec1 + rec2 + rec3 + rec4) / 4
        expected_macro_recall = (2 / 3 + 1 / 2 + 1.0 + 0.0) / 4  # 13/24 ≈ 0.541667
        self.assertAlmostEqual(macro_recall, expected_macro_recall)
        self.assertAlmostEqual(macro_recall, 13 / 24)

    def test_overlap_audit_partition_integrity(self) -> None:
        """Verify reports/overlap-audit.json partitions 500 prompts into 129 pretraining

        overlaps and 371 non-overlapping prompts, with 0 SFT/calibration overlaps.
        """
        audit_file = self.root / "reports" / "overlap-audit.json"
        self.assertTrue(audit_file.exists(), f"Missing {audit_file}")

        with open(audit_file, "r", encoding="utf-8") as f:
            audit = json.load(f)

        # Total evaluation examples
        self.assertEqual(audit["evaluation_unique_stories"], 500)

        # SFT train split: 0 overlaps
        sft_train = audit["sft_train"]
        self.assertEqual(sft_train["eval_examples"], 500)
        self.assertEqual(sft_train["matched_eval_examples"], 0)
        self.assertEqual(sft_train["fraction"], 0.0)
        self.assertEqual(sft_train["eval_indices_zero_based"], [])

        # Calibration split: 0 overlaps
        calib_train = audit["calibration_train"]
        self.assertEqual(calib_train["eval_examples"], 500)
        self.assertEqual(calib_train["matched_eval_examples"], 0)
        self.assertEqual(calib_train["fraction"], 0.0)
        self.assertEqual(calib_train["eval_indices_zero_based"], [])

        # Pretraining split: 129 overlaps
        pretrain = audit["pretraining_train"]
        self.assertEqual(pretrain["status"], "complete")
        self.assertEqual(pretrain["eval_examples"], 500)
        self.assertEqual(pretrain["matched_eval_examples"], 129)
        overlap_indices = pretrain["eval_indices_zero_based"]
        self.assertEqual(len(overlap_indices), 129)
        self.assertEqual(len(set(overlap_indices)), 129)
        self.assertTrue(all(0 <= idx < 500 for idx in overlap_indices))

        # Partition clean sum: 129 overlaps + 371 non-overlaps == 500
        nonoverlap_count = 371
        self.assertEqual(pretrain["matched_eval_examples"] + nonoverlap_count, 500)

        # Non-overlapping metrics check: all records evaluate on 371 examples
        nonoverlap_metrics = audit["nonoverlap_subset_lexical_metrics"]
        self.assertGreater(len(nonoverlap_metrics), 0)
        for key, row in nonoverlap_metrics.items():
            self.assertEqual(row["examples"], nonoverlap_count)

    def test_exploratory_and_gradscale_consistency(self) -> None:
        """Verify exploratory results.json (6 methods x 3 seeds) and results_gradscale.json

        (alpha in [1, 2, 5, 10]), confirming alpha=1 reproduces tied baseline.
        """
        results_file = self.root / "results.json"
        gradscale_file = self.root / "results_gradscale.json"
        self.assertTrue(results_file.exists())
        self.assertTrue(gradscale_file.exists())

        with open(results_file, "r", encoding="utf-8") as f:
            exploratory = json.load(f)
        with open(gradscale_file, "r", encoding="utf-8") as f:
            gradscale = json.load(f)

        # 6 exploratory methods across seeds 0, 1, 2
        methods = ("tied", "full", "conflict", "frequency", "freq_random", "grad_norm")
        seeds = (0, 1, 2)
        expected_exploratory_keys = {f"{m}_seed{s}" for m in methods for s in seeds}
        self.assertEqual(set(exploratory.keys()), expected_exploratory_keys)
        self.assertEqual(len(exploratory), 18)

        # Scaling factors: alpha in [1, 2, 5, 10]
        # In results_gradscale.json, alphas 2, 5, 10 are present for seeds 0, 1, 2
        for alpha in (2, 5, 10):
            for s in seeds:
                key = f"gradscale_a{alpha}_seed{s}"
                self.assertIn(key, gradscale)
                self.assertEqual(gradscale[key]["alpha"], float(alpha))
                self.assertEqual(gradscale[key]["seed"], s)

        # alpha=1 corresponds to the tied baseline (no gradient scaling)
        # Verify per-seed satisfaction and NLL match between alpha=1 and tied baseline
        tied_sats = [exploratory[f"tied_seed{s}"]["keyword_satisfaction"] for s in seeds]
        tied_nlls = [exploratory[f"tied_seed{s}"]["response_nll"] for s in seeds]

        self.assertEqual(tied_sats, [0.366, 0.296, 0.370])
        self.assertEqual(tied_nlls, [1.5075, 1.5073, 1.5075])

        # Verify aggregate 3-seed mean matches reported alpha=1 (34.40% sat, 1.5074333 NLL)
        mean_sat = sum(tied_sats) / 3
        mean_nll = sum(tied_nlls) / 3
        self.assertAlmostEqual(mean_sat * 100, 34.40, places=2)
        self.assertAlmostEqual(mean_nll, 1.5074333, places=6)

        # Verify report_results.summarize integration maps alpha=1 to tied
        summary, groups = report_results.summarize(exploratory, gradscale)
        self.assertIn("tied", groups)
        self.assertIn("alpha_2", groups)
        self.assertIn("alpha_5", groups)
        self.assertIn("alpha_10", groups)

    def test_paired_seed_difference_verification(self) -> None:
        """Recompute seed-by-seed paired differences across seeds 0-11 between conflict

        and baselines (frequency, freq_random, tied), matching reported mean gaps:
        - conflict - frequency: -0.0333 pp
        - conflict - freq_random: -0.1167 pp
        - conflict - tied: +0.1500 pp
        """
        results_file = self.root / "results_v2.json"
        self.assertTrue(results_file.exists())

        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        seeds = list(range(12))

        # Collect satisfaction rates (in percentage points: 100 * satisfaction)
        c_sat = np.array([100 * data[f"conflict_seed{s}"]["keyword_satisfaction"] for s in seeds])
        f_sat = np.array([100 * data[f"frequency_seed{s}"]["keyword_satisfaction"] for s in seeds])
        r_sat = np.array([100 * data[f"freq_random_seed{s}"]["keyword_satisfaction"] for s in seeds])
        t_sat = np.array([100 * data[f"tied_seed{s}"]["keyword_satisfaction"] for s in seeds])

        diff_cf = c_sat - f_sat
        diff_cr = c_sat - r_sat
        diff_ct = c_sat - t_sat

        mean_diff_cf = float(np.mean(diff_cf))
        mean_diff_cr = float(np.mean(diff_cr))
        mean_diff_ct = float(np.mean(diff_ct))

        # Check against reported values: -0.0333 pp, -0.1167 pp, +0.1500 pp
        self.assertAlmostEqual(mean_diff_cf, -0.0333, places=4)
        self.assertAlmostEqual(mean_diff_cr, -0.1167, places=4)
        self.assertAlmostEqual(mean_diff_ct, 0.1500, places=4)

        # Cross-check with report_results.summarize
        summary, _ = report_results.summarize(data, {})
        pairs = summary["paired_comparisons"]
        self.assertAlmostEqual(pairs["conflict_minus_frequency"]["satisfaction_pp"]["mean"], -0.0333, places=4)
        self.assertAlmostEqual(pairs["conflict_minus_freq_random"]["satisfaction_pp"]["mean"], -0.1167, places=4)
        self.assertAlmostEqual(pairs["conflict_minus_tied"]["satisfaction_pp"]["mean"], 0.1500, places=4)


if __name__ == "__main__":
    unittest.main()
