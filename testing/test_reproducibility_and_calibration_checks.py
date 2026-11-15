"""Standalone unit tests for calibration eligibility, selector diagnostics,
provenance guards, and checkpoint metadata consistency.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyze import spearman
from reproducibility import atomic_json, sha256


def canonical_json_hash(obj: object) -> str:
    """Compute canonical JSON SHA-256 hash matching report_results.py."""
    canonical_bytes = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


class ReproducibilityAndCalibrationChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ROOT

    def test_calibration_eligibility_and_budget_rules(self) -> None:
        """Verify calibration eligibility filtering rules:
        - Filters out special tokens (<unk>=0, <eos>=1)
        - Enforces minimum frequency (>= 20)
        - Enforces strictly positive gradient norms (g_in_norm > 0, g_out_norm > 0)
        - Caps selected tokens at min(400, int(0.10 * eligible_count))
        - Confirms exactly 400 selected tokens from 6,341 eligible rows.
        """
        calib_file = self.root / "calibration.json"
        self.assertTrue(calib_file.exists(), f"Missing {calib_file}")

        with open(calib_file, "r", encoding="utf-8") as f:
            calib = json.load(f)

        vocab_size = calib["vocab_size"]
        self.assertEqual(vocab_size, 8000)

        eligible = np.array(calib["eligible"], dtype=bool)
        freq = np.array(calib["freq"], dtype=np.int64)
        g_in_norm = np.array(calib["g_in_norm"], dtype=np.float64)
        g_out_norm = np.array(calib["g_out_norm"], dtype=np.float64)
        selected_ids = calib["selected_token_ids"]

        # 1. Total eligible count verification
        eligible_count = int(eligible.sum())
        self.assertEqual(eligible_count, 6341)
        self.assertEqual(calib["eligible_count"], 6341)

        # 2. Special tokens must never be eligible
        special_token_ids = [0, 1]  # <unk>, <eos>
        for tid in special_token_ids:
            self.assertFalse(eligible[tid], f"Special token {tid} should not be eligible")

        # 3. Frequency threshold: every eligible token has freq >= 20
        min_freq = calib["min_freq"]
        self.assertEqual(min_freq, 20)
        self.assertTrue(np.all(freq[eligible] >= min_freq))

        # Tokens with frequency < 20 must NOT be eligible
        self.assertTrue(np.all(~eligible[freq < min_freq]))

        # 4. Gradient norms: both input and output norm must be strictly positive
        self.assertTrue(np.all(g_in_norm[eligible] > 0))
        self.assertTrue(np.all(g_out_norm[eligible] > 0))

        # 5. Budget rule: capped at min(400, eligible_count // 10)
        max_select_cap = min(calib["max_select"], eligible_count // 10)
        self.assertEqual(max_select_cap, 400)
        self.assertEqual(len(selected_ids), 400)
        self.assertEqual(len(set(selected_ids)), 400)

        # All selected token IDs must be eligible
        self.assertTrue(all(eligible[tid] for tid in selected_ids))

        # Unit verification of filtering logic on synthetic mock data
        mock_freq = np.array([100, 100, 10, 50, 50])  # tokens 0..4
        mock_in_norm = np.array([1.0, 1.0, 1.0, 0.0, 1.0])
        mock_out_norm = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        mock_special = {0}

        synthetic_eligible = np.zeros(5, dtype=bool)
        for i in range(5):
            synthetic_eligible[i] = (
                i not in mock_special
                and mock_freq[i] >= 20
                and mock_in_norm[i] > 0
                and mock_out_norm[i] > 0
            )

        # Only token 1 (freq 100, norms > 0, not special) and token 4 (freq 50, norms > 0)
        self.assertEqual(synthetic_eligible.tolist(), [False, True, False, False, True])

    def test_selector_independence_diagnostics(self) -> None:
        """Verify Spearman correlation and Jaccard overlap helpers return expected
        mathematical bounds, and confirm the near-zero Jaccard overlap (~0.022)
        between top-400 conflict tokens and top-400 frequency tokens.
        """
        def jaccard(a: set[int], b: set[int]) -> float:
            return len(a & b) / max(len(a | b), 1)

        # 1. Bounds & mathematical properties of Spearman rank correlation
        # Perfect positive correlation
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        # Perfect negative correlation
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)
        # Bounded in [-1.0, 1.0]
        rng = np.random.default_rng(42)
        x_rand = rng.normal(size=50)
        y_rand = rng.normal(size=50)
        rho = spearman(x_rand, y_rand)
        self.assertGreaterEqual(rho, -1.0)
        self.assertLessEqual(rho, 1.0)

        # 2. Bounds & mathematical properties of Jaccard similarity
        self.assertEqual(jaccard({1, 2, 3}, {1, 2, 3}), 1.0)
        self.assertEqual(jaccard({1, 2}, {3, 4}), 0.0)
        self.assertAlmostEqual(jaccard({1, 2}, {2, 3}), 1 / 3)

        # 3. Calibration selector diagnostics from calibration.json
        calib_file = self.root / "calibration.json"
        with open(calib_file, "r", encoding="utf-8") as f:
            calib = json.load(f)

        conflict_selected = set(calib["selected_token_ids"])
        eligible = np.array(calib["eligible"], dtype=bool)
        freq = np.array(calib["freq"], dtype=np.float64)

        # Select top-400 frequency tokens among eligible rows
        freq_eligible = freq.copy()
        freq_eligible[~eligible] = -1.0
        top_freq_order = np.argsort(-freq_eligible)[:400].tolist()
        freq_selected = set(top_freq_order)

        # Compute overlap
        intersection_count = len(conflict_selected & freq_selected)
        overlap_jaccard = jaccard(conflict_selected, freq_selected)

        # Exactly 17 shared tokens out of 783 unique tokens -> ~0.0217 (~0.022)
        self.assertEqual(intersection_count, 17)
        self.assertAlmostEqual(overlap_jaccard, 17 / (800 - 17), places=6)
        self.assertAlmostEqual(overlap_jaccard, 0.022, places=3)
        self.assertLess(overlap_jaccard, 0.05)

        # Rank correlation between conflict and frequency among eligible tokens is near zero
        conflict = np.array([np.nan if c is None else c for c in calib["conflict"]])
        valid_mask = eligible & ~np.isnan(conflict)
        rank_corr = spearman(conflict[valid_mask], freq[valid_mask])
        self.assertAlmostEqual(rank_corr, 0.112, places=3)

    def test_provenance_and_atomic_write_guards(self) -> None:
        """Test canonical JSON hashing and atomic file writing on temporary files
        to ensure deterministic hashes across dict key orderings and clean file
        replacement without corrupting existing files.
        """
        # 1. Deterministic canonical JSON hashing across key orderings
        dict_a = {
            "version": 1,
            "metrics": {"satisfaction": 0.366, "nll": 1.5075},
            "name": "tied",
        }
        dict_b = {
            "name": "tied",
            "version": 1,
            "metrics": {"nll": 1.5075, "satisfaction": 0.366},
        }

        hash_a = canonical_json_hash(dict_a)
        hash_b = canonical_json_hash(dict_b)
        self.assertEqual(hash_a, hash_b)

        # Permuting keys produces different non-canonical JSON string, but same canonical hash
        raw_dump_a = json.dumps(dict_a)
        raw_dump_b = json.dumps(dict_b)
        self.assertNotEqual(raw_dump_a, raw_dump_b)

        # 2. Atomic write guards with atomic_json
        with tempfile.TemporaryDirectory() as tmp_dir:
            target_path = Path(tmp_dir) / "sub_dir" / "artifact.json"
            initial_data = {"status": "in_progress", "step": 100}

            # Write file initially
            atomic_json(target_path, initial_data)
            self.assertTrue(target_path.exists())
            self.assertEqual(json.loads(target_path.read_text(encoding="utf-8")), initial_data)

            # Verify no lingering .tmp file
            tmp_candidate = target_path.with_suffix(target_path.suffix + ".tmp")
            self.assertFalse(tmp_candidate.exists())

            # Verify atomic overwrite replaces content cleanly
            updated_data = {"status": "complete", "step": 1000}
            atomic_json(target_path, updated_data)
            self.assertEqual(json.loads(target_path.read_text(encoding="utf-8")), updated_data)
            self.assertFalse(tmp_candidate.exists())

            # Verify failed atomic write (e.g. non-serializable NaN) leaves original file intact
            corrupt_data = {"status": "failed", "value": float("nan")}
            with self.assertRaises(ValueError):
                atomic_json(target_path, corrupt_data)

            # Original data remains intact and uncorrupted
            self.assertEqual(json.loads(target_path.read_text(encoding="utf-8")), updated_data)

    def test_checkpoint_and_report_metadata_consistency(self) -> None:
        """Verify reports/checkpoint-verification.json and reports/main/summary.json
        contain the expected tied parameter count (13,817,856), pretraining step
        (28,503), and SHA-256 hash (b18fabcfc8430769487a3bb96d1e8b83c7eb49510145eba6fecb300e3bd270a0).
        """
        chkpt_meta_file = self.root / "reports" / "checkpoint-verification.json"
        summary_file = self.root / "reports" / "main" / "summary.json"

        self.assertTrue(chkpt_meta_file.exists(), f"Missing {chkpt_meta_file}")
        self.assertTrue(summary_file.exists(), f"Missing {summary_file}")

        with open(chkpt_meta_file, "r", encoding="utf-8") as f:
            chkpt_meta = json.load(f)
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)

        # 1. Checkpoint verification metadata
        expected_sha = "b18fabcfc8430769487a3bb96d1e8b83c7eb49510145eba6fecb300e3bd270a0"
        self.assertEqual(chkpt_meta["checkpoint_sha256"], expected_sha)
        self.assertEqual(chkpt_meta["parameters"], 13817856)
        self.assertEqual(chkpt_meta["step"], 28503)
        self.assertEqual(chkpt_meta["strict_backbone_validation"], "passed")
        self.assertEqual(chkpt_meta["finite_forward_pass"], "passed")

        # 2. Main summary report metadata
        self.assertIn("sources", summary)
        self.assertIn("results_v2.json", summary["sources"])
        r_sources = summary["sources"]["results_v2.json"]
        self.assertIn("file_sha256", r_sources)
        self.assertIn("canonical_json_sha256", r_sources)

        # Check tied baseline in summary matches expected seed count (12 seeds)
        tied_summary = summary["methods"]["tied"]
        self.assertEqual(tied_summary["seeds"], list(range(12)))
        self.assertEqual(tied_summary["satisfaction_percent"]["n"], 12)


if __name__ == "__main__":
    unittest.main()
